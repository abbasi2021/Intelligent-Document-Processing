import asyncio
import json
import logging
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Tuple, Optional
import traceback
import fitz  # PyMuPDF
import re
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

# ==================== پیکربندی ====================
# بارگذاری متغیرهای محیطی از فایل .env (این فایل commit نمی‌شود)
load_dotenv()

api_base_url = os.environ["API_BASE_URL"]
api_key = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "tmg-chat-gemma-27b")

# مسیر فایل Schema/Prompt واقعی -> این فایل هم در .gitignore است و هرگز commit نمی‌شود
EXTRACTION_CONFIG_PATH = os.environ.get("EXTRACTION_CONFIG_PATH", "extraction_config.json")

if not os.path.exists(EXTRACTION_CONFIG_PATH):
    raise FileNotFoundError(
        f"فایل پیکربندی '{EXTRACTION_CONFIG_PATH}' یافت نشد. "
        f"از روی extraction_config.example.json یک نسخه محلی بسازید و آن را با Schema واقعی خود پر کنید."
    )

with open(EXTRACTION_CONFIG_PATH, "r", encoding="utf-8") as _f:
    EXTRACTION_CONFIG = json.load(_f)

RESPONSE_FORMAT = EXTRACTION_CONFIG["response_format"]
SYSTEM_PROMPT_TEMPLATE = EXTRACTION_CONFIG["system_prompt_template"]

http_client= httpx.AsyncClient(verify=False)
client= AsyncOpenAI(
    base_url=api_base_url,
    api_key=api_key,
    http_client=http_client
)

BASE_INPUT_DIR =r"input"
BASE_OUTPUT_DIR =r"output"
STATE_DIR =r"output\state"


# تنظیمات پردازش
BATCH_SIZE =100
MAX_CONCURRENT_REQUESTS =1
MAX_RETRIES = 3

# ایجاد دایرکتوری‌ها
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

# ==================== پیکربندی لاگ ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(STATE_DIR, 'processing.log'), encoding='utf-8')
    ]
)
logger = logging.getLogger()

# ==================== توابع کمکی ====================
def reverse_variable(value: str) -> str:
    """معکوس کردن اعداد فارسی که از fitz استخراج شده‌اند"""
    if pd.isna(value) or value is None or value == '' or value == 'None':
        return ''
    
    value_str = str(value).strip()
    if not value_str:
        return ''
    
    # فقط اعداد را معکوس می‌کنیم
    # اگر رشته شامل فقط اعداد فارسی/انگلیسی است
    # persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    english_digits = '0123456789'
    
    is_numeric = all(c in  english_digits for c in value_str.replace(' ', ''))
    # is_numeric = all(c in persian_digits + english_digits for c in value_str.replace(' ', ''))
    
    if is_numeric and len(value_str) > 0:
        return value_str[::-1]
    
    return value_str


def extract_text_from_pdf_fitz(pdf_path: str) -> str:
    """استخراج متن از PDF با استفاده از fitz (PyMuPDF)"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += page.get_text() + "\n"
        
        doc.close()
        return text.strip()
    
    except Exception as e:
        logger.error(f"خطا در استخراج متن از {pdf_path}: {e}")
        raise


def clean_model_response(response_text: str) -> str:
    """تمیزسازی خروجی مدل و استخراج JSON"""
    try:
        # حذف کاراکترهای اضافی
        clean_text = response_text.replace("'", "").replace("\n", "").strip()
        
        # حذف ```json و ```
        if clean_text.find('json') != -1:
            index = clean_text.find('json')
            clean_text = clean_text[index + 4:]
        
        # حذف ``` در انتها
        clean_text = clean_text.replace('```', '').strip()
        
        # پیدا کردن JSON
        start_index = clean_text.find("{")
        end_index = clean_text.rfind("}") + 1
        
        if start_index == -1 or end_index == 0:
            raise ValueError("JSON معتبر در پاسخ یافت نشد")
        
        json_string = clean_text[start_index:end_index]
        
        # تست پارس کردن JSON
        json.loads(json_string)
        
        return json_string
    
    except Exception as e:
        logger.error(f"خطا در تمیزسازی پاسخ مدل: {e}")
        logger.error(f"متن اصلی: {response_text[:500]}...")
        raise


# ==================== مدیریت وضعیت ====================
class StateManager:
    """مدیریت وضعیت پردازش فایل‌ها"""
    
    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        self.processed_file = os.path.join(state_dir, 'processed_files.txt')
        self.failed_file = os.path.join(state_dir, 'failed_files.txt')
        self.batch_counter_file = os.path.join(state_dir, 'batch_counter.txt')  # اضافه شد
        self.lock = asyncio.Lock()
        
        for file_path in [self.processed_file, self.failed_file]:
            if not os.path.exists(file_path):
                open(file_path, 'w', encoding='utf-8').close()
    
    def load_set(self, file_path: str) -> set:
        """بارگذاری مجموعه فایل‌ها از فایل"""
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f if line.strip())
        return set()
    
    async def save_to_file(self, file_path: str, item: str):
        """ذخیره یک آیتم در فایل"""
        async with self.lock:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f"{item}\n")
    
    async def mark_processed(self, file_path: str):
        """علامت‌گذاری فایل به عنوان پردازش‌شده"""
        await self.save_to_file(self.processed_file, file_path)
        logger.info(f"✅ پردازش شد: {os.path.basename(file_path)}")
    
    async def mark_failed(self, file_path: str):
        """علامت‌گذاری فایل به عنوان ناموفق"""
        await self.save_to_file(self.failed_file, file_path)
        logger.warning(f"❌ ناموفق: {os.path.basename(file_path)}")
    
    def get_processed_files(self) -> set:
        """دریافت لیست فایل‌های پردازش‌شده"""
        return self.load_set(self.processed_file)
    
    def get_failed_files(self) -> set:
        """دریافت لیست فایل‌های ناموفق"""
        return self.load_set(self.failed_file)
    
    def get_next_batch_number(self) -> int:
        """دریافت شماره batch بعدی (اضافه شد)"""
        if os.path.exists(self.batch_counter_file):
            with open(self.batch_counter_file, 'r') as f:
                try:
                    return int(f.read().strip()) + 1
                except:
                    return 1
        return 1
    
    def update_batch_counter(self, batch_num: int):
        """به‌روزرسانی شمارنده batch (اضافه شد)"""
        with open(self.batch_counter_file, 'w') as f:
            f.write(str(batch_num))


# ==================== پردازشگر PDF ====================
class PDFProcessor:
    """پردازش فایل‌های PDF با مدل Gemma"""
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def process_with_model(self, text: str, pdf_name: str) -> str:
        """پردازش متن با مدل Gemma و دریافت پاسخ خام"""
        try:
            # loop = asyncio.get_event_loop()

            # Schema و prompt واقعی از فایل بیرونی extraction_config.json خوانده می‌شوند
            # (این فایل commit نمی‌شود - به .gitignore نگاه کنید)
            response_format = RESPONSE_FORMAT
            prompt = SYSTEM_PROMPT_TEMPLATE.format(response_format=response_format)

            # response = await loop.run_in_executor(
            response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": prompt
                        },
                        {
                            "role": "user",
                            "content": f"از متن زیر اطلاعات وکالتنامه را استخراج کن:\n\n{text}"
                        }
                    ],
                    temperature=0.2
                )
            
            
            raw_response = response.choices[0].message.content
            logger.info(f"✅ پاسخ خام دریافت شد برای {pdf_name}")
            
            return raw_response
            
        except Exception as e:
            logger.error(f"خطا در فراخوانی مدل برای {pdf_name}: {e}")
            raise
    
    def parse_and_structure_response(self, raw_response: str, pdf_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """پارس و ساختاردهی پاسخ مدل"""
        try:
            # تمیزسازی و استخراج JSON
            json_string = clean_model_response(raw_response)
            data = json.loads(json_string)
            


            metadata=data.get('metadata',{})
            # استخراج شماره سند
            doc_number =metadata.get('document_number', '')
            doc_number = reverse_variable(doc_number)
            metadata["document_number"]=doc_number
            # is_power_of_attorney=metadata.get('is_power_of_attorney',False)

            # پردازش metadata
            categorization_data = metadata.copy()
            df_categorization = pd.DataFrame([categorization_data]) if categorization_data else pd.DataFrame()
            if not df_categorization.empty:
                df_categorization['pdf_name'] = pdf_name
            

                    


            df_parties=pd.DataFrame()

            if data.get('parties') is not None:
                parties= data.get('parties', {})
            # پردازش اطلاعات وکلا
                lawyers_data =parties.get('first_party', [])
                df_lawyers = pd.DataFrame(lawyers_data) if lawyers_data else pd.DataFrame()

                # پردازش اطلاعات موکلین
                clients_data = parties.get('second_party', [])
                df_clients = pd.DataFrame(clients_data) if clients_data else pd.DataFrame()

                
                # ترکیب وکلا و موکلین
                if not df_lawyers.empty or not df_clients.empty:
                    data_combined = pd.concat([df_lawyers, df_clients], axis=0, ignore_index=True)
                    data_combined['document_number'] = doc_number
                    data_combined['pdf_name'] = pdf_name
                    
                    # معکوس کردن اعداد
                    if 'national_id' in data_combined.columns:
                        data_combined['national_id'] = "'"+ data_combined['national_id'].astype(str).apply(reverse_variable)
                    if 'phone' in data_combined.columns:
                        data_combined['phone'] = "'"+ data_combined['phone'].astype(str).apply(reverse_variable)
                    if 'company_national_id' in data_combined.columns:
                        data_combined['company_national_id'] = "'"+ data_combined['company_national_id'].astype(str).apply(reverse_variable)
                    if 'manager_national_id' in data_combined.columns:
                        data_combined['manager_national_id'] ="'"+  data_combined['manager_national_id'].astype(str).apply(reverse_variable)
                    if 'postal_code' in data_combined.columns:
                        data_combined['postal_code'] ="'"+  data_combined['postal_code'].astype(str).apply(reverse_variable)


                
            else:
                data_combined = pd.DataFrame()
            

            
            logger.info(f"✅ پارس موفق برای {pdf_name}")
            return data_combined, df_categorization, doc_number
            
        except Exception as e:
            logger.error(f"خطا در پارس پاسخ برای {pdf_name}: {e}")
            raise
            
    async def process_single_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """پردازش یک فایل PDF"""
        async with self.semaphore:
            retry_count = 0
            
            while retry_count < MAX_RETRIES:
                try:
                    # استخراج متن با fitz
                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(None, extract_text_from_pdf_fitz, pdf_path)
                    
                    if not text or len(text) < 50:
                        raise ValueError("متن استخراج‌شده خالی یا کوتاه است")
                    
                    # پردازش با مدل
                    pdf_name = os.path.basename(pdf_path)
                    raw_response = await self.process_with_model(text, pdf_name)
                    
                    # پارس و ساختاردهی
                    df_parties, df_metadata, doc_number = self.parse_and_structure_response(
                        raw_response, pdf_name
                    )
                    
                    result = {
                        'pdf_path': pdf_path,
                        'pdf_name': pdf_name,
                        'df_parties': df_parties,
                        'df_metadata': df_metadata,
                        'doc_number': doc_number,
                        'status': 'success',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    logger.info(f"✅ پردازش کامل: {pdf_name}")
                    return pdf_path, result
                    
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"⚠️ تلاش {retry_count}/{MAX_RETRIES} برای {os.path.basename(pdf_path)}: {e}")
                    
                    if retry_count < MAX_RETRIES:
                        await asyncio.sleep(2 ** retry_count)
                    else:
                        logger.error(f"❌ شکست کامل: {os.path.basename(pdf_path)}")
                        await self.state_manager.mark_failed(pdf_path)
                        raise


# ==================== مدیریت دسته‌ها ====================
class BatchManager:
    """مدیریت دسته‌بندی و پردازش فایل‌ها"""
    
    def __init__(self, base_input_dir: str, base_output_dir: str, 
                 state_manager: StateManager, batch_size: int = 100):
        self.base_input_dir = base_input_dir
        self.base_output_dir = base_output_dir
        self.state_manager = state_manager
        self.batch_size = batch_size
        self.processor = PDFProcessor(state_manager)
    
    def get_all_pdf_files(self) -> List[str]:
        """دریافت لیست تمام فایل‌های PDF"""
        pdf_files = []
        for root, dirs, files in os.walk(self.base_input_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        return sorted(pdf_files)
    
    def create_batches(self, pdf_files: List[str]) -> List[List[str]]:
        """ایجاد دسته‌های فایل"""
        batches = []
        for i in range(0, len(pdf_files), self.batch_size):
            batch = pdf_files[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def get_batch_output_paths(self, batch_num: int) -> Tuple[str, str]:
        """دریافت مسیرهای فایل‌های خروجی برای هر دسته"""
        batch_dir = os.path.join(self.base_output_dir, f'batch_{batch_num:04d}')
        os.makedirs(batch_dir, exist_ok=True)
        
        parties_path = os.path.join(batch_dir, f'parties_batch_{batch_num:04d}.csv')
        categorization_path = os.path.join(batch_dir, f'metadata_batch_{batch_num:04d}.csv')
        
        return parties_path, categorization_path
    
    async def save_single_file_result(self, df_parties: pd.DataFrame, 
                                      df_categorization: pd.DataFrame,
                                      batch_num: int):
        """ذخیره فوری نتایج یک فایل (اضافه شد - ذخیره بعد از هر فایل)"""
        parties_path, categorization_path = self.get_batch_output_paths(batch_num)
        
        # ذخیره اطلاعات طرفین (اگر موجود باشد)
        if not df_parties.empty:
            # تبدیل به رشته
            df_parties_copy = df_parties.copy()
            df_parties_copy = df_parties_copy.astype(str)
            
            # اطمینان از اینکه document_number به صورت متن ذخیره شود
            if 'document_number' in df_parties_copy.columns:
                df_parties_copy['document_number'] = "'" + df_parties_copy['document_number'].astype(str)
            
            # ذخیره (append mode)
            if os.path.exists(parties_path):
                df_parties_copy.to_csv(parties_path, mode='a', header=False, sep= ',',
                                      index=False, encoding='utf-8-sig')
            else:
                df_parties_copy.to_csv(parties_path, mode='w', header=True, sep= ',',
                                      index=False, encoding='utf-8-sig')
        else:
            logger.debug(f"parties خالی")
        
        # ذخیره اطلاعات دسته‌بندی (اگر موجود باشد)
        if not df_categorization.empty:
            # تبدیل به رشته
            df_categorization_copy = df_categorization.copy()
            df_categorization_copy = df_categorization_copy.astype(str)
            
            # اطمینان از اینکه document_number به صورت متن ذخیره شود
            if 'document_number' in df_categorization_copy.columns:
                df_categorization_copy['document_number'] = "'" + df_categorization_copy['document_number'].astype(str)
            
            # ذخیره (append mode)
            if os.path.exists(categorization_path):
                df_categorization_copy.to_csv(categorization_path, mode='a', header=False,sep= ',',
                                             index=False, encoding='utf-8-sig')
            else:
                df_categorization_copy.to_csv(categorization_path, mode='w', header=True,sep= ',',
                                             index=False, encoding='utf-8-sig')
        else:
            logger.error(f"metadata خالی است ")
        



    async def process_batch(self, batch: List[str], batch_num: int):
        """پردازش یک دسته از فایل‌ها (ذخیره فوری بعد از هر فایل)"""
        batch_start_time = datetime.now()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 شروع پردازش دسته {batch_num} با {len(batch)} فایل")
        logger.info(f"⏰ زمان شروع: {batch_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        # فیلتر کردن فایل‌های پردازش‌شده
        processed = self.state_manager.get_processed_files()
        pending_files = [f for f in batch if f not in processed]
        
        if not pending_files:
            logger.info(f"⏭️ تمام فایل‌های دسته {batch_num} قبلاً پردازش شده‌اند")
            return
        
        logger.info(f"📋 {len(pending_files)} فایل باقی‌مانده برای پردازش")
        logger.info(f"🚀 شروع پردازش موازی {len(pending_files)} فایل...")
        
        # شمارنده برای آمار
        saved_count = 0
        failed_count = 0
        
        # ⬇️ تغییر اصلی: ایجاد taskها
        tasks = {
            asyncio.create_task(self.processor.process_single_pdf(pdf)): pdf 
            for pdf in pending_files
        }
        
        # ⬇️ پردازش به محض تمام شدن هر فایل
        for completed_task in asyncio.as_completed(tasks.keys()):
            try:
                pdf_path_str, result_dict = await completed_task
                
                # ذخیره فوری نتایج این فایل
                try:
                    await self.save_single_file_result(
                        result_dict['df_parties'],
                        result_dict['df_metadata'],
                        batch_num
                    )
                    
                    # علامت‌گذاری به عنوان پردازش‌شده
                    await self.state_manager.mark_processed(pdf_path_str)
                    saved_count += 1
                    
                    logger.info(f"💾 {saved_count}/{len(pending_files)} - {os.path.basename(pdf_path_str)} ذخیره شد")
                
                except Exception as e:
                    logger.error(f"❌ خطا در ذخیره {os.path.basename(pdf_path_str)}: {e}")
                    await self.state_manager.mark_failed(pdf_path_str)
                    failed_count += 1
            
            except Exception as e:
                # پیدا کردن فایلی که خطا داده
                failed_pdf = tasks[completed_task]
                logger.error(f"❌ خطا در پردازش {os.path.basename(failed_pdf)}: {e}")
                await self.state_manager.mark_failed(failed_pdf)
                failed_count += 1
        
        logger.info(f"✅ پردازش موازی تمام شد")
        
        # به‌روزرسانی شمارنده batch
        self.state_manager.update_batch_counter(batch_num)
        
        # پایان زمان‌سنجی و لاگ
        batch_end_time = datetime.now()
        batch_duration = (batch_end_time - batch_start_time).total_seconds()
        
        logger.info(f"✅ دسته {batch_num} کامل شد")
        logger.info(f"📊 آمار: {saved_count} موفق، {failed_count} ناموفق از {len(pending_files)} فایل")
        logger.info(f"⏰ زمان پایان: {batch_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏱️ مدت زمان اجرای دسته {batch_num}: {batch_duration:.2f} ثانیه ({batch_duration/60:.2f} دقیقه)")
        logger.info(f"{'='*60}\n")
    
    async def process_all_batches(self):
        """پردازش تمام دسته‌ها (اصلاح شد - شماره‌گذاری صحیح batch)"""
        # دریافت تمام فایل‌ها
        all_pdfs = self.get_all_pdf_files()
        logger.info(f"📚 تعداد کل فایل‌های PDF: {len(all_pdfs)}")
        
        # فیلتر کردن فایل‌های پردازش‌شده
        processed = self.state_manager.get_processed_files()
        pending_pdfs = [f for f in all_pdfs if f not in processed]
        logger.info(f"⏳ فایل‌های باقی‌مانده: {len(pending_pdfs)}")
        
        if not pending_pdfs:
            logger.info("✅ تمام فایل‌ها قبلاً پردازش شده‌اند!")
            return
        
        # ایجاد دسته‌ها
        batches = self.create_batches(pending_pdfs)
        logger.info(f"📦 تعداد دسته‌ها: {len(batches)}")
        
        # دریافت شماره batch شروع (اضافه شد)
        starting_batch_num = self.state_manager.get_next_batch_number()
        logger.info(f"🔢 شماره batch شروع: {starting_batch_num}")
        
        # پردازش هر دسته
        for i, batch in enumerate(batches):
            batch_num = starting_batch_num + i  # اصلاح شد: شماره‌گذاری ادامه‌دار
            try:
                await self.process_batch(batch, batch_num)
            except Exception as e:
                logger.error(f"❌ خطای کلی در دسته {batch_num}: {e}")
                logger.error(traceback.format_exc())
        
        # گزارش نهایی
        self.print_final_report()
    
    def print_final_report(self):
        """چاپ گزارش نهایی"""
        processed = self.state_manager.get_processed_files()
        failed = self.state_manager.get_failed_files()
        all_pdfs = self.get_all_pdf_files()
        
        logger.info("\n" + "="*60)
        logger.info("📊 گزارش نهایی پردازش")
        logger.info("="*60)
        logger.info(f"✅ تعداد فایل‌های پردازش‌شده: {len(processed)}")
        logger.info(f"❌ تعداد فایل‌های ناموفق: {len(failed)}")
        logger.info(f"📚 تعداد کل فایل‌ها: {len(all_pdfs)}")
        logger.info(f"⏳ باقی‌مانده: {len(all_pdfs) - len(processed)}")
        logger.info("="*60 + "\n")

# ==================== تابع اصلی ====================
async def main():
    """تابع اصلی اجرای برنامه"""
    try:
        # ایجاد مدیران
        state_manager = StateManager(STATE_DIR)
        batch_manager = BatchManager(
            base_input_dir=BASE_INPUT_DIR,
            base_output_dir=BASE_OUTPUT_DIR,
            state_manager=state_manager,
            batch_size=BATCH_SIZE
        )
        
        # شروع پردازش
        logger.info("🚀 شروع پردازش فایل‌ها...")
        start_time = datetime.now()
        
        await batch_manager.process_all_batches()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"\n✅ پردازش کامل شد!")
        logger.info(f"⏱️ زمان کل: {duration:.2f} ثانیه ({duration/60:.2f} دقیقه)")
        
    except Exception as e:
        logger.error(f"❌ خطای کلی در برنامه: {e}")
        logger.error(traceback.format_exc())


# ==================== اجرا ====================
if __name__ == "__main__":
    asyncio.run(main())