# Vekalatname Extractor

استخراج ساخت‌یافته اطلاعات (طرفین، متادیتا) از اسناد رسمی (وکالت‌نامه و مشابه) با استفاده از یک مدل زبانی.

## نصب

```bash
pip install -r requirements.txt
```

## پیکربندی (قبل از اجرا الزامی است)

این ریپازیتوری به‌صورت عمدی **بدون** اطلاعات حساس و بدون Schema واقعی منتشر شده. قبل از اجرا باید دو فایل را خودتان محلی بسازید (هیچ‌کدام commit نمی‌شوند):

1. **`.env`** — یک کپی از `.env.example` بسازید و مقادیر واقعی API را وارد کنید:
   ```bash
   cp .env.example .env
   ```

2. **`extraction_config.json`** — یک کپی از `extraction_config.example.json` بسازید و Schema و پرامپت واقعی خودتان را در آن قرار دهید:
   ```bash
   cp extraction_config.example.json extraction_config.json
   ```
   ساختار فایل:
   ```json
   {
     "system_prompt_template": "متن پرامپت شما، با یک {response_format} برای جای‌گذاری خودکار Schema",
     "response_format": { "...": "Schema واقعی JSON Schema شما اینجا" }
   }
   ```

## اجرا

```bash
python app.py
```

## ساختار پوشه‌ها

```
input/   ← فایل‌های PDF ورودی (خودتان اضافه کنید)
output/               ← خروجی CSV و لاگ‌ها (به‌صورت خودکار ساخته می‌شود)
```

> **نکته امنیتی:** فایل‌های `.env` و `extraction_config.json` در `.gitignore` قرار دارند و هرگز نباید commit شوند، چون شامل کلید API و منطق تجاری (Schema استخراج) هستند.
