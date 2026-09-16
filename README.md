# Document Extractor

Structured information extraction from documents (such as power of attorney documents and similar files), including parties and metadata, using a Large Language Model (LLM).

## Installation

```bash
pip install -r requirements.txt
```

## Configuration (Required Before Running)

Before running the application, you must create the following files locally. These files contain sensitive information and are **not committed to the repository**.

### `.env`

Create a local copy of `.env.example` and enter your actual API credentials:

```bash
cp .env.example .env
```

### `extraction_config.json`

Create a local copy of `extraction_config.example.json` and configure it according to your requirements.

## Usage

```bash
python app.py
```

## Folder Structure

```text
input/     ← Input PDF files (add your files here)
output/    ← Generated CSV files and logs (created automatically)
```
