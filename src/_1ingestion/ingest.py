import json
import boto3
import pdfplumber
import os
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

s3 = boto3.client('s3')
bucket = 'aws-rag-docs'
prefix = ''
local_dir = './pdfs'

def download_pdfs_from_s3(bucket, prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    downloaded_paths = []

    for page in paginator.paginate(Bucket="aws-rag-docs", Prefix=''):
        for obj in page.get("Contents", []):
            key = obj['Key']
            if not key.endswith('.pdf'):
                continue

            filename = os.path.basename(key)
            local_path = os.path.join(local_dir, filename)

            print(f"Downloading s3://{bucket}/{key} to {local_path}")
            s3.download_file(bucket, key, local_path)
            downloaded_paths.append(local_path)

        if not downloaded_paths:
            print("No PDFs found in the bucket")

        return downloaded_paths

def extract_txt(pdf):
    text = []
    with pdfplumber.open(pdf) as pdf:
        pages = pdf.pages
        for page in tqdm(pages, desc="Extracting text from PDF", unit="page", leave=False):
            page_txt = page.extract_text()
            if page_txt:
                text.append(page_txt)
    
    full_text = "\n".join(text)
    return full_text

def chunk_text(text, chunk_size = 700, overlap = 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = overlap,
        separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    print("Starting PDF ingestion...")
    # pdf_paths = download_pdfs_from_s3(bucket, prefix, local_dir)
    # print(f"Downloaded {len(pdf_paths)} PDFs")

    pdf_folder = 'pdfs'

    pdf_files = [
        os.path.join(pdf_folder, filename)
        for filename in os.listdir(pdf_folder)
        if filename.lower().endswith(".pdf")
    ]

    all_chunks = []

    for pdf in tqdm(pdf_files, desc="Processing Pdfs", unit="file"):
        filename = os.path.basename(pdf)
        print(f"Processing {filename}")

        pdf_path = os.path.join(pdf_folder, filename)
        text = extract_txt(pdf_path)
        print(f"Extracted {len(text)} characters from {filename}")

        chunks = chunk_text(text)
        print(f"Chunked into {len(chunks)} chunks")

        for chunk in chunks:
            all_chunks.append({"source": filename, "text": chunk})
    
    print(f"Total chunks across all PDFs: {len(all_chunks)}")

    if all_chunks:
        print("\n--- Sample chunk (first one) ---")
        print(f"Source: {all_chunks[0]['source']}")
        print(all_chunks[0]["text"][:500])
        print("--- end sample ---")
    
    output_file = "all_chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_chunks)} chunks to {output_file}") 