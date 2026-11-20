import json
import boto3
import pdfplumber
import os
from tqdm import tqdm
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

s3 = boto3.client('s3')
bucket = 'aws-rag-docs'
prefix = ''
local_dir = './pdfs'

def download_pdfs_from_s3(bucket, prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    downloaded_paths = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
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

def is_toc_page(page_text):
    """Flags a whole page as table-of-contents if dot-leaders dominate it."""
    dot_lines = len(re.findall(r'\.{4,}\s*\d+', page_text))
    total_lines = max(page_text.count("\n"), 1)
    return dot_lines / total_lines > 0.3

def clean_page_text(page_text):
    """Strips copyright/legal/version-notice boilerplate lines from a page."""
    patterns = [
        r'Copyright © .*? All rights reserved\.?',
        r'Note[:\s]+This document describes.*?API\.?',
        r"Amazon's trademarks and trade dress may not be used.*?sponsored by Amazon\.?",
        r'^\s*Amazon Web Services, Inc\. or its affiliates.*$',
    ]
    for p in patterns:
        page_text = re.sub(p, '', page_text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return page_text.strip()

def extract_txt(pdf, max_pages=None, verbose=False):
    text = []
    with pdfplumber.open(pdf) as pdf:
        pages = pdf.pages
        if max_pages is not None:
            pages = pages[:max_pages]

        for i, page in enumerate(
            tqdm(pages, desc="Extracting text from PDF", unit="page", leave=False), 1
        ):
            page_txt = page.extract_text()

            if verbose:
                print(f"\n{'='*20} PAGE {i} (raw) {'='*20}")
                print(page_txt if page_txt else "[No text extracted]")

            if not page_txt:
                continue

            if is_toc_page(page_txt):
                if verbose:
                    print(f"[Page {i} SKIPPED: detected as TOC]")
                continue

            cleaned = clean_page_text(page_txt)

            if verbose:
                print(f"{'-'*20} PAGE {i} (after cleaning) {'-'*20}")
                print(cleaned if cleaned else "[Empty after cleaning — page was pure boilerplate]")
                print(f"{'='*55}")

            if cleaned:
                text.append(cleaned)

    full_text = "\n".join(text)
    return full_text

def chunk_text(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    pdf_folder = 'pdfs'

    # Stage A: download from S3 (skip if PDFs are already local in pdf_folder)
    # if not os.path.isdir(pdf_folder) or not os.listdir(pdf_folder):
    #     print("Downloading PDFs from S3...")
    #     download_pdfs_from_s3(bucket, prefix, pdf_folder)

    pdf_files = sorted(
        os.path.join(pdf_folder, f)
        for f in os.listdir(pdf_folder)
        if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        print("No PDFs found.")
    else:
        all_chunks = []

        for pdf_path in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
            filename = os.path.basename(pdf_path)

            # Stage B: extract + clean (TOC removed, copyright/boilerplate stripped)
            text = extract_txt(pdf_path, verbose=False)

            if not text.strip():
                tqdm.write(f"  [skip] No usable text extracted from {filename}")
                continue

            # Stage C: chunk
            chunks = chunk_text(text, chunk_size=500, overlap=50)
            tqdm.write(f"  {filename}: {len(text)} chars -> {len(chunks)} chunks")

            for chunk in chunks:
                all_chunks.append({"source": filename, "text": chunk})

        print(f"\nTotal chunks across all documents: {len(all_chunks)}")

        # persist chunks — checkpoint for the embedding stage
        output_path = "all_chunks.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(all_chunks)} chunks to {output_path}")
        print("Ready for embedding (Stage D).")