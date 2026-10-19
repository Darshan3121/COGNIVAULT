from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from datetime import datetime
import time
from pypdf import PdfReader
import fitz
import pytesseract
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import tempfile
import os
import re
import hashlib
import uuid


router = APIRouter(prefix="/api")


# ==================================================
# EMBEDDING MODEL
# ==================================================

embedding_model = HuggingFaceEmbeddings(
    model_name=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
)


# ==================================================
# CHROMADB
# ==================================================

vector_db = Chroma(
    collection_name=os.environ.get("CHROMA_COLLECTION", "akos_documents_multilingual"),
    embedding_function=embedding_model,
    persist_directory=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "chroma_db")
)


# ==================================================
# TEXT CLEANING
# ==================================================

def clean_text(text: str) -> str:

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ==================================================
# QUESTION-BASED CHUNKING
# ==================================================

def split_into_question_chunks(text: str):

    question_pattern = re.compile(
        r"(?m)(?=^\s*\d+\.\s+)"
    )

    sections = question_pattern.split(text)

    chunks = []

    for section in sections:

        section = section.strip()

        if not section:
            continue

        if len(section) <= 1800:

            chunks.append(section)

        else:

            start = 0
            chunk_size = 1200
            overlap = 150

            while start < len(section):

                end = start + chunk_size

                chunk = section[start:end].strip()

                if chunk:
                    chunks.append(chunk)

                start += chunk_size - overlap

    return chunks


# ==================================================
# THEORY CHUNKING
# ==================================================

def split_theory_text(text: str):

    chunks = []

    chunk_size = 1200
    overlap = 150

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks



# ==================================================
# TASK STATE MANAGEMENT
# ==================================================

TASKS = {}

def update_task(task_id, **kwargs):
    if task_id in TASKS:
        TASKS[task_id].update(kwargs)

@router.get("/upload-status/{task_id}")
def get_upload_status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    return TASKS[task_id]

import unicodedata

def is_extraction_readable(text: str) -> bool:
    if not text or len(text.strip()) < 50:
        return False
        
    if '\x00' in text or '\ufffd' in text:
        return False
        
    text = text.strip()
    total_chars = len(text)
    
    meaningful_chars = 0
    symbols = 0
    
    for c in text:
        cat = unicodedata.category(c)
        if cat.startswith('L') or cat.startswith('N') or cat.startswith('M'):
            meaningful_chars += 1
        elif cat.startswith('P') or cat.startswith('S'):
            symbols += 1
            
    tokens = text.split()
    if not tokens:
        return False
        
    one_char_tokens = sum(1 for t in tokens if len(t) == 1)
    avg_token_len = sum(len(t) for t in tokens) / len(tokens)
    
    meaningful_ratio = meaningful_chars / total_chars
    symbol_ratio = symbols / total_chars
    
    if meaningful_ratio < 0.4:
        return False
        
    if symbol_ratio > 0.4:
        return False
        
    if avg_token_len < 2.5 and (one_char_tokens / len(tokens)) > 0.4:
        return False
        
    if text.count('?') > total_chars * 0.1:
        return False
        
    return True

def process_pdf_background(task_id: str, document_id: str, filename: str, temp_path: str):
    update_task(task_id, status="processing", stage="extracting", progress=0)
    try:
        reader = PdfReader(temp_path)
        text = ""
        total_pages = len(reader.pages)
        native_text_pages = 0
        ocr_pages = 0
        ocr_failed_pages = 0

        fitz_doc = fitz.open(temp_path)

        for page_index, page in enumerate(reader.pages):
            page_number = page_index + 1
            extracted_text = page.extract_text()
            
            is_native_readable = is_extraction_readable(extracted_text)
            
            if is_native_readable:
                print(f"[EXTRACTION] Page {page_number}: native extraction PASS ({len(extracted_text.strip())} chars)")
                native_text_pages += 1
                text += (
                    f"\n[PAGE {page_number}]\n"
                    f"{extracted_text.strip()}\n"
                )
            else:
                print(f"[EXTRACTION] Page {page_number}: native extraction rejected as unreadable")
                update_task(task_id, stage=f"OCR page {page_number}/{total_pages}", progress=int(20 + (page_number/total_pages)*30))
                print(f"[OCR] Page {page_number}/{total_pages}: OCR fallback triggered")
                
                try:
                    fitz_page = fitz_doc.load_page(page_index)
                    mat = fitz.Matrix(4, 4)
                    pix = fitz_page.get_pixmap(matrix=mat)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    try:
                        installed_langs = pytesseract.get_languages()
                        desired_langs = ["eng"]
                        valid_langs = [l for l in desired_langs if l in installed_langs]
                        ocr_lang = "+".join(valid_langs) if valid_langs else "eng"
                    except:
                        ocr_lang = "eng"
                        
                    ocr_text = pytesseract.image_to_string(img, lang=ocr_lang)
                    
                    is_ocr_readable = is_extraction_readable(ocr_text)
                    
                    if is_ocr_readable:
                        print(f"[OCR] Page {page_number}/{total_pages}: OCR PASS ({len(ocr_text.strip())} chars)")
                        ocr_pages += 1
                        text += (
                            f"\n[PAGE {page_number}]\n"
                            f"{ocr_text.strip()}\n"
                        )
                    else:
                        ocr_failed_pages += 1
                        print(f"[OCR] Page {page_number}/{total_pages}: OCR failed text quality heuristics")

                except Exception as e:
                    ocr_failed_pages += 1
                    print(f"ocr_quality=FAIL")
                    print(f"reason=OCR Exception: {e}")

        fitz_doc.close()

        update_task(task_id, stage="chunking", progress=60)
        text = clean_text(text)

        if not text:
            update_task(task_id, status="failed", stage="failed", progress=0, error="No readable text was found in this PDF. The PDF may contain scanned images.")
            return

        question_matches = re.findall(r"(?m)^\s*\d+\.\s+", text)
        question_count = len(question_matches)

        if question_count >= 3:
            chunks = split_into_question_chunks(text)
            chunk_type = "question_based"
        else:
            chunks = split_theory_text(text)
            chunk_type = "theory"

        unique_chunks = []
        seen_hashes = set()
        for chunk in chunks:
            chunk = clean_text(chunk)
            if not chunk: continue
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                unique_chunks.append(chunk)

        chunks = unique_chunks
        ids = []
        for index, chunk in enumerate(chunks):
            base_string = f"{document_id}_{index}_{chunk}"
            chunk_id = hashlib.sha256(base_string.encode("utf-8")).hexdigest()
            ids.append(chunk_id)

        update_task(task_id, stage="Generating multilingual embeddings...", progress=70)
        
        if chunks:
            try:
                vector_db.add_texts(
                    texts=chunks,
                    ids=ids,
                    metadatas=[
                        {
                            "document_id": document_id,
                            "language": "unknown",
                            "filename": filename,
                            "chunk_index": index,
                            "chunk_type": chunk_type
                        }
                        for index in range(len(chunks))
                    ]
                )
            except Exception as e:
                update_task(task_id, status="failed", stage="failed", error=f"ChromaDB persistence failed: {str(e)}")
                return

        extraction_method = "ocr" if ocr_pages > 0 and native_text_pages == 0 else ("mixed" if ocr_pages > 0 else "native")
        
        update_task(task_id, status="completed", stage="completed", progress=100, result={
            "document_id": document_id,
            "filename": filename,
            "total_pages": total_pages,
            "native_text_pages": native_text_pages,
            "ocr_pages": ocr_pages,
            "extraction_method": extraction_method,
            "total_characters": len(text),
            "total_chunks": len(chunks),
            "chunk_type": chunk_type,
            "message": "PDF processed and stored in Cognivault successfully."
        })

    except Exception as e:
        update_task(task_id, status="failed", stage="failed", progress=0, error=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# ==================================================
# PDF UPLOAD
# ==================================================

@router.post("/upload-pdf")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)): 

    # ==================================================
    # STEP 1: VALIDATE FILE
    # ==================================================

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )


    # ==================================================
    # STEP 2: CREATE UNIQUE DOCUMENT ID
    # ==================================================

    document_id = str(uuid.uuid4())

    print("\n--- COGNIVAULT DOCUMENT ---")
    print("Document ID:", document_id)
    print("Filename:", file.filename)


    
    temp_path = None
    task_id = str(uuid.uuid4())
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    TASKS[task_id] = {
        "task_id": task_id,
        "document_id": document_id,
        "filename": file.filename,
        "status": "queued",
        "progress": 0,
        "stage": "queued",
        "created_at": datetime.now().isoformat()
    }

    background_tasks.add_task(process_pdf_background, task_id, document_id, file.filename, temp_path)

    return {
        "task_id": task_id,
        "document_id": document_id,
        "filename": file.filename,
        "status": "queued"
    }

