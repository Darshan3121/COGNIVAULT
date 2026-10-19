import os
from fastapi import APIRouter
from pydantic import BaseModel
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

router = APIRouter(prefix="/api")

embedding_model = HuggingFaceEmbeddings(
    model_name=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
)

vector_db = Chroma(
    collection_name=os.environ.get("CHROMA_COLLECTION", "akos_documents_multilingual"),
    embedding_function=embedding_model,
    persist_directory=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "chroma_db")
)

@router.get("/analytics/{document_id}")
async def get_analytics(document_id: str):
    raw = vector_db._collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"]
    )
    docs = raw.get("documents", [])
    if not docs:
        return {
            "error": "No document found"
        }
        
    num_chunks = len(docs)
    
    all_text = " ".join(docs)
    char_count = len(all_text)
    
    all_text_clean = re.sub(r'http\S+|www.\S+', ' ', all_text)
    all_text_clean = re.sub(r'\S+@\S+', ' ', all_text_clean)
    all_text_clean = re.sub(r'(?<=\b\w)\s+(?=\w\b)', '', all_text_clean)
    all_text_clean = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', all_text_clean)
    
    all_text_lower = all_text_clean.lower()
    all_text_clean = re.sub(r'[^a-z]+', ' ', all_text_lower)
    
    raw_words = all_text_clean.split()
    word_count = len(raw_words)
    
    industry_pillars = {
        'Warehouse Operations': ['warehouse', 'distribution', 'storage', 'fulfillment'],
        'Inventory Management': ['inventory', 'stock', 'replenishment', 'sku', 'cycle count'],
        'Logistics & Routing': ['logistics', 'routing', 'dispatch', 'freight', 'shipping'],
        'Supply Chain Protocol': ['supplychain', 'procurement', 'vendor', 'sourcing'],
        'Staff Training': ['trainedstaff', 'training', 'onboard', 'personnel', 'managestaff'],
        'Operational Efficiency': ['efficiency', 'process', 'improvement', 'lean', 'sixsigma'],
        'Budget Control': ['budget', 'cost', 'expense', 'financial', 'p&l'],
        'Client Relations': ['client', 'customer', 'service', 'account'],
        'Systems Architecture': ['software', 'system', 'technology', 'erp', 'wms'],
        'Data Analytics': ['analysis', 'data', 'metrics', 'reporting', 'kpi'],
        'Programming & Software': ['programming', 'software', 'algorithm', 'code', 'syntax', 'function'],
        'Machine Learning': ['ai', 'machine learning', 'neural network', 'deep learning', 'model']
    }
    
    found_insights = []
    for insight, keywords in industry_pillars.items():
        for kw in keywords:
            if kw in all_text_clean:
                found_insights.append(insight)
                break
                
    top_insights = list(set(found_insights))[:10]
    
    return {
        "word_count": word_count,
        "char_count": char_count,
        "num_chunks": num_chunks,
        "top_insights": top_insights
    }

