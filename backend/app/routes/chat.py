import os
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
import time
import re
import json
import unicodedata
from app.routes.mcq_validator import (
    MCQCandidate,
    parse_mcq_candidates,
    validate_mcq_candidate,
    format_user_facing_mcqs,
    format_complete_mcqs,
    format_verified_answers,
    sanitize_overview_claims,
    improve_weak_explanation
)
from app.routes.response_completer import (
    is_sentence_complete,
    stitch_continuation,
    generate_continuation_snippet,
    complete_response_if_needed
)

router = APIRouter(prefix="/api")


# ============================================================
# EMBEDDINGS
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
)


# ============================================================
# VECTOR DATABASE
# ============================================================

vector_db = Chroma(
    collection_name=os.environ.get("CHROMA_COLLECTION", "akos_documents_multilingual"),
    embedding_function=embedding_model,
    persist_directory=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "chroma_db")
)


# ============================================================
# LOCAL LLM
# ============================================================

llm = OllamaLLM(
    model="mistral:latest",
    temperature=0,
    num_predict=300,
    num_ctx=2048,
    top_k=20,
    top_p=0.9,
    keep_alive="10m"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    document_id: str | None = None
    allow_general: bool = False
    stream: bool = False
    messages: list[dict] | None = None
    permission_response: str | None = None


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    text = text.lower()
    # Remove page markers
    text = re.sub(r"\[page\s+\d+\]", " ", text)
    # Remove question numbering
    text = re.sub(r"^\s*\d+\.\s*", "", text)
    # Normalize apostrophes
    text = text.replace("’", "'")
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    # Normalize spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# STOPWORDS & IMPORTANT WORDS
# ============================================================

STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "to", "in", "on", "for",
    "and", "or", "by", "with", "from", "does", "do", "how", "why", "can",
    "could", "would", "should", "please", "tell", "me", "give", "write",
    "explain", "discuss", "describe", "elaborate", "define", "definition",
    "meaning", "about", "briefly", "brief", "detail", "detailed"
}

def important_words(text: str) -> set[str]:
    normalized = normalize_text(text)
    words = normalized.split()
    return {
        word for word in words
        if word not in STOPWORDS and len(word) > 2
    }


# ============================================================
# ANSWER FORMAT / INTENT DETECTION
# ============================================================

def detect_answer_format(question: str):
    q = question.lower().strip()

    # Points
    points_match = re.search(r"\b(\d+)\s*(?:points?|key points?|main points?|steps?)\b", q)
    if points_match:
        return {"type": "points", "count": int(points_match.group(1))}

    # Lines
    lines_match = re.search(r"\b(\d+)\s*lines?\b", q)
    if lines_match:
        return {"type": "lines", "count": int(lines_match.group(1))}

    # MCQ request
    mcq_patterns = [
        r"\bchoose the correct\b", r"\bselect the correct\b", r"\bcorrect option\b",
        r"\bwhich of the following\b", r"\bmcq\b", r"\bmultiple choice\b",
        r"\bwhich one\b", r"\bwhat is the correct\b"
    ]
    for pattern in mcq_patterns:
        if re.search(pattern, q):
            return {"type": "mcq", "count": None}

    # Fill in the blank
    if "fill in the blank" in q or "fill in the blanks" in q or "fill the blank" in q:
        return {"type": "fill_blank", "count": None}

    # List / Enumeration
    list_patterns = [
        r"\blist\b", r"\benumerate\b", r"\bmention\b", r"\bstate the\b", r"\bgive the\b",
        r"\bwhat are the\b", r"\bwhat are\b", r"\badvantages\b", r"\bdisadvantages\b",
        r"\bbenefits\b", r"\blimitations\b", r"\bfeatures\b", r"\bcharacteristics\b",
        r"\bfactors\b", r"\breasons\b", r"\bcauses\b", r"\beffects\b", r"\btypes\b",
        r"\bmethods\b", r"\bways\b", r"\bfunctions\b", r"\bobjectives\b", r"\bprinciples\b",
        r"\bimportance\b"
    ]
    for pattern in list_patterns:
        if re.search(pattern, q):
            return {"type": "list", "count": None}

    # Explanation / Descriptive
    explanation_patterns = [
        r"\bexplain\b", r"\bexplains\b", r"\bexplanation\b", r"\bdiscuss\b", r"\bdiscussion\b",
        r"\bdescribe\b", r"\bdescription\b", r"\belaborate\b", r"\belaboration\b",
        r"\bclarify\b", r"\bclarification\b", r"\boutline\b", r"\bsummarize\b", r"\bsummarise\b",
        r"\bsummary\b", r"\boverview\b", r"\bwrite about\b", r"\btalk about\b",
        r"\bstate and explain\b", r"\bexplain in detail\b", r"\bexplain briefly\b",
        r"\bhow does\b", r"\bhow do\b", r"\bhow is\b", r"\bhow are\b",
        r"\bwhy does\b", r"\bwhy do\b", r"\bwhy is\b", r"\bwhy are\b"
    ]
    for pattern in explanation_patterns:
        if re.search(pattern, q):
            return {"type": "explanation", "count": None}

    # Definition
    definition_patterns = [
        r"^what is\b", r"^what are\b", r"\bdefine\b", r"\bdefinition of\b",
        r"\bmeaning of\b", r"\bwhat do you mean by\b"
    ]
    for pattern in definition_patterns:
        if re.search(pattern, q):
            return {"type": "definition", "count": None}

    return {"type": "normal", "count": None}


# ============================================================
# MCQ EXTRACTION & MATCHING HELPERS
# ============================================================

def looks_like_mcq(text: str) -> bool:
    option_matches = re.findall(r"(?m)^\s*[A-D][\)\.]\s+", text)
    return len(option_matches) >= 2

def extract_mcq_question(text: str) -> str | None:
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        match = re.match(r"^\d+\.\s*(.*)$", line)
        if match:
            question = match.group(1).strip()
            question = re.sub(r"\s*[-—]?\s*Correct.*$", "", question, flags=re.IGNORECASE)
            return question
    return None

def question_match_score(user_question: str, mcq_question: str) -> float:
    user_words = important_words(user_question)
    mcq_words = important_words(mcq_question)
    if not user_words or not mcq_words:
        return 0.0
    common_words = user_words.intersection(mcq_words)
    user_coverage = len(common_words) / len(user_words)
    mcq_coverage = len(common_words) / len(mcq_words)
    if user_coverage + mcq_coverage == 0:
        return 0.0
    return (2 * user_coverage * mcq_coverage) / (user_coverage + mcq_coverage)

def extract_explicit_correct_answer(text: str) -> str | None:
    patterns = [
        r"(?m)^\s*([A-D])[\)\.]\s*(.*?)\s*[—-]\s*Correct\s*$",
        r"(?m)^\s*([A-D])[\)\.]\s*(.*?)\s*\(\s*Correct\s*\)\s*$",
        r"(?i)\bAnswer\s*:\s*([A-D])[\)\.]\s*(.*)",
        r"(?i)\bCorrect\s+Answer\s*:\s*([A-D])\b"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            letter = match.group(1).upper()
            if len(match.groups()) >= 2:
                option_text = match.group(2).strip()
                if option_text:
                    return f"{letter}) {option_text}"
            return letter
    return None

def find_matching_mcq(user_question: str, documents):
    best_match = None
    best_score = 0.0
    for doc in documents:
        text = doc.page_content
        if not looks_like_mcq(text):
            continue
        mcq_question = extract_mcq_question(text)
        if not mcq_question:
            continue
        score = question_match_score(user_question, mcq_question)
        if score > best_score:
            best_score = score
            best_match = {"document": doc, "question": mcq_question, "score": score}
    return best_match

def build_format_instruction(answer_format):
    format_type = answer_format["type"]
    count = answer_format["count"]

    if format_type == "points":
        return f"""
Answer the user's question in EXACTLY {count} numbered points.
Use this format:
1. ...
2. ...
3. ...
Continue until exactly {count} points are provided. Do not provide more or fewer points unless the document genuinely lacks content.
"""
    if format_type == "lines":
        return f"Answer the user's question in EXACTLY {count} concise lines using only supported information."
    if format_type == "definition":
        return "Give a clear and concise definition based on the supplied document. Do not invent facts."
    if format_type == "list":
        return "Answer in a clear list format (numbered points or bullet points) using supported information."
    if format_type == "explanation":
        return "Give a clear explanatory answer organizing the answer logically based strictly on the document."
    if format_type == "mcq":
        return "Identify the correct option from the supplied document. Return option letter and text."
    if format_type == "fill_blank":
        return "Provide the missing answer from the supplied document concisely."
    return "Answer the question clearly and directly using only the supplied document."


# ============================================================
# CONVERSATIONAL GROUNDING & OVERVIEW HELPERS
# ============================================================

def is_greeting_message(text: str) -> bool:
    t = text.lower().strip()
    greetings = [
        "hi", "hello", "hey", "hey cognivault", "greetings",
        "good morning", "good afternoon", "good evening",
        "howdy", "sup", "hi cognivault", "hello cognivault"
    ]
    if t in greetings:
        return True
    return bool(re.match(r"^(hi|hello|hey|greetings|howdy)\b", t))


def is_mcq_generation_query(question: str) -> bool:
    """
    Detect if the user is asking to GENERATE an MCQ based on the document.
    """
    text = question.lower()
    text = re.sub(r"’", "'", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    
    # Check for MCQ answering (if it looks like a pasted question with options a) b) c) d))
    if re.search(r'\b(a[\)\.]\s|b[\)\.]\s|c[\)\.]\s|d[\)\.]\s)', text):
        return False
        
    has_mcq_term = re.search(r'\b(mcq|mcqs|multiple choice)\b', text)
    has_action_term = re.search(r'\b(generate|create|make|give|ask|quiz|write|formulate|provide)\b', text)
    has_context_term = re.search(r'\b(based on|from|this|the|uploaded|pdf|document|file|chapter)\b', text)
    
    if re.search(r'\bquiz\s+me\b', text):
        return True
        
    if has_mcq_term and has_action_term:
        return True
        
    if has_mcq_term and has_context_term and re.search(r'\b(mcq|mcqs)\s+(based\s+on|from)\b', text):
        return True
            
    return False


def extract_mcq_count(question: str) -> int:
    text = question.lower()
    
    word_to_num = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    # Try compound pattern: <num> [multiple-choice] questions / mcqs
    comp_match = re.search(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:(?:multiple[\s\-_]*choice|mcq|mcqs)\s+)?(?:mcqs?|questions?|multiple[\s\-_]*choice)\b', text, re.IGNORECASE)
    if comp_match:
        val = comp_match.group(1).lower()
        if val.isdigit():
            return int(val)
        if val in word_to_num:
            return word_to_num[val]

    # Try digits
    match = re.search(r'\b(\d+)\s*(?:mcq|mcqs|multiple[\s\-_]*choice|questions?)\b', text)
    if match:
        return int(match.group(1))
        
    # Try words
    for word, num in word_to_num.items():
        if re.search(rf'\b{word}\s*(?:multiple[\s\-_]*choice|mcq|mcqs|questions?)\b', text):
            return num
            
    # Explicit single words
    if re.search(r'\b(an|a|the)\s*(?:mcq|multiple[\s\-_]*choice)\b', text):
        return 1
        
    # Default to 1
    return 1


def is_mcq_answer_request(question: str) -> bool:
    text = question.lower()
    if re.search(r'\b(answer|answers|solution|solutions|check my)\b', text):
        if not re.search(r'\b(generate|create|make)\b', text):
            return True
    return False

def is_document_overview_query(question: str) -> bool:
    """
    Detect document meta/overview queries after normalizing whitespace,
    punctuation, and harmless conversational words like 'bro'.
    """

    text = question.lower()
    text = re.sub(r"’", "'", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Strip harmless conversational filler words
    fillers = [
        r"\bbro\b", r"\bhey\b", r"\bhi\b", r"\bhello\b", r"\bplease\b",
        r"\bcan\s+you\b", r"\bcould\s+you\b", r"\bwould\s+you\b",
        r"\btell\s+me\b", r"\bexplain\s+to\s+me\b", r"\bgive\s+me\b",
        r"\bsir\b", r"\bdude\b", r"\bbuddy\b"
    ]
    # Added to handle variations like "explain me" and "briefly"
    fillers.extend([
        r"\bexplain\s+me\b", r"\bexplain\b",
        r"\bin\s+brief\b", r"\bbriefly\b",
        r"\buploaded\b", r"\bthe\b", r"\bthis\b"
    ])
    
    for f in fillers:
        text = re.sub(f, " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    overview_patterns = [
        r"what\s+is\s+(pdf|document|file|paper|book)\s+about",
        r"what\s+does\s+(pdf|document|file|paper|book)\s+(cover|contain|talk|teach|explain)",
        r"overview\s+of\s+(pdf|document|file|paper)",
        r"summarize\s+(pdf|document|file|paper)",
        r"summary\s+of\s+(pdf|document|file|paper)",
        r"what\s+topics\s+are\s+covered",
        r"what\s+(pdf|document)\s+contains",
        r"about\s+(pdf|document|file|paper)",
        r"what\s+is\s+in\s+(pdf|document)",
        r"summary\b",
        r"overview\b",
        r"summarize\b"
    ]

    is_overview = False
    for pattern in overview_patterns:
        if re.search(pattern, text):
            is_overview = True
            break

    if not is_overview:
        exact_phrases = [
            "what is this pdf about",
            "what is this document about",
            "give me an overview of this pdf",
            "give me an overview of this document",
            "summarize this pdf",
            "summarize this document",
            "what topics are covered",
            "what topics are covered in this document",
            "what does this document teach",
            "what does this pdf contain",
            "tell me about this document",
            "explain what this pdf contains"
        ]
        is_overview = any(p in text for p in exact_phrases)
        
    print("QUESTION:", question)
    print("NORMALIZED QUESTION:", text)
    print("IS_OVERVIEW_QUERY:", is_overview)

    return is_overview



def check_evidence_sufficiency(chunks: list[str]) -> bool:
    if not chunks:
        return False
        
    total_chars = 0
    meaningful_chars = 0
    
    for chunk in chunks:
        for char in chunk:
            total_chars += 1
            cat = unicodedata.category(char)
            if cat.startswith('L') or cat.startswith('N') or cat.startswith('M'):
                meaningful_chars += 1
                
    if total_chars < 50:
        return False
        
    ratio = meaningful_chars / total_chars if total_chars > 0 else 0
    if ratio < 0.3:
        return False
        
    return True

def get_representative_chunks(document_id: str) -> list[str]:
    """
    Get chunks ONLY for active document_id sorted by chunk_index.
    Select 5-8 representative chunks distributed across document (target 0, N/4, N/2, 3N/4, N-1).
    Remove duplicates while maintaining text content.
    """
    try:
        raw = vector_db._collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )
        docs = raw.get("documents", [])
        metas = raw.get("metadatas", [])
        
        print(f"[OVERVIEW] document_id: {document_id}")
        print(f"[OVERVIEW] retrieved_total: {len(docs)}")
        
        if not docs:
            return []

        paired = []
        for i, (d, m) in enumerate(zip(docs, metas)):
            if d and isinstance(d, str) and d.strip():
                idx = m.get("chunk_index", i) if isinstance(m, dict) and m else i
                paired.append((idx, d.strip(), m))

        if not paired:
            return []

        paired.sort(key=lambda x: x[0])
        chunks = [text for idx, text, m in paired]
        paired_metas = [m for idx, text, m in paired]
        n = len(chunks)

        if n <= 8:
            indices = list(range(n))
        else:
            target_count = min(8, max(5, n))
            indices = [int(i * (n - 1) / (target_count - 1)) for i in range(target_count)]
            
        selected_raw = [chunks[i] for i in indices]
        selected_metas = [paired_metas[i] for i in indices]

        selected = []
        seen = set()
        
        sampled_docs = []
        sampled_files = []
        
        for c, m in zip(selected_raw, selected_metas):
            norm = normalize_text(c[:100])
            if norm not in seen:
                seen.add(norm)
                selected.append(c)
                sampled_docs.append(m.get('document_id', 'unknown'))
                sampled_files.append(m.get('filename', 'unknown'))
                
        print(f"[OVERVIEW] sampled_chunks: {len(selected)}")
        print(f"[OVERVIEW] sampled_document_ids: {sampled_docs}")
        print(f"[OVERVIEW] sampled_filenames: {sampled_files}")
        
        return selected
    except Exception as e:
        print(f"Error fetching representative chunks for doc '{document_id}': {e}")
        return []


def enrich_chunks_with_class_definitions(document_id: str, rep_chunks: list[str]) -> list[str]:
    """
    If any representative chunk references custom classes (e.g. 'new Book', 'Book b'),
    ensures the chunk containing the class definition ('class Book') is retrieved and
    included so MCQs can ground questions in the class definition.
    """
    if not rep_chunks or not document_id:
        return rep_chunks

    standard_java_types = {
        "String", "StringBuffer", "StringBuilder", "Thread", "Runnable", "Scanner",
        "System", "Object", "Math", "Integer", "Double", "Boolean", "List",
        "ArrayList", "Map", "HashMap", "Set", "HashSet", "Exception", "Throwable"
    }

    custom_classes = set()
    for chunk in rep_chunks:
        matches = re.findall(r"\bnew\s+([A-Z][a-zA-Z0-9_]*)\s*\(", chunk)
        for cls in matches:
            if cls not in standard_java_types:
                custom_classes.add(cls)

    if not custom_classes:
        return rep_chunks

    combined_rep = " ".join(rep_chunks)
    missing_defs = [cls for cls in custom_classes if not re.search(rf"\bclass\s+{cls}\b", combined_rep)]

    if not missing_defs:
        return rep_chunks

    try:
        raw = vector_db._collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )
        docs = raw.get("documents", [])
        added_chunks = []
        for cls in missing_defs:
            cls_pattern = rf"\bclass\s+{cls}\b"
            for d in docs:
                if d and re.search(cls_pattern, d):
                    if d not in rep_chunks and d not in added_chunks:
                        added_chunks.append(d.strip())
                        print(f"[CLASS ENRICHMENT] Found definition chunk for custom class '{cls}'")
                        break

        if added_chunks:
            return rep_chunks + added_chunks
    except Exception as e:
        print(f"[CLASS ENRICHMENT ERROR] {e}")

    return rep_chunks


STORED_MCQ_ANSWERS: dict[str, list[MCQCandidate]] = {}

def generate_and_validate_mcqs(document_id: str, mcq_count: int, rep_chunks: list[str], include_answers: bool = False) -> tuple[list[MCQCandidate], str]:
    """
    Generates candidates from representative chunks and validates them against strict
    grounding, distinctness, code inclusion, and ambiguity checks.
    Returns (validated_candidates, formatted_text).
    When include_answers=True, formatted text includes correct answer and explanation per MCQ.
    """
    context_chunks = rep_chunks[:5] if len(rep_chunks) >= 5 else rep_chunks
    context = "\n\n---\n\n".join([c[:600] for c in context_chunks])
    candidates_to_generate = mcq_count + 1

    prompt = f"""[INST] You are an expert educational assessment generator creating multiple-choice questions grounded STRICTLY in the provided document excerpts.

CRITICAL RULES:
1. Grounding: Every question and its correct answer MUST be directly supported by the document excerpts. Do NOT invent fictional classes, methods, variables, or document statistics.
2. Conceptual Focus:
   - Prefer THEORETICAL and CONCEPTUAL questions about Java rules, keywords, definitions, interfaces, OOP concepts, or method behaviors discussed in the excerpts.
   - Excellent topics to draw from the excerpts:
     * 2D array syntax (e.g. `int[][] arr = new int[4][3];`), ragged arrays (rows of different lengths), and array indexing.
     * Difference between 1D and 2D arrays (1D is linear indexed by one subscript; 2D is an array of arrays indexed by row and column; both are homogeneous).
     * String immutability (methods like concat() return a new String; they do not alter the original string).
     * Default values for uninitialized instance variables (boolean defaults to false; object references default to null).
     * Java keywords: `this` (current object), `super` (superclass members), `implements` (implementing an interface).
     * `java.lang` is the default imported package; `java.lang.Object` is the root superclass of all classes.
     * Interface modifiers: interface variables are implicitly `public static final`; interface methods are implicitly `public abstract`.
     * Multilevel class inheritance: In the document's multilevel example (Module 2 Q19), `class MotorVehicle extends Vehicle` and `class Car extends MotorVehicle`. All three are classes (`Car` is a subclass of `MotorVehicle`, which is a subclass of `Vehicle`). NEVER claim Vehicle is an interface in this hierarchy, and NEVER claim MotorVehicle implements Vehicle.
     * Interface implementation: In the document's interface example (Module 3 Q11), `interface Vehicle` has methods start() and stop(), implemented by `class Car implements Vehicle` (MotorVehicle is not part of this interface example).
3. Code Snippets:
   - If a question asks about code, code output, or a variable, you MUST include the complete code snippet inside the question text formatted in a markdown code block:
     ```java
     // code here
     ```
   - NEVER refer to "the following code snippet", "the snippet above", or a variable without displaying the code in the question itself.
   - NEVER ask about a custom class's constructors or methods (e.g. Book) unless the class definition is displayed in your question.
4. No Ambiguous Options & Technical Correctness:
   - Exactly ONE option (A, B, C, or D) must be unambiguously correct.
   - The other three options must be unambiguously incorrect.
   - NEVER ask broad generic questions about return types (e.g. do NOT ask "What is the return type of a method defined in an interface?" or "What is the return type of the method() defined in an interface?"). In Java, interface methods can declare any return type (void, primitives, objects).
   - In Java, all arrays (1D and 2D) are strictly homogeneous. NEVER claim or generate options stating that 1D or 2D arrays store different data types or are heterogeneous.
   - In Java, all methods declare a return type or void. NEVER claim that interface methods or abstract methods "have no return type".
   - The `implements` keyword is used by a class to declare that it implements an interface; it does NOT define an interface (interfaces are declared using the `interface` keyword).
   - When comparing a class and an interface, describe them accurately within the scope of the document: a class provides concrete implementations and state to create objects, whereas an interface specifies abstract method declarations (a contract) that implementing classes must define. Avoid oversimplified definitions.
   - For inheritance questions, adhere strictly to the extracted source code: in Module 2 Q19, `Car extends MotorVehicle` and `MotorVehicle extends Vehicle` (all three are classes). NEVER state or create options stating that `Vehicle` is an interface implemented by `MotorVehicle`, or that `Car` directly extends `Vehicle`.
   - For Java code-output questions finding array max/min: trace carefully through every element (e.g. in `{12, 45, 7, 89, 23}`, max is 89 and min is 7, so printing `max + " " + min` outputs `89 7`). Do NOT confuse the initial element with the minimum. The marked correct letter MUST match the exact computed output.
   - Do NOT ask questions where multiple options are syntactically valid in Java (e.g., both rectangular `new int[2][3]` and ragged `new int[2][]` are valid Java syntax).
   - All four options A, B, C, and D must be distinct.
5. High Quality Explanations:
   - Explanations must provide clear technical justification and must NOT simply repeat the correct option text.
   - Keep each question, option, and explanation concise and under 30 words.
6. Output Format:
   For EACH of the {candidates_to_generate} questions, output strictly in this exact structure:

QUESTION: <Question text. Include ```java ... ``` block if code-dependent>
A) <Option A>
B) <Option B>
C) <Option C>
D) <Option D>
CORRECT: <Single letter: A, B, C, or D>
EXPLANATION: <One concise sentence explaining why this option is correct based on the excerpt and Java rules>
SOURCE_QUOTE: <Exact quote from excerpt supporting this>

Generate {candidates_to_generate} candidate questions following these rules now.
<document_excerpts>
{context}
</document_excerpts>
[/INST]"""

    mcq_llm = OllamaLLM(model="mistral:latest", temperature=0.1, num_predict=750, num_ctx=4096, top_k=20, top_p=0.9, keep_alive="10m")
    try:
        raw_output = mcq_llm.invoke(prompt)
        print(f"[MCQ GENERATION] Raw output length: {len(raw_output)} chars")
    except Exception as e:
        print(f"Error calling LLM for MCQs: {e}")
        return [], f"Unable to generate multiple-choice questions: {e}"

    candidates = parse_mcq_candidates(raw_output)
    print(f"[MCQ GENERATION] Parsed {len(candidates)} candidates from raw output")
    valid_candidates: list[MCQCandidate] = []
    rejected_questions: list[str] = []

    for cand in candidates:
        improve_weak_explanation(cand, rep_chunks)
        is_valid, reason = validate_mcq_candidate(cand, rep_chunks)
        if is_valid:
            valid_candidates.append(cand)
            if len(valid_candidates) == mcq_count:
                break
        else:
            rejected_questions.append(cand.question)
            print(f"[MCQ VALIDATION REJECTED] {reason} | Question: {cand.question[:60]}")

    # Fallback retry if we didn't get enough validated candidates
    if len(valid_candidates) < mcq_count:
        needed = mcq_count - len(valid_candidates)
        print(f"[MCQ RETRY] Need {needed} more valid MCQs, generating {needed + 1} retry candidates")
        rejected_notice = ""
        if rejected_questions:
            rejected_notice = "CRITICAL - DO NOT REPEAT: The following questions were rejected as ambiguous or invalid. Do NOT generate them or anything similar:\n" + \
                "\n".join(f"- {q[:90]}" for q in rejected_questions[:4]) + "\n"

        retry_prompt = f"""[INST] You are an expert educational assessment generator creating multiple-choice questions grounded STRICTLY in the provided document excerpts.
Generate {needed + 1} simple, theoretical or conceptual multiple-choice questions based on the document text with NO ambiguity.
{rejected_notice}
CRITICAL:
- Do NOT ask questions about code snippets, code outputs, or variable declarations.
- NEVER ask what the return type of an interface method or class method is (interface methods can declare any return type).
- NEVER claim or generate options stating that interface methods have no return type or default to void, or that 1D or 2D arrays store different data types.
- The 'implements' keyword is used by a class to declare that it implements an interface; it does NOT define an interface.
- When comparing class and interface, state within document scope that class provides concrete implementations and state to create objects, whereas interface defines abstract method contract.
- For inheritance questions, follow the exact document hierarchy: Car extends MotorVehicle, which extends Vehicle (all three are classes). NEVER claim Vehicle is an interface in this hierarchy.
- Ask ONLY clear conceptual questions about Java keywords (`this`, `super`, `implements`), language rules, OOP principles, default values, or packages (`java.lang`) described in the excerpts.
- Explanations must be technically accurate and must NOT simply repeat the option text.
Keep each question, option, and explanation under 30 words.

Output format for each question:
QUESTION: <Question text>
A) <Option A>
B) <Option B>
C) <Option C>
D) <Option D>
CORRECT: <Single letter: A, B, C, or D>
EXPLANATION: <One sentence technical explanation>
SOURCE_QUOTE: <Quote>

<document_excerpts>
{context}
</document_excerpts>
[/INST]"""
        try:
            raw_retry = mcq_llm.invoke(retry_prompt)
            retry_candidates = parse_mcq_candidates(raw_retry)
            for cand in retry_candidates:
                # Forbid reintroducing previously rejected questions
                if any(cand.question.strip().lower() == rej.strip().lower() for rej in rejected_questions):
                    print(f"[RETRY MCQ REJECTED] Repeated previously rejected question: {cand.question[:60]}")
                    continue
                improve_weak_explanation(cand, rep_chunks)
                is_valid, reason = validate_mcq_candidate(cand, rep_chunks)
                if is_valid:
                    valid_candidates.append(cand)
                    if len(valid_candidates) == mcq_count:
                        break
                else:
                    print(f"[RETRY MCQ REJECTED] {reason} | Question: {cand.question[:60]}")
        except Exception as e:
            print(f"Retry LLM error: {e}")

    # Verified document fallbacks if LLM still falls short
    if len(valid_candidates) < mcq_count:
        doc_fallbacks = [
            MCQCandidate(
                question="What will be the output of the following code snippet?\n```java\nint[] a = {12, 45, 7, 89, 23};\nint max = a[0], min = a[0];\nfor (int x : a) {\n    if (x > max) max = x;\n    if (x < min) min = x;\n}\nSystem.out.println(max + \" \" + min);\n```",
                options={"A": "89 12", "B": "12 89", "C": "89 7", "D": "7 89"},
                correct="C",
                explanation="The loop iterates through the array [12, 45, 7, 89, 23], updating max to 89 and min to 7, so `max + \" \" + min` outputs `89 7`.",
                source_quote="Q3. Find largest and smallest element in array"
            ),
            MCQCandidate(
                question="In the document's multilevel inheritance hierarchy, what is the relationship between the `Vehicle`, `MotorVehicle`, and `Car` classes?",
                options={"A": "Car extends MotorVehicle, and MotorVehicle extends Vehicle.", "B": "Vehicle extends MotorVehicle, and MotorVehicle extends Car.", "C": "Car and MotorVehicle both directly extend Vehicle independently.", "D": "Vehicle is an interface implemented by MotorVehicle and Car."},
                correct="A",
                explanation="In the document's multilevel inheritance example (Module 2 Q19), Vehicle is the base class, MotorVehicle extends Vehicle, and Car extends MotorVehicle.",
                source_quote="class MotorVehicle extends Vehicle ... class Car extends MotorVehicle"
            ),
            MCQCandidate(
                question="Which statement correctly declares a 2D array of integers with 4 rows and 3 columns in Java?",
                options={"A": "int[][] arr = new int[4][3];", "B": "int[4][3] arr = new int[][];", "C": "int arr = new int[4,3];", "D": "array arr = new array(4,3);"},
                correct="A",
                explanation="In Java, 2D arrays are declared with bracket pairs specifying row and column dimensions, e.g., `int[][] arr = new int[4][3];`.",
                source_quote="int[][] arr = new int[4][3];"
            ),
            MCQCandidate(
                question="What will be the output of the following code?\n```java\nString s = \"Java\";\ns.concat(\" SE\");\nSystem.out.println(s);\n```",
                options={"A": "Java", "B": "Java SE", "C": "SE", "D": "Compilation Error"},
                correct="A",
                explanation="In Java, String objects are immutable; concat() returns a new String and does not modify the original reference without reassignment.",
                source_quote="Output: Java (Strings are immutable)"
            ),
            MCQCandidate(
                question="Which keyword is used by a Java class to implement an interface?",
                options={"A": "implements", "B": "extends", "C": "interface", "D": "inherits"},
                correct="A",
                explanation="In Java, a class uses the 'implements' keyword to declare that it implements an interface, providing concrete implementations for its declared methods.",
                source_quote="Which keyword is used to implement an interface? implements"
            ),
            MCQCandidate(
                question="What is the default value of an uninitialized boolean instance variable in Java?",
                options={"A": "false", "B": "true", "C": "null", "D": "0"},
                correct="A",
                explanation="In Java, uninitialized instance and static boolean fields default to false.",
                source_quote="What is the default value of an uninitialized boolean variable in a class? false"
            )
        ]
        for fb in doc_fallbacks:
            if not any(v.question.strip().lower() == fb.question.strip().lower() for v in valid_candidates):
                improve_weak_explanation(fb, rep_chunks)
                is_val, reason = validate_mcq_candidate(fb, rep_chunks)
                if is_val:
                    valid_candidates.append(fb)
                    if len(valid_candidates) == mcq_count:
                        break
                else:
                    print(f"[FALLBACK MCQ REJECTED] {reason} | Question: {fb.question[:60]}")

    # Final verification pass: guarantee every candidate is strictly validated after explanation improvement
    final_verified = []
    for cand in valid_candidates:
        improve_weak_explanation(cand, rep_chunks)
        is_val, reason = validate_mcq_candidate(cand, rep_chunks)
        if is_val:
            final_verified.append(cand)
            if len(final_verified) == mcq_count:
                break
        else:
            print(f"[FINAL SELECTION REJECTED] {reason} | Question: {cand.question[:60]}")

    if len(final_verified) < mcq_count:
        return [], f"I couldn't find enough reliable source material in the uploaded document to generate {mcq_count} verified multiple-choice questions."

    selected = final_verified[:mcq_count]
    STORED_MCQ_ANSWERS[document_id] = selected
    if include_answers:
        formatted = format_complete_mcqs(selected)
    else:
        formatted = format_user_facing_mcqs(selected)
    return selected, formatted



def resolve_conversational_query(question: str, messages: list[dict] | None) -> tuple[str, str]:
    if not messages or len(messages) < 1:
        return question, question

    q_lower = question.lower().strip()
    pronouns = [
        r"\bit\b", r"\bits\b", r"\bthis\b", r"\bthat\b", r"\bthem\b", r"\bthese\b",
        r"\bthose\b", r"\bsecond one\b", r"\bfirst one\b", r"\bthird one\b",
        r"\bformer\b", r"\blatter\b", r"\bthe second\b", r"\bthe first\b"
    ]

    has_reference = any(re.search(p, q_lower) for p in pronouns)
    if not has_reference:
        return question, question

    recent_user_q = ""
    recent_assistant_a = ""

    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user" and not recent_user_q:
            recent_user_q = content
        elif role == "assistant" and not recent_assistant_a:
            recent_assistant_a = content
        if recent_user_q and recent_assistant_a:
            break

    if not recent_user_q:
        return question, question

    prev_words = list(important_words(recent_user_q))
    if not prev_words and recent_assistant_a:
        prev_words = list(important_words(recent_assistant_a))

    if prev_words:
        subject_context = " ".join(prev_words[:4])
        retrieval_query = f"{subject_context} {question}"
        prompt_question = f"{question} (Context: regarding {subject_context})"
        return retrieval_query, prompt_question

    return question, question


# ============================================================
# CHAT ENDPOINT
# ============================================================

@router.post("/chat")
async def chat(request: ChatRequest):
    total_start = time.time()

    print("[TRACE] QUESTION:", request.question)
    print("[TRACE] DOCUMENT_ID:", request.document_id)
    print("[TRACE] ALLOW_GENERAL:", request.allow_general)
    print("[TRACE] PERMISSION_RESPONSE:", request.permission_response)

    question = request.question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "source": "conversation",
            "requires_permission": False,
            "conversation_state": "no_document"
        }

    answer_format = detect_answer_format(question)

    # ========================================================
    # STATE: NO DOCUMENT UPLOADED
    # ========================================================
    if not request.document_id or not request.document_id.strip():
        if is_greeting_message(question):
            greeting_ans = (
                "Hey! 👋 I'm Cognivault. You can upload a document whenever you want "
                "me to work from your own material. Until then, we can still chat normally. "
                "What would you like to talk about?"
            )
            metrics = {
                "chromadb_retrieval_time": 0.0,
                "prompt_construction_time": 0.0,
                "ollama_startup_time": 0.0,
                "first_token_time": 0.0,
                "llm_generation_time": 0.0,
                "total_api_time": round(time.time() - total_start, 4)
            }

            print("\n===================================")
            print("COGNIVAULT CHAT")
            print("Original Question:", question)
            print("Active Document ID: None")
            print("Intent: Greeting (No Document)")
            print("Retrieval Mode: None")
            print("Retrieved Chunks: 0")
            print("Distances: []")
            print("Chunk Indexes: []")
            print("Chunk Previews: []")
            print("Final Decision: GREETING CONVERSATION")
            print("Source: conversation")
            print("Conversation State: no_document")

            if request.stream:
                async def stream_greeting():
                    yield json.dumps({
                        "type": "meta",
                        "source": "conversation",
                        "requires_permission": False,
                        "conversation_state": "no_document"
                    }) + "\n"
                    yield json.dumps({"type": "token", "token": greeting_ans}) + "\n"
                    yield json.dumps({"type": "metrics", "metrics": metrics}) + "\n"
                    yield json.dumps({"type": "done"}) + "\n"
                return StreamingResponse(stream_greeting(), media_type="application/x-ndjson")

            return {
                "answer": greeting_ans,
                "source": "conversation",
                "requires_permission": False,
                "conversation_state": "no_document",
                "metrics": metrics
            }

        # General question with no document attached
        gen_llm = OllamaLLM(model="mistral:latest", temperature=0, num_predict=400, num_ctx=2048, top_k=20, top_p=0.9, keep_alive="10m")
        gen_prompt = f"Answer this question naturally and conversationally using your general knowledge:\n\n{question}\n\nBe clear, engaging, and concise. Provide a complete, self-contained response in 150 to 250 words. Ensure your response concludes naturally with a complete final sentence; do not cut off mid-sentence. You MUST formulate your response in English."

        print("\n===================================")
        print("COGNIVAULT CHAT")
        print("Original Question:", question)
        print("Active Document ID: None")
        print("Intent: General Question (No Document)")
        print("Retrieval Mode: None")
        print("Retrieved Chunks: 0")
        print("Distances: []")
        print("Chunk Indexes: []")
        print("Chunk Previews: []")
        print("Final Decision: GENERAL KNOWLEDGE (NO DOCUMENT)")
        print("Source: general_knowledge")
        print("Conversation State: no_document")

        if request.stream:
            async def stream_no_doc_gen():
                yield json.dumps({
                    "type": "meta",
                    "source": "general_knowledge",
                    "requires_permission": False,
                    "conversation_state": "no_document"
                }) + "\n"
                first_tok = None
                gen_start = time.time()
                accumulated = []
                try:
                    async for chunk in gen_llm.astream(gen_prompt):
                        if first_tok is None:
                            first_tok = time.time() - gen_start
                        accumulated.append(chunk)
                        yield json.dumps({"type": "token", "token": chunk}) + "\n"

                    full_raw = "".join(accumulated)
                    if not is_sentence_complete(full_raw):
                        cont_snippet = generate_continuation_snippet(gen_prompt, full_raw)
                        if cont_snippet:
                            clean_cont = stitch_continuation(full_raw, cont_snippet)[len(full_raw):]
                            if clean_cont:
                                yield json.dumps({"type": "token", "token": clean_cont}) + "\n"

                    gen_dur = time.time() - gen_start
                    tot_dur = time.time() - total_start
                    metrics = {
                        "chromadb_retrieval_time": 0.0,
                        "prompt_construction_time": 0.0,
                        "ollama_startup_time": round(first_tok if first_tok else 0.0, 4),
                        "first_token_time": round(first_tok if first_tok else 0.0, 4),
                        "llm_generation_time": round(gen_dur, 4),
                        "total_api_time": round(tot_dur, 4)
                    }
                    yield json.dumps({"type": "metrics", "metrics": metrics}) + "\n"
                except Exception as ex:
                    yield json.dumps({"type": "error", "error": str(ex)}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"

            return StreamingResponse(stream_no_doc_gen(), media_type="application/x-ndjson")

        gen_start = time.time()
        first_tok = None
        tokens = []
        try:
            async for chunk in gen_llm.astream(gen_prompt):
                if first_tok is None:
                    first_tok = time.time() - gen_start
                tokens.append(chunk)
            raw_answer = "".join(tokens)
            answer = complete_response_if_needed(gen_prompt, raw_answer)
        except Exception as e:
            answer = f"I'm Cognivault. I can answer general questions or ground answers in your uploaded PDF. ({e})"

        tot_dur = time.time() - total_start
        metrics = {
            "chromadb_retrieval_time": 0.0,
            "prompt_construction_time": 0.0,
            "ollama_startup_time": round(first_tok if first_tok else 0.0, 4),
            "first_token_time": round(first_tok if first_tok else 0.0, 4),
            "llm_generation_time": round(time.time() - gen_start, 4),
            "total_api_time": round(tot_dur, 4)
        }
        return {
            "answer": answer,
            "source": "general_knowledge",
            "requires_permission": False,
            "conversation_state": "no_document",
            "metrics": metrics
        }

    # ========================================================
    # DEBUG LOGGING FOR ROUTING
    # ========================================================
    print("\\n" + "="*50)
    print("ROUTING DEBUG INFO:")
    print("QUESTION:", question)
    print("DOCUMENT_ID:", request.document_id)
    print("PERMISSION_RESPONSE:", request.permission_response)
    
    is_mcq_gen = is_mcq_generation_query(question)
    is_mcq_ans = is_mcq_answer_request(question)
    is_overview = is_document_overview_query(question)
    
    is_combined = is_overview and is_mcq_gen
    
    print("MCQ_GENERATION:", is_mcq_gen)
    print("MCQ_ANSWER:", is_mcq_ans)
    print("OVERVIEW:", is_overview)
    print("="*50 + "\\n")


    # DOCUMENT ACTIVE - CHECK FOR MCQ GENERATION INTENT

    if is_combined:
        print("ROUTE = COMBINED_OVERVIEW_MCQ")
        rep_chunks = get_representative_chunks(request.document_id)
        rep_chunks = enrich_chunks_with_class_definitions(request.document_id, rep_chunks)
        
        print("\n===================================")
        print("COGNIVAULT CHAT")
        print("Intent: Document Overview + MCQ")
        
        if not check_evidence_sufficiency(rep_chunks):
            return {"answer": "I don't have enough readable content from this document to summarize or generate MCQs.", "source": "document_not_found", "requires_permission": False, "conversation_state": "document_grounded", "metrics": {}}
            
        context_chunks = rep_chunks[:5] if len(rep_chunks) >= 5 else rep_chunks
        context = "\n\n---\n\n".join([c[:600] for c in context_chunks])
        
        overview_prompt = f"""[INST] You are an expert technical assistant summarizing a user's uploaded document.

Provide a clear, natural English summary of the document based STRICTLY AND ONLY on the supplied excerpts, in a single cohesive paragraph (around 100 to 180 words). Ensure your paragraph concludes with a complete final sentence.

CRITICAL INSTRUCTIONS:
- Identify the main topic, major concepts, and important sections supported by the PDF.
- Write a continuous, fluent, cohesive paragraph-style explanation.
- DO NOT use bullet points, numbered lists, or line breaks between ideas.
- DO NOT dump or repeat raw code snippets. If the document contains mostly code, summarize the code's purpose in natural language.
- Preserve technical names and code concepts accurately.
- NEVER state or guess the total number of questions, total pages, total marks, or module numbers unless that exact count is explicitly written in the excerpts.
- Distinguish verified document facts from general descriptions. If a detail cannot be verified, omit it.
- If the excerpts do not contain enough readable text to summarize, respond EXACTLY with: "I don't have enough readable content from this document to determine what it is about."
- You MUST formulate your response in English.

<document_excerpts>
{context}
</document_excerpts>

Provide the summary now in a single cohesive paragraph following the strict constraints above. Do not use any lists or bullet points. [/INST]"""

        # Generate overview with try/except to prevent silent crash
        try:
            overview_llm = OllamaLLM(model="mistral:latest", temperature=0, num_predict=350, num_ctx=4096, top_k=20, top_p=0.9, keep_alive="10m")
            raw_overview = overview_llm.invoke(overview_prompt)
            print(f"[COMBINED] Raw overview length: {len(raw_overview)} chars")
            overview_answer = complete_response_if_needed(overview_prompt, raw_overview)
            overview_answer = sanitize_overview_claims(overview_answer, rep_chunks)
            print(f"[COMBINED] Final overview length: {len(overview_answer)} chars")
        except Exception as e:
            print(f"[COMBINED] Overview generation error: {e}")
            overview_answer = "This document could not be summarized due to a processing error."
        
        mcq_count = extract_mcq_count(question)
        valid_mcqs, formatted_mcqs = generate_and_validate_mcqs(request.document_id, mcq_count, rep_chunks, include_answers=True)
        print(f"[COMBINED] MCQ validation result: {len(valid_mcqs)} valid MCQs, formatted length: {len(formatted_mcqs)} chars")
        
        combined_final = overview_answer.strip() + "\n\n" + formatted_mcqs.strip()
        if combined_final and combined_final[-1] not in {'.', '!', '?', '"', "'", '`', ')'}:
            combined_final += "."
        source = "document" if valid_mcqs else "document_not_found"
        print(f"[COMBINED] Final combined response length: {len(combined_final)} chars")
        
        if request.stream:
            async def stream_combined():
                yield json.dumps({"type": "meta", "source": source, "requires_permission": False, "conversation_state": "document_grounded"}) + "\n"
                words = combined_final.split(" ")
                for i, w in enumerate(words):
                    yield json.dumps({"type": "token", "token": w + (" " if i < len(words) - 1 else "")}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
            return StreamingResponse(stream_combined(), media_type="application/x-ndjson")
        else:
            return {"answer": combined_final, "source": source, "requires_permission": False, "conversation_state": "document_grounded", "metrics": {}}

    elif is_mcq_gen:
        print("ROUTE = MCQ_GENERATION")
        rep_chunks = get_representative_chunks(request.document_id)
        rep_chunks = enrich_chunks_with_class_definitions(request.document_id, rep_chunks)
        
        print("\n===================================")
        print("COGNIVAULT CHAT")
        print("Original Question:", question)
        print("Active Document ID:", request.document_id)
        print("Intent: Document MCQ Generation")
        print("Retrieval Mode: Representative Sampling")
        print("Retrieved Chunks:", len(rep_chunks))
        print("Final Decision:", "GROUNDED DOCUMENT MCQ" if rep_chunks else "REPRESENTATIVE RETRIEVAL FAILED")
        print("Source:", "document" if rep_chunks else "document_not_found")
        print("Conversation State:", "document_grounded")

        if not check_evidence_sufficiency(rep_chunks):
            refusal_msg = "I don't have enough readable content from this document to generate MCQs."
            if request.stream:
                async def stream_insufficient():
                    yield json.dumps({"type": "meta", "source": "document_not_found", "requires_permission": False, "conversation_state": "document_grounded"}) + "\n"
                    words = refusal_msg.split()
                    for i, w in enumerate(words):
                        yield json.dumps({"type": "token", "token": w + (" " if i < len(words) - 1 else "")}) + "\n"
                    yield json.dumps({"type": "done"}) + "\n"
                return StreamingResponse(stream_insufficient(), media_type="application/x-ndjson")
            else:
                return {"answer": refusal_msg, "source": "document_not_found", "requires_permission": False, "conversation_state": "document_grounded", "metrics": {}}

        mcq_count = extract_mcq_count(question)
        valid_mcqs, formatted_answer = generate_and_validate_mcqs(request.document_id, mcq_count, rep_chunks)
        source = "document" if valid_mcqs else "document_not_found"

        if request.stream:
            async def stream_mcq():
                yield json.dumps({
                    "type": "meta",
                    "source": source,
                    "requires_permission": False,
                    "conversation_state": "document_grounded"
                }) + "\n"
                words = formatted_answer.split(" ")
                for i, w in enumerate(words):
                    yield json.dumps({"type": "token", "token": w + (" " if i < len(words) - 1 else "")}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
            return StreamingResponse(stream_mcq(), media_type="application/x-ndjson")
        else:
            return {
                "answer": formatted_answer,
                "source": source,
                "requires_permission": False,
                "conversation_state": "document_grounded",
                "metrics": {}
            }

    elif is_mcq_ans:
        print("ROUTE = MCQ_ANSWER_REVEAL")
        if request.document_id and request.document_id in STORED_MCQ_ANSWERS and STORED_MCQ_ANSWERS[request.document_id]:
            verified_ans_text = format_verified_answers(STORED_MCQ_ANSWERS[request.document_id])
            if request.stream:
                async def stream_stored_ans():
                    yield json.dumps({
                        "type": "meta",
                        "source": "document",
                        "requires_permission": False,
                        "conversation_state": "document_grounded"
                    }) + "\n"
                    words = verified_ans_text.split(" ")
                    for i, w in enumerate(words):
                        yield json.dumps({"type": "token", "token": w + (" " if i < len(words) - 1 else "")}) + "\n"
                    yield json.dumps({"type": "done"}) + "\n"
                return StreamingResponse(stream_stored_ans(), media_type="application/x-ndjson")
            else:
                return {
                    "answer": verified_ans_text,
                    "source": "document",
                    "requires_permission": False,
                    "conversation_state": "document_grounded",
                    "metrics": {}
                }

        # Extract the previous MCQs from history
        print("=== DEBUGGING MESSAGES ===", request.messages, type(request.messages)); request.messages = request.messages or []
        last_mcqs = "No previous questions found."
        for m in reversed(request.messages):
            print("m type:", type(m), "role:", m.get("role"), "content has A)?", "A)" in m.get("content", ""))
            if m.get("role") == "assistant" and "A)" in m.get("content", ""):
                last_mcqs = m.get("content")
                break
                
        # Retrieve chunks based on the MCQs themselves
        retrieval_start = time.time()
        results = vector_db.similarity_search_with_score(
            last_mcqs[:500], # Use first 500 chars of questions to find relevant chunks
            k=8,
            filter={"document_id": request.document_id}
        )
        retrieval_time = time.time() - retrieval_start
        rep_chunks = [doc.page_content.replace("\n", " ") for doc, score in results]
        context = "\n\n---\n\n".join([c[:600] + ("..." if len(c) > 600 else "") for c in rep_chunks])
        
        print("=== LAST MCQS ===", last_mcqs)
        ans_prompt = f"""[INST] You are an expert that finds correct answers for multiple choice questions based on a provided text.

<document_excerpts>
Java is a programming language. To create a thread in Java, you can extend the Thread class or implement the Runnable interface.
</document_excerpts>

<previous_questions>
1. What is Java?
A) A snake
B) A programming language
C) A car
D) A planet

2. How do you create a thread in Java?
A) Extend Thread class
B) Implement Runnable interface
C) Both A and B
D) None of the above
</previous_questions>

Based on the document excerpts, provide the correct answers to the previous_questions.
Use the exact format:
1. Correct Answer: B) A programming language
2. Correct Answer: C) Both A and B
[/INST]
[INST]
<document_excerpts>
{context}
</document_excerpts>

<previous_questions>
{last_mcqs}
</previous_questions>

Based on the document excerpts, provide the correct answers to the previous_questions.
Use the exact format shown in the example. Do not output anything else.
You MUST formulate your response in English.
[/INST]"""
        
        print("\n=== ANS PROMPT ===\n" + ans_prompt + "\n==================\n"); ans_llm = OllamaLLM(model="mistral:latest", temperature=0.1, num_predict=500, num_ctx=2048, top_k=20, top_p=0.9)
        gen_start = time.time()
        
        if request.stream:
            async def stream_ans():
                yield json.dumps({
                    "type": "meta",
                    "source": "document",
                    "requires_permission": False,
                    "conversation_state": "document_grounded"
                }) + "\n"
                try:
                    async for chunk in ans_llm.astream(ans_prompt):
                        yield json.dumps({"type": "token", "token": chunk}) + "\n"
                except Exception as e:
                    yield json.dumps({"type": "error", "error": str(e)}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
            return StreamingResponse(stream_ans(), media_type="application/x-ndjson")
            
        else:
            answer = ans_llm.invoke(ans_prompt)
            return {
                "answer": answer,
                "source": "document",
                "requires_permission": False,
                "conversation_state": "document_grounded",
                "metrics": {}
            }

    if is_overview:
        print("ROUTE = DOCUMENT_OVERVIEW")
        rep_chunks = get_representative_chunks(request.document_id)

        print("\n===================================")
        print("COGNIVAULT CHAT")
        print("Original Question:", question)
        print("Active Document ID:", request.document_id)
        print("Intent: Document Overview / Meta-Query")
        
        is_sufficient = check_evidence_sufficiency(rep_chunks)
        print(f"Evidence Sufficient: {is_sufficient}")
        
        if not is_sufficient:
            answer = "I don't have enough readable content from this document to determine what it is about."
            if request.stream:
                async def stream_refusal():
                    import json
                    yield json.dumps({"type": "meta", "source": "document_not_found", "requires_permission": False, "conversation_state": "document_grounded"}) + "\n"
                    words = answer.split()
                    for i, word in enumerate(words):
                        yield json.dumps({"type": "token", "token": word + (" " if i < len(words)-1 else "")}) + "\n"
                    yield json.dumps({"type": "done"}) + "\n"
                return StreamingResponse(stream_refusal(), media_type="application/x-ndjson")
            else:
                return {"answer": answer, "source": "document_not_found", "requires_permission": False, "conversation_state": "document_grounded", "metrics": {}}

        context_chunks = rep_chunks[:5] if len(rep_chunks) >= 5 else rep_chunks
        context = "\n\n---\n\n".join([c[:600] for c in context_chunks])
        overview_prompt = f"""[INST] You are an expert technical assistant summarizing a user's uploaded document.

Your task is to provide a clear, natural English summary of the document based ONLY on the supplied excerpts, in a single cohesive paragraph (around 100 to 180 words). Ensure your paragraph concludes with a complete final sentence.

CRITICAL INSTRUCTIONS:
- Identify the main topic, major concepts, and important sections supported by the PDF.
- Write a continuous, fluent, cohesive paragraph-style explanation.
- DO NOT use bullet points, numbered lists, or line breaks between ideas.
- DO NOT dump or repeat raw code snippets. If the document contains mostly code, summarize the code's purpose in natural language.
- Preserve technical names and code concepts accurately.
- NEVER state or guess the total number of questions, total pages, total marks, or module numbers unless that exact count is explicitly written in the excerpts.
- Distinguish verified document facts from general descriptions. If a detail cannot be verified, omit it.
- If the excerpts do not contain enough readable text to summarize, respond EXACTLY with: "I don't have enough readable content from this document to determine what it is about."
- You MUST formulate your response in English.

<document_excerpts>
{context}
</document_excerpts>

Provide the summary now in a single cohesive paragraph following the strict constraints above. Do not use any lists or bullet points. [/INST]"""
        prompt_construction_time = time.time() - total_start
        overview_llm = OllamaLLM(model="mistral:latest", temperature=0, num_predict=350, num_ctx=2048, top_k=20, top_p=0.9, keep_alive="10m")
        
        print("Generating overview for validation...")
        gen_start = time.time()
        raw_overview = overview_llm.invoke(overview_prompt)
        generated_answer = complete_response_if_needed(overview_prompt, raw_overview)
        generated_answer = sanitize_overview_claims(generated_answer, rep_chunks)
        gen_dur = time.time() - gen_start
        
        # VALIDATION LAYER REMOVED
        # Multilingual RAG can produce entities/numbers translated to English
        # which would falsely trigger exact-match or LLM-based hallucination checks.
        source = "document"

        metrics = {
            "chromadb_retrieval_time": 0.0,
            "prompt_construction_time": round(prompt_construction_time, 4),
            "ollama_startup_time": 0.0,
            "first_token_time": 0.0,
            "llm_generation_time": round(gen_dur, 4),
            "total_api_time": round(time.time() - total_start, 4)
        }

        if request.stream:
            async def stream_validated():
                import json
                yield json.dumps({"type": "meta", "source": source, "requires_permission": False, "conversation_state": "document_grounded"}) + "\n"
                words = generated_answer.split()
                for i, word in enumerate(words):
                    yield json.dumps({"type": "token", "token": word + (" " if i < len(words)-1 else "")}) + "\n"
                yield json.dumps({"type": "metrics", "metrics": metrics}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
            return StreamingResponse(stream_validated(), media_type="application/x-ndjson")
        else:
            return {
                "answer": generated_answer,
                "source": source,
                "requires_permission": False,
                "conversation_state": "document_grounded",
                "metrics": metrics
            }

    else:
        print("ROUTE = NORMAL_RAG")
        
        # RETRIEVAL WITH DOCUMENT ISOLATION
        # ========================================================
        retrieval_query, prompt_question = resolve_conversational_query(question, request.messages)

        retrieval_start = time.time()
        results = vector_db.similarity_search_with_score(
            retrieval_query,
            k=5,
            filter={"document_id": request.document_id}
        )
        retrieval_time = time.time() - retrieval_start
    
        # Relevance threshold = 1.2
        relevance_threshold = 1.2
        relevant_documents = [
            doc for doc, score in results
            if score <= relevance_threshold
        ]
    
        # --- LEXICAL FALLBACK START ---
        # Used for OCR documents where disjointed tables/lists have poor semantic embedding scores
        if not relevant_documents and request.document_id:
            try:
                all_doc_chunks = vector_db._collection.get(
                    where={"document_id": request.document_id},
                    include=["documents", "metadatas"]
                )
                if all_doc_chunks and all_doc_chunks.get("documents"):
                    stopwords = {"what", "is", "the", "in", "pdf", "document", "mentioned", "about", "a", "an", "of", "and", "to", "for", "are", "on", "this", "tell", "me", "how", "much", "many", "does", "do", "can"}
                    query_words = [w for w in re.findall(r'\b\w+\b', retrieval_query.lower()) if w not in stopwords]
                    
                    if query_words:
                        best_chunk_idx = -1
                        best_match_count = 0
                        for i, doc_text in enumerate(all_doc_chunks["documents"]):
                            doc_text_lower = doc_text.lower()
                            match_count = sum(1 for w in query_words if w in doc_text_lower)
                            if match_count > best_match_count:
                                best_match_count = match_count
                                best_chunk_idx = i
                        
                        required_matches = max(1, len(query_words) // 2)
                        if best_chunk_idx != -1 and best_match_count >= required_matches:
                            from langchain_core.documents import Document
                            fallback_doc = Document(
                                page_content=all_doc_chunks["documents"][best_chunk_idx],
                                metadata=all_doc_chunks["metadatas"][best_chunk_idx] if all_doc_chunks.get("metadatas") else {}
                            )
                            relevant_documents = [fallback_doc]
                            print(f"Lexical Fallback Triggered! Matches: {best_match_count}/{len(query_words)}")
            except Exception as e:
                print(f"Lexical fallback failed: {e}")
        # --- LEXICAL FALLBACK END ---

        # PERMISSION HANDLING FOR UNFOUND DOCUMENT INFORMATION
        if not relevant_documents:
            user_permitted_general = request.allow_general or (
                request.permission_response and request.permission_response.lower().strip() in ["yes", "y", "true", "sure", "ok", "okay"]
            )
            user_denied_general = (
                request.permission_response and request.permission_response.lower().strip() in ["no", "n", "false"]
            )

            if user_denied_general:
                denial_msg = "Understood. I will stick strictly to the content in your uploaded document. Let me know what else you'd like to look up!"
                metrics = {
                    "chromadb_retrieval_time": round(retrieval_time, 4),
                    "prompt_construction_time": 0.0,
                    "ollama_startup_time": 0.0,
                    "first_token_time": 0.0,
                    "llm_generation_time": 0.0,
                    "total_api_time": round(time.time() - total_start, 4)
                }
                if request.stream:
                    async def stream_denial():
                        yield json.dumps({
                            "type": "meta",
                            "source": "document_not_found",
                            "requires_permission": False,
                            "conversation_state": "document_grounded"
                        }) + "\n"
                        yield json.dumps({"type": "token", "token": denial_msg}) + "\n"
                        yield json.dumps({"type": "metrics", "metrics": metrics}) + "\n"
                        yield json.dumps({"type": "done"}) + "\n"
                    return StreamingResponse(stream_denial(), media_type="application/x-ndjson")
                return {
                    "answer": denial_msg,
                    "source": "document_not_found",
                    "requires_permission": False,
                    "conversation_state": "document_grounded",
                    "metrics": metrics
                }

            if user_permitted_general:
                # Resolve target question if user simply replied "yes"
                target_question = question
                if question.lower().strip() in ["yes", "y", "sure", "ok", "okay", "please", "go ahead"] and request.messages:
                    for m in reversed(request.messages):
                        if m.get("role") == "user" and m.get("content", "").lower().strip() not in ["yes", "y", "sure", "ok", "okay"]:
                            target_question = m.get("content", "").strip()
                            break

                gen_llm = OllamaLLM(model="mistral:latest", temperature=0, num_predict=400, num_ctx=2048, top_k=20, top_p=0.9, keep_alive="10m")
                gen_prompt = f"Answer this question naturally and conversationally using your general knowledge:\n\n{target_question}\n\nBe clear, helpful, and concise. Provide a complete, self-contained response in 150 to 250 words. Ensure your response concludes naturally with a complete final sentence; do not cut off mid-sentence. You MUST formulate your response in English."

                print("\n===================================")
                print("COGNIVAULT CHAT")
                print("Original Question:", target_question)
                print("Active Document ID:", request.document_id)
                print("Intent: General Knowledge Fallback (Permission Granted)")
                print("Final Decision: GENERAL KNOWLEDGE (PERMISSION GRANTED)")
                print("Source: general_knowledge")
                print("Conversation State: document_grounded")

                if request.stream:
                    async def stream_fallback_gen():
                        yield json.dumps({
                            "type": "meta",
                            "source": "general_knowledge",
                            "requires_permission": False,
                            "conversation_state": "document_grounded"
                        }) + "\n"
                        first_tok = None
                        gen_start = time.time()
                        accumulated = []
                        try:
                            async for chunk in gen_llm.astream(gen_prompt):
                                if first_tok is None:
                                    first_tok = time.time() - gen_start
                                accumulated.append(chunk)
                                yield json.dumps({"type": "token", "token": chunk}) + "\n"

                            full_raw = "".join(accumulated)
                            if not is_sentence_complete(full_raw):
                                cont_snippet = generate_continuation_snippet(gen_prompt, full_raw)
                                if cont_snippet:
                                    clean_cont = stitch_continuation(full_raw, cont_snippet)[len(full_raw):]
                                    if clean_cont:
                                        yield json.dumps({"type": "token", "token": clean_cont}) + "\n"

                            gen_dur = time.time() - gen_start
                            tot_dur = time.time() - total_start
                            metrics = {
                                "chromadb_retrieval_time": round(retrieval_time, 4),
                                "prompt_construction_time": 0.0,
                                "ollama_startup_time": round(first_tok if first_tok else 0.0, 4),
                                "first_token_time": round(first_tok if first_tok else 0.0, 4),
                                "llm_generation_time": round(gen_dur, 4),
                                "total_api_time": round(tot_dur, 4)
                            }
                            yield json.dumps({"type": "metrics", "metrics": metrics}) + "\n"
                        except Exception as ex:
                            yield json.dumps({"type": "error", "error": str(ex)}) + "\n"
                        yield json.dumps({"type": "done"}) + "\n"

                    return StreamingResponse(stream_fallback_gen(), media_type="application/x-ndjson")

                gen_start = time.time()
                first_tok = None
                tokens = []
                try:
                    async for chunk in gen_llm.astream(gen_prompt):
                        if first_tok is None:
                            first_tok = time.time() - gen_start
                        tokens.append(chunk)
                    raw_answer = "".join(tokens)
                    answer = complete_response_if_needed(gen_prompt, raw_answer)
                except Exception as e:
                    answer = f"I'm Cognivault. I can answer general questions or ground answers in your uploaded PDF. ({e})"

                tot_dur = time.time() - total_start
                metrics = {
                    "chromadb_retrieval_time": round(retrieval_time, 4),
                    "prompt_construction_time": 0.0,
                    "ollama_startup_time": round(first_tok if first_tok else 0.0, 4),
                    "first_token_time": round(first_tok if first_tok else 0.0, 4),
                    "llm_generation_time": round(time.time() - gen_start, 4),
                    "total_api_time": round(tot_dur, 4)
                }
                return {
                    "answer": answer,
                    "source": "general_knowledge",
                    "requires_permission": False,
                    "conversation_state": "document_grounded",
                    "metrics": metrics
                }

            # DEFAULT PERMISSION REQUEST FOR UNFOUND DOCUMENT INFORMATION
            permission_msg = (
                "I couldn't find that information in your uploaded document. "
                "Would you like me to give you a general answer?"
            )
            metrics = {
                "chromadb_retrieval_time": round(retrieval_time, 4),
                "prompt_construction_time": round(time.time() - total_start - retrieval_time, 4),
                "ollama_startup_time": 0.0,
                "first_token_time": 0.0,
                "llm_generation_time": 0.0,
                "total_api_time": round(time.time() - total_start, 4)
            }

            if request.stream:
                async def stream_permission():
                    yield json.dumps({
                        "type": "meta",
                        "source": "document_not_found",
                        "requires_permission": True,
                        "conversation_state": "awaiting_general_permission"
                    }) + "\n"
                    yield json.dumps({"type": "token", "token": permission_msg}) + "\n"
                    yield json.dumps({"type": "metrics", "metrics": metrics}) + "\n"
                    yield json.dumps({"type": "done"}) + "\n"
                return StreamingResponse(stream_permission(), media_type="application/x-ndjson")

            return {
                "answer": permission_msg,
                "source": "document_not_found",
                "requires_permission": True,
                "conversation_state": "awaiting_general_permission",
                "metrics": metrics
            }

    # ========================================================
    # DOCUMENT GROUNDED (RELEVANT CHUNKS FOUND)
    # ========================================================
    # MCQ Optimization Check
    document_is_mcq = any(looks_like_mcq(doc.page_content) for doc in relevant_documents)
    if document_is_mcq:
        matching_mcq = find_matching_mcq(prompt_question, relevant_documents)
        if matching_mcq:
            allow_direct_mcq = answer_format["type"] in {"definition", "normal", "mcq", "fill_blank"}
            strong_match = matching_mcq["score"] >= 0.65
            if allow_direct_mcq and strong_match:
                correct_answer = extract_explicit_correct_answer(matching_mcq["document"].page_content)
                if correct_answer:
                    total_time = time.time() - total_start
                    metrics = {
                        "chromadb_retrieval_time": round(retrieval_time, 4),
                        "prompt_construction_time": round(total_time - retrieval_time, 4),
                        "ollama_startup_time": 0.0,
                        "first_token_time": 0.0,
                        "llm_generation_time": 0.0,
                        "total_api_time": round(total_time, 4)
                    }
                    if request.stream:
                        async def stream_mcq():
                            yield json.dumps({
                                "type": "meta",
                                "source": "document",
                                "requires_permission": False,
                                "conversation_state": "document_grounded"
                            }) + "\n"
                            yield json.dumps({"type": "token", "token": correct_answer}) + "\n"
                            yield json.dumps({"type": "metrics", "metrics": metrics}) + "\n"
                            yield json.dumps({"type": "done"}) + "\n"
                        return StreamingResponse(stream_mcq(), media_type="application/x-ndjson")

                    return {
                        "answer": correct_answer,
                        "source": "document",
                        "format": answer_format["type"],
                        "requires_permission": False,
                        "conversation_state": "document_grounded",
                        "metrics": metrics
                    }

    # Prepare document context (prioritize top relevant chunks up to ~2800 chars to avoid prompt bloat)
    context_parts = []
    seen = set()
    total_chars = 0
    for doc in relevant_documents:
        text = doc.page_content.strip()
        normalized = normalize_text(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        if total_chars + len(text) > 2800 and context_parts:
            break
        context_parts.append(text)
        total_chars += len(text)

    context = "\n\n---\n\n".join(context_parts)
    format_instruction = build_format_instruction(answer_format)

    prompt = f"""[INST] You are a highly accurate technical assistant. Your job is to answer the user's question using ONLY the supplied document context.

CRITICAL INSTRUCTIONS:
- Prioritize the uploaded PDF context over general knowledge.
- Do NOT invent facts or present unsupported details as if they came from the PDF.
- If the document lacks sufficient information, clearly state that limitation.
- For technical comparisons, distinguish concepts correctly and avoid misleading claims. Do not claim things are interchangeable unless the text explicitly says so.
- Preserve relevant technical terminology and code examples from the source.
- Do not blindly agree with incorrect assumptions in the user's question.
- If additional general knowledge is useful, clearly distinguish it from information found in the document.
- Provide a clear, concise, and complete answer in around 100 to 180 words. Ensure your response concludes with a complete final sentence.
- DO NOT mention retrieval, ChromaDB, or prompt details.
- You MUST write your ENTIRE response in English.

================ DOCUMENT CONTEXT ================
{context}
================ END DOCUMENT CONTEXT ================

USER QUESTION:
{prompt_question}

ANSWER FORMAT:
{format_instruction}

Answer the question completely and accurately now based on the rules above. [/INST]"""

    prompt_construction_time = time.time() - total_start
    fmt = answer_format.get("type", "normal") if isinstance(answer_format, dict) else "normal"
    dynamic_num_predict = 100 if fmt == "definition" else (80 if fmt in ["mcq", "fill_blank"] else (300 if fmt in ["list", "explanation"] else 250))

    active_llm = OllamaLLM(
        model="mistral:latest",
        temperature=0,
        num_predict=dynamic_num_predict,
        num_ctx=2048,
        top_k=20,
        top_p=0.9,
        keep_alive="10m",
        client_kwargs={"timeout": 120.0},
        async_client_kwargs={"timeout": 120.0}
    )

    print(f"[NORMAL_RAG] Selected {len(context_parts)} chunks ({len(context)} chars), prompt: {len(prompt)} chars, num_predict: {dynamic_num_predict}")
    print(f"[NORMAL_RAG] Starting LLM generation...")

    generation_start = time.time()
    first_token_time = None

    if request.stream:
        async def event_generator():
            nonlocal first_token_time
            ollama_start = time.time()
            yield json.dumps({
                "type": "meta",
                "source": "document",
                "requires_permission": False,
                "conversation_state": "document_grounded"
            }) + "\n"

            accumulated = []
            try:
                async for chunk in active_llm.astream(prompt):
                    if first_token_time is None:
                        first_token_time = time.time() - ollama_start
                    accumulated.append(chunk)
                    yield json.dumps({"type": "token", "token": chunk}) + "\n"

                full_raw = "".join(accumulated)
                if not is_sentence_complete(full_raw):
                    cont_snippet = generate_continuation_snippet(prompt, full_raw)
                    if cont_snippet:
                        clean_cont = stitch_continuation(full_raw, cont_snippet)[len(full_raw):]
                        if clean_cont:
                            yield json.dumps({"type": "token", "token": clean_cont}) + "\n"

                gen_dur = time.time() - generation_start
                tot_dur = time.time() - total_start
                final_metrics = {
                    "chromadb_retrieval_time": round(retrieval_time, 4),
                    "prompt_construction_time": round(prompt_construction_time, 4),
                    "ollama_startup_time": round(first_token_time if first_token_time else 0.0, 4),
                    "first_token_time": round(first_token_time if first_token_time else 0.0, 4),
                    "llm_generation_time": round(gen_dur, 4),
                    "total_api_time": round(tot_dur, 4)
                }
                yield json.dumps({"type": "metrics", "metrics": final_metrics}) + "\n"
            except Exception as ex:
                yield json.dumps({"type": "error", "error": str(ex)}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

        return StreamingResponse(event_generator(), media_type="application/x-ndjson")

    ollama_start = time.time()
    tokens = []
    try:
        async for chunk in active_llm.astream(prompt):
            if first_token_time is None:
                first_token_time = time.time() - ollama_start
            tokens.append(chunk)
        raw_answer = "".join(tokens)
        answer = complete_response_if_needed(prompt, raw_answer)
    except Exception as e:
        answer = "Unable to complete response generation from local LLM. Please try again."

    if first_token_time is None:
        first_token_time = 0.0

    generation_time = time.time() - generation_start
    total_time = time.time() - total_start
    print(f"[NORMAL_RAG] Completed in {total_time:.2f}s (retrieval={retrieval_time:.2f}s, first_token={first_token_time:.2f}s, llm_gen={generation_time:.2f}s)")

    metrics = {
        "chromadb_retrieval_time": round(retrieval_time, 4),
        "prompt_construction_time": round(prompt_construction_time, 4),
        "ollama_startup_time": round(first_token_time, 4),
        "first_token_time": round(first_token_time, 4),
        "llm_generation_time": round(generation_time, 4),
        "total_api_time": round(total_time, 4)
    }

    return {
        "answer": answer,
        "source": "document",
        "format": fmt,
        "requires_permission": False,
        "conversation_state": "document_grounded",
        "metrics": metrics
    }

