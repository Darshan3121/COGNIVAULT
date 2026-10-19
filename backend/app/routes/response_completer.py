import re
from langchain_ollama import OllamaLLM

def is_sentence_complete(text: str) -> bool:
    """Checks whether the text ends with complete sentence punctuation or closing delimiter."""
    trimmed = text.strip()
    if not trimmed:
        return True
    return trimmed[-1] in {'.', '!', '?', '"', "'", '`', '}', '>'}

def stitch_continuation(original_text: str, continuation_text: str) -> str:
    """
    Stitches continuation text onto original_text, deduplicating any overlapping
    words at the boundary seam.
    """
    continuation_text = continuation_text.strip()
    if not continuation_text:
        return original_text.strip()

    orig_words = original_text.strip().split()
    cont_words = continuation_text.split()

    max_check = min(8, len(orig_words), len(cont_words))
    overlap_len = 0
    for k in range(max_check, 0, -1):
        tail_slice = [w.lower().strip(".,!?;:\"'") for w in orig_words[-k:]]
        head_slice = [w.lower().strip(".,!?;:\"'") for w in cont_words[:k]]
        if tail_slice == head_slice:
            overlap_len = k
            break

    if overlap_len > 0:
        continuation_text = " ".join(cont_words[overlap_len:])

    return (original_text.rstrip() + " " + continuation_text.lstrip()).strip()

def generate_continuation_snippet(original_prompt: str, generated_text: str) -> str:
    """
    Invokes Mistral for a single bounded continuation step (max 1 attempt, no infinite loop)
    to conclude the cut-off text in 1-2 sentences without repeating words.
    """
    tail = generated_text[-200:] if len(generated_text) > 200 else generated_text
    continuation_prompt = f"""[INST] You were generating a response to:
"{original_prompt[:150]}"

Your response was cut off at the token limit. Here is the end of the text generated so far:
"...{tail}"

Continue directly from the exact point of cutoff to finish the sentence and bring the response to a natural conclusion in 1 to 2 sentences.
CRITICAL: Do NOT repeat any words from the text above. Output ONLY the continuation text that completes the thought. [/INST]"""

    cont_llm = OllamaLLM(
        model="mistral:latest",
        temperature=0,
        num_predict=120,
        num_ctx=2048,
        top_k=20,
        top_p=0.9,
        keep_alive="10m",
        client_kwargs={"timeout": 60.0},
        async_client_kwargs={"timeout": 60.0}
    )
    try:
        cont_text = cont_llm.invoke(continuation_prompt).strip()
        if cont_text.startswith('"') and cont_text.endswith('"') and len(cont_text) > 2:
            cont_text = cont_text[1:-1].strip()
        return cont_text
    except Exception as e:
        print(f"[CONTINUATION ERROR] {e}")
        return ""

def complete_response_if_needed(prompt: str, text: str) -> str:
    """
    Ensures response concludes with a complete sentence.
    If truncated mid-sentence by length limit, triggers a safe, bounded continuation.
    Avoids infinite retries by attempting continuation at most once.
    """
    if is_sentence_complete(text):
        return text

    print(f"[SAFE CONTINUATION TRIGGERED] Output ended mid-sentence: '{text[-60:]}'")
    cont_text = generate_continuation_snippet(prompt, text)
    if cont_text:
        stitched = stitch_continuation(text, cont_text)
    else:
        stitched = text

    if not is_sentence_complete(stitched):
        last_punct = max(stitched.rfind('.'), stitched.rfind('!'), stitched.rfind('?'))
        if last_punct > 100:
            stitched = stitched[:last_punct + 1].strip()
        else:
            stitched = stitched.rstrip() + "."

    return stitched
