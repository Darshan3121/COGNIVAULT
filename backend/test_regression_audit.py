import json
import urllib.request
import re
import sys
import time
from app.routes.mcq_validator import (
    MCQCandidate,
    validate_mcq_candidate,
    parse_mcq_candidates,
    format_user_facing_mcqs,
    format_complete_mcqs,
    format_verified_answers,
    sanitize_overview_claims,
    improve_weak_explanation,
    validate_explanation_accuracy,
    verify_java_code_semantics,
    verify_inheritance_hierarchy_accuracy
)

BASE_URL = "http://127.0.0.1:8000/api/chat"
JAVA_DOC_ID = "84c2ef71-3618-4d55-9ac9-4cbea54f757c"

def post_chat(data, timeout=600):
    q = data.get("question", "")[:60]
    print(f"  [post_chat] Querying: '{q}' (timeout={timeout}s)...")
    t0 = time.time()
    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        elapsed = time.time() - t0
        print(f"  [post_chat] Finished in {elapsed:.2f}s (status {resp.status})")
        return json.loads(resp.read().decode("utf-8"))

def run_audit():
    results = {}
    
    # -------------------------------------------------------------
    # TEST 1: Code-dependent MCQ that must display its code snippet
    # -------------------------------------------------------------
    print("\n--- TEST 1: Code-dependent MCQ Snippet Verification ---")
    try:
        doc_chunks = ["Java data types. int dec = 100;"]
        bad_cand = MCQCandidate(
            question="What is the data type of the variable `dec` in the following code snippet?",
            options={"A": "int", "B": "double", "C": "float", "D": "byte"},
            correct="A",
            explanation="dec is an integer."
        )
        is_val_bad, reason_bad = validate_mcq_candidate(bad_cand, doc_chunks)
        assert not is_val_bad, f"Failed: Bad candidate should be rejected! Reason: {reason_bad}"
        
        good_cand = MCQCandidate(
            question="What is the data type of the variable `dec` in the following code snippet?\n```java\nint dec = 100;\n```",
            options={"A": "int", "B": "double", "C": "float", "D": "byte"},
            correct="A",
            explanation="dec is an integer."
        )
        is_val_good, reason_good = validate_mcq_candidate(good_cand, doc_chunks)
        assert is_val_good, f"Failed: Good candidate should be accepted! Reason: {reason_good}"
        print("  [PASS] Code-dependent MCQ without code snippet rejected; with code snippet accepted.")
        results["Test 1: Code-dependent MCQ snippet requirement"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 1: Code-dependent MCQ snippet requirement"] = f"FAIL: {e}"

    # -------------------------------------------------------------
    # TEST 2: Ambiguous MCQ where two options are valid (2D array)
    # -------------------------------------------------------------
    print("\n--- TEST 2: Ambiguous 2D Array Syntax Rejection ---")
    try:
        doc_chunks = ["Java supports 1D and 2D arrays."]
        ambiguous_cand = MCQCandidate(
            question="Which of the following is a valid syntax to declare a 2D array in Java?",
            options={
                "A": "int[][] a = new int[2][3];",
                "B": "int[][] a = new int[2][];",
                "C": "int a[2][3] = new int[][];",
                "D": "int[][] a = new int[][3];"
            },
            correct="A",
            explanation="Both rectangular and ragged arrays are valid in Java."
        )
        is_val, reason = validate_mcq_candidate(ambiguous_cand, doc_chunks)
        assert not is_val, f"Failed: Ambiguous candidate should be rejected! Reason: {reason}"
        print(f"  [PASS] Ambiguous 2D array question correctly rejected: {reason}")
        results["Test 2: Ambiguous 2D array syntax rejection"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 2: Ambiguous 2D array syntax rejection"] = f"FAIL: {e}"

    # -------------------------------------------------------------
    # TEST 3: Constructor question without supporting class definition
    # -------------------------------------------------------------
    print("\n--- TEST 3: Constructor Question Context Check ---")
    try:
        doc_chunks = ["new Book().show(); new Book('Java', 450).show();"]
        bad_ctor = MCQCandidate(
            question="Which is a valid constructor of the Book class?",
            options={"A": "Book()", "B": "Book(String, int)", "C": "Book(int)", "D": "Book(double)"},
            correct="A",
            explanation="Book constructor"
        )
        is_val, reason = validate_mcq_candidate(bad_ctor, doc_chunks)
        assert not is_val, f"Failed: Constructor question without class definition should be rejected! Reason: {reason}"
        print(f"  [PASS] Constructor question without class definition context rejected: {reason}")
        results["Test 3: Constructor question without class definition"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 3: Constructor question without class definition"] = f"FAIL: {e}"

    # -------------------------------------------------------------
    # TEST 4: Overview that must not invent the number of questions
    # -------------------------------------------------------------
    print("\n--- TEST 4: Overview Question Count Grounding ---")
    try:
        # Document without question count mentioned
        doc_no_counts = ["Java thread tutorial. Explains Thread class and Runnable interface."]
        hallucinated_text = "The document is a question bank consisting of 100 questions covering Java."
        sanitized = sanitize_overview_claims(hallucinated_text, doc_no_counts)
        assert "100 questions" not in sanitized, f"Failed: 100 questions should have been removed: {sanitized}"
        print("  [PASS] Unverified question count correctly stripped from overview.")
        
        # Document with question count mentioned
        doc_with_counts = ["Done — all 100 questions answered with simple, short code."]
        valid_text = "The document contains 100 questions with short Java solutions."
        kept = sanitize_overview_claims(valid_text, doc_with_counts)
        assert "100 questions" in kept, f"Failed: Verified count should be preserved: {kept}"
        print("  [PASS] Verified question count correctly preserved.")
        results["Test 4: Overview question count grounding"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 4: Overview question count grounding"] = f"FAIL: {e}"

    # -------------------------------------------------------------
    # TEST 7: Normal RAG chat and unsupported-question handling
    # (Fastest live API calls first)
    # -------------------------------------------------------------
    print("\n--- TEST 7: Normal RAG Grounded Answer and Refusal Guard ---")
    try:
        # Grounded query
        payload_grounded = {
            "document_id": JAVA_DOC_ID,
            "question": "How do you create threads in Java according to this PDF?",
            "stream": False,
            "messages": []
        }
        res_g = post_chat(payload_grounded)
        ans_g = res_g.get("answer", "")
        assert "thread" in ans_g.lower() or "runnable" in ans_g.lower(), f"Unexpected answer: {ans_g}"
        print("  [PASS] Grounded question answered accurately based on document.")

        # Unsupported query
        payload_unsupported = {
            "document_id": JAVA_DOC_ID,
            "question": "How do you bake a chocolate cake?",
            "stream": False,
            "messages": []
        }
        res_u = post_chat(payload_unsupported)
        ans_u = res_u.get("answer", "")
        assert ("couldn't find" in ans_u.lower() or "not found" in ans_u.lower() or "permission" in ans_u.lower() or res_u.get("source") == "document_not_found"), f"Expected refusal, got: {ans_u}"
        print("  [PASS] Unsupported question correctly refused by document boundary.")
        results["Test 7: Normal RAG chat and unsupported-question handling"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 7: Normal RAG chat and unsupported-question handling"] = f"FAIL: {e}"

    # -------------------------------------------------------------
    # TEST 8: Document isolation (cannot leak chunks to another doc)
    # -------------------------------------------------------------
    print("\n--- TEST 8: Document Isolation ---")
    try:
        # Query Java doc asking about Hindi test document or Kannada text
        payload_isolation = {
            "document_id": JAVA_DOC_ID,
            "question": "What does the Hindi document say about historical monuments?",
            "stream": False,
            "messages": []
        }
        res_iso = post_chat(payload_isolation)
        ans_iso = res_iso.get("answer", "")
        # Should not find Hindi content in Java PDF
        assert ("couldn't find" in ans_iso.lower() or "not" in ans_iso.lower()), f"Expected refusal/isolation, got: {ans_iso}"
        print("  [PASS] Document isolation enforced: no cross-document chunk leakage.")
        results["Test 8: Document isolation"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 8: Document isolation"] = f"FAIL: {e}"

    # -------------------------------------------------------------
    # TEST 5: Exactly three valid MCQs
    # -------------------------------------------------------------
    print("\n--- TEST 5: Live API Exactly Three Valid MCQs ---")
    try:
        payload_mcq = {
            "document_id": JAVA_DOC_ID,
            "question": "Give me three MCQs based on this PDF.",
            "stream": False,
            "messages": []
        }
        res_mcq = post_chat(payload_mcq)
        ans_mcq = res_mcq.get("answer", "")
        print(f"MCQ Response Preview:\n{ans_mcq[:300]}...\n")
        
        # Count questions
        q_count = len(re.findall(r"(?m)^\s*\d+\.\s+", ans_mcq))
        assert q_count == 3, f"Expected exactly 3 MCQs, got {q_count}"
        
        # Check blank lines between options
        assert "\nA) " in ans_mcq or "\nA)" in ans_mcq
        assert "\n\nB) " in ans_mcq or "\nB) " in ans_mcq
        
        # Check no internal reasoning leaked
        assert "CORRECT:" not in ans_mcq
        assert "EXPLANATION:" not in ans_mcq
        assert "SOURCE_QUOTE:" not in ans_mcq
        
        # Check no ambiguous return type questions or fallacies leaked
        if "return type" in ans_mcq.lower():
            assert "```" in ans_mcq or "{" in ans_mcq, "Ambiguous return type question leaked into standalone MCQs!"
        assert "have no return type" not in ans_mcq.lower()
        assert "different data types" not in ans_mcq.lower()
        
        print(f"  [PASS] Exactly 3 validated MCQs returned with no internal leaks.")
        results["Test 5: Exactly three valid MCQs"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 5: Exactly three valid MCQs"] = f"FAIL: {e}"

    # -------------------------------------------------------------
    # TEST 6: Combined overview plus exactly three MCQs with answers
    # -------------------------------------------------------------
    print("\n--- TEST 6: Live API Combined Overview + Three MCQs with Answers ---")
    try:
        payload_comb = {
            "document_id": JAVA_DOC_ID,
            "question": "Tell me about the PDF, then generate exactly three MCQs based on it.",
            "stream": False,
            "messages": []
        }
        res_comb = post_chat(payload_comb)
        ans_comb = res_comb.get("answer", "")
        print(f"Combined Response Length: {len(ans_comb)} chars")
        print(f"Combined Response Preview:\n{ans_comb[:500]}...\n")
        
        # Verify source attribution
        assert res_comb.get("source") == "document", f"Expected source=document, got {res_comb.get('source')}"
        
        # Must have overview paragraph followed by MCQs
        first_q_pos = ans_comb.find("1. ")
        assert first_q_pos > 50, f"Expected overview paragraph before question 1, but found first question at index {first_q_pos}"
        
        overview_part = ans_comb[:first_q_pos].strip()
        mcq_part = ans_comb[first_q_pos:].strip()
        
        # Verify overview quality
        overview_words = len(overview_part.split())
        assert overview_words >= 40, f"Overview too short: {overview_words} words (expected >= 40)"
        assert overview_part[-1] in {'.', '!', '?'}, f"Overview does not end with complete sentence punctuation: '{overview_part[-30:]}'"
        print(f"  [OK] Overview: {overview_words} words, ends with '{overview_part[-1]}'")
        
        # Count questions
        q_count_comb = len(re.findall(r"(?m)^\s*\d+\.\s+", mcq_part))
        assert q_count_comb == 3, f"Expected exactly 3 MCQs in combined output, got {q_count_comb}"
        
        # Verify each MCQ has all 4 options
        for opt in ["A)", "B)", "C)", "D)"]:
            opt_count = mcq_part.count(opt)
            assert opt_count >= 3, f"Expected at least 3 occurrences of {opt} (one per MCQ), got {opt_count}"
        
        # Verify correct answers are present (not stripped)
        correct_answer_matches = re.findall(r"\*\*Correct Answer:\s*[A-D]\)", ans_comb)
        assert len(correct_answer_matches) == 3, f"Expected 3 correct answer lines, got {len(correct_answer_matches)}: {correct_answer_matches}"
        print(f"  [OK] All 3 correct answers present: {correct_answer_matches}")
        
        # Verify explanations are present
        explanation_matches = re.findall(r"\*\*Explanation:\*\*", ans_comb)
        assert len(explanation_matches) == 3, f"Expected 3 explanation lines, got {len(explanation_matches)}"
        print(f"  [OK] All 3 explanations present")
        
        # Verify no internal metadata leaked
        assert "SOURCE_QUOTE:" not in ans_comb, "SOURCE_QUOTE leaked into combined response!"
        assert "supporting_chunk" not in ans_comb.lower(), "supporting_chunk leaked into combined response!"
        
        # Verify no technical fallacies or ambiguous questions leaked into combined response
        assert "have no return type" not in ans_comb.lower(), "Fallacy 'have no return type' leaked into combined response!"
        assert "different data types" not in ans_comb.lower(), "Fallacy 'different data types' leaked into combined response!"
        for exp_line in re.findall(r"\*\*Explanation:\*\*\s*(.*)", ans_comb):
            exp_line_lower = exp_line.lower()
            assert "have no return type" not in exp_line_lower, f"Fallacy 'have no return type' in explanation: {exp_line}"
            assert "different data types" not in exp_line_lower, f"Fallacy 'different data types' in explanation: {exp_line}"
            if "implements" in exp_line_lower:
                assert "defines an interface" not in exp_line_lower and "define an interface" not in exp_line_lower, f"Defect 'defines an interface' in implements explanation: {exp_line}"
                assert "declare that it implements an interface" in exp_line_lower or "uses 'implements'" in exp_line_lower or "uses the 'implements' keyword" in exp_line_lower, f"Implements explanation must explain that a class uses implements to declare that it implements an interface: {exp_line}"
            if "interface is a blueprint for defining methods" in exp_line_lower:
                assert "within the scope of the document" in exp_line_lower or "contract" in exp_line_lower or "abstract" in exp_line_lower, f"Oversimplified definition in explanation: {exp_line}"
        if "return type" in mcq_part.lower():
            # If a return type question appears, it MUST have a code block declaring the method
            assert "```" in mcq_part or "{" in mcq_part, "Generic return type question without code block leaked into combined response!"

        # Verify no mid-sentence cutoff in the combined response
        trimmed = ans_comb.strip()
        assert trimmed[-1] in {'.', '!', '?', '"', "'", ')', '*'}, f"Combined response ends abruptly: '{trimmed[-30:]}'"
        
        print(f"  [PASS] Combined overview ({overview_words} words) + 3 MCQs with answers and explanations returned correctly.")
        results["Test 6: Combined overview plus three MCQs with answers"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 6: Combined overview plus three MCQs with answers"] = f"FAIL: {e}"

    # -------------------------------------------------------------
    # TEST 13: Live API Single PDF Document Overview
    # -------------------------------------------------------------
    print("\n--- TEST 13: Live API Single PDF Document Overview ---")
    try:
        payload_overview = {
            "document_id": JAVA_DOC_ID,
            "question": "Tell me about the PDF",
            "stream": False,
            "messages": []
        }
        res_ov = post_chat(payload_overview)
        ans_ov = res_ov.get("answer", "").strip()
        print(f"Overview Preview (first 150): {ans_ov[:150]}...")
        print(f"Overview Ending: {repr(ans_ov[-100:])}")
        assert res_ov.get("source") == "document", f"Expected source=document, got {res_ov.get('source')}"
        assert len(ans_ov) > 200, f"Expected overview length > 200 chars, got {len(ans_ov)}"
        assert ans_ov[-1] in {'.', '!', '?'}, f"Overview does not end with complete sentence punctuation: '{ans_ov[-20:]}'"
        assert "100 questions" not in ans_ov, "Unverified question count leaked into overview!"
        print("  [PASS] Single PDF overview returned complete, fluent, and grounded paragraph.")
        results["Test 13: Single PDF document overview"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 13: Single PDF document overview"] = f"FAIL: {e}"

    # -------------------------------------------------------------
    # TEST 9: Ambiguous interface return type MCQ rejection
    # -------------------------------------------------------------
    print("\n--- TEST 9: Ambiguous Interface Return Type MCQ Rejection ---")
    try:
        doc_chunks = ["Java interfaces define methods. interface Vehicle { void start(); }"]
        
        # Variant 1: "a method defined in an interface"
        q1 = MCQCandidate(
            question="What is the return type of a method defined in an interface?",
            options={"A": "void", "B": "int", "C": "boolean", "D": "double"},
            correct="A",
            explanation="The start method returns void."
        )
        is_val1, r1 = validate_mcq_candidate(q1, doc_chunks)
        assert not is_val1, f"Failed: 'a method defined in an interface' should be rejected! Reason: {r1}"
        
        # Variant 2: "the method() defined in an interface"
        q2 = MCQCandidate(
            question="What is the return type of the method() defined in an interface?",
            options={"A": "int", "B": "void", "C": "boolean", "D": "String"},
            correct="B",
            explanation="Interface methods are by default abstract and have no return type."
        )
        is_val2, r2 = validate_mcq_candidate(q2, doc_chunks)
        assert not is_val2, f"Failed: 'the method() defined in an interface' should be rejected! Reason: {r2}"

        # Variant 3: "the `method()` defined in an interface"
        q3 = MCQCandidate(
            question="What is the return type of the `method()` defined in an interface?",
            options={"A": "int", "B": "void", "C": "boolean", "D": "String"},
            correct="B",
            explanation="Methods in interface."
        )
        is_val3, r3 = validate_mcq_candidate(q3, doc_chunks)
        assert not is_val3, f"Failed: 'the `method()` defined in an interface' should be rejected! Reason: {r3}"

        # Variant 4: Exact user prompt: "What is the return type of the `method()` in an interface?"
        q4 = MCQCandidate(
            question="What is the return type of the `method()` in an interface?",
            options={"A": "int", "B": "void", "C": "String", "D": "boolean"},
            correct="B",
            explanation="Interface methods are by default abstract and have no return type specified, so they return void."
        )
        is_val4, r4 = validate_mcq_candidate(q4, doc_chunks)
        assert not is_val4, f"Failed: Exact user prompt MCQ should be rejected! Reason: {r4}"

        # Direct explanation accuracy test for exact user prompt explanation
        from app.routes.mcq_validator import validate_explanation_accuracy
        is_acc, acc_reason = validate_explanation_accuracy(q4)
        assert not is_acc, f"Failed: Exact fallacy explanation should be rejected! Reason: {acc_reason}"

        print(f"  [PASS] All variants of ambiguous interface return type questions correctly rejected.")
        results["Test 9: Ambiguous interface return type MCQ rejection"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 9: Ambiguous interface return type MCQ rejection"] = f"FAIL: {e}"

    # -------------------------------------------------------------
    # TEST 10: Valid single-answer MCQ acceptance and internal evidence retention
    # -------------------------------------------------------------
    print("\n--- TEST 10: Valid Single-Answer MCQ and Evidence Retention ---")
    try:
        doc_chunks = ["To create a thread, you can extend the Thread class or implement the Runnable interface."]
        valid_mcq = MCQCandidate(
            question="Which interface must be implemented to create a thread using the interface approach in Java?",
            options={"A": "Runnable", "B": "Thread", "C": "Callable", "D": "Process"},
            correct="A",
            explanation="The Runnable interface is implemented to create a thread.",
            source_quote="implement the Runnable interface"
        )
        is_val, reason = validate_mcq_candidate(valid_mcq, doc_chunks)
        assert is_val, f"Failed: Valid MCQ should be accepted! Reason: {reason}"
        assert valid_mcq.supporting_chunk, "Failed: supporting_chunk was not retained internally!"
        
        # Verify format_user_facing_mcqs does NOT leak supporting_chunk or answers
        user_facing = format_user_facing_mcqs([valid_mcq])
        assert "CORRECT:" not in user_facing
        assert "EXPLANATION:" not in user_facing
        assert "SOURCE_QUOTE:" not in user_facing
        assert valid_mcq.supporting_chunk not in user_facing or len(valid_mcq.supporting_chunk) < 10
        print(f"  [PASS] Valid single-answer MCQ accepted and internal evidence retained without UI leak.")
        results["Test 10: Valid single-answer MCQ and evidence retention"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 10: Valid single-answer MCQ and evidence retention"] = f"FAIL: {e}"

    # -------------------------------------------------------------
    # TEST 11: General knowledge response completeness (Horror Story)
    # -------------------------------------------------------------
    print("\n--- TEST 11: General Knowledge Response Completeness ---")
    try:
        payload_story = {
            "document_id": None,
            "question": "Write a short horror story",
            "stream": False
        }
        res_story = post_chat(payload_story)
        ans_story = res_story.get("answer", "")
        print(f"Story Response Length: {len(ans_story)}")
        print(f"Story Preview (first 150): {ans_story[:150]}...")
        print(f"Story Ending: {repr(ans_story[-100:])}")

        assert res_story.get("source") == "general_knowledge", f"Expected general_knowledge source, got {res_story.get('source')}"
        assert len(ans_story) > 350, f"Expected story length > 350 chars, got {len(ans_story)}"
        
        # Verify it concludes naturally with proper punctuation
        trimmed_ans = ans_story.strip()
        assert trimmed_ans[-1] in {'.', '!', '?', '"', "'"}, f"Story does not end with sentence punctuation: '{trimmed_ans[-20:]}'"
        assert not trimmed_ans.endswith("they began"), "Story truncated mid-sentence at 'they began'!"
        assert not trimmed_ans.endswith("I want"), "Story truncated mid-sentence at 'I want'!"

        print("  [PASS] General knowledge story concluded naturally with complete sentences.")
        results["Test 11: General knowledge response completeness"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 11: General knowledge response completeness"] = f"FAIL: {e}"

    # -------------------------------------------------------------
    # TEST 12: General knowledge fallback with document attached and permission flow
    # -------------------------------------------------------------
    print("\n--- TEST 12: Document Fallback Permission and Execution ---")
    try:
        # Step 1: Query with document attached for unrelated topic -> requires permission
        payload_step1 = {
            "document_id": JAVA_DOC_ID,
            "question": "How do you bake a chocolate cake?",
            "stream": False
        }
        res_step1 = post_chat(payload_step1)
        assert res_step1.get("requires_permission") is True, f"Expected requires_permission=True, got {res_step1}"
        assert res_step1.get("source") == "document_not_found", f"Expected source=document_not_found, got {res_step1}"
        print("  [PASS] Step 1: Unrelated question correctly triggered permission prompt.")

        # Step 2: Grant permission -> answers with general knowledge
        payload_step2 = {
            "document_id": JAVA_DOC_ID,
            "question": "How do you bake a chocolate cake?",
            "allow_general": True,
            "permission_response": "yes",
            "stream": False
        }
        res_step2 = post_chat(payload_step2)
        ans_step2 = res_step2.get("answer", "").strip()
        assert res_step2.get("requires_permission") is False, f"Expected requires_permission=False, got {res_step2}"
        assert res_step2.get("source") == "general_knowledge", f"Expected source=general_knowledge, got {res_step2.get('source')}"
        assert len(ans_step2) > 350, f"Expected response length > 350 chars, got {len(ans_step2)}"
        assert ans_step2[-1] in {'.', '!', '?', '"', "'"}, f"Fallback response does not end with punctuation: '{ans_step2[-20:]}'"
        print("  [PASS] Step 2: Permission granted correctly routed to general knowledge generation.")
        results["Test 12: Document fallback permission and execution"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 12: Document fallback permission and execution"] = f"FAIL: {e}"

    # -------------------------------------------------------------
    # TEST 14: Java Primitive and Code-Output Semantic Evaluation
    # -------------------------------------------------------------
    print("\n--- TEST 14: Java Code-Output Semantic Verification ---")
    try:
        doc_chunks = ["int[] a = {1, 2, 3}; System.out.println(a[1]); Output: 2. String s = \"Java\"; s.concat(\" SE\"); System.out.println(s); Output: Java"]
        
        # Array indexing: valid candidate
        cand_arr_ok = MCQCandidate(
            question="What will be the output of the following code?\n```java\nint[] a = {1, 2, 3};\nSystem.out.println(a[1]);\n```",
            options={"A": "1", "B": "2", "C": "3", "D": "ArrayIndexOutOfBoundsException"},
            correct="B",
            explanation="Java arrays are zero-indexed."
        )
        is_val_arr_ok, r_arr_ok = validate_mcq_candidate(cand_arr_ok, doc_chunks)
        assert is_val_arr_ok, f"Failed: Correct array indexing question should be accepted! Reason: {r_arr_ok}"

        # Array indexing: wrong output candidate
        cand_arr_bad = MCQCandidate(
            question="What will be the output of the following code?\n```java\nint[] a = {1, 2, 3};\nSystem.out.println(a[1]);\n```",
            options={"A": "1", "B": "2", "C": "3", "D": "0"},
            correct="A",  # WRONG: a[1] is 2, not 1
            explanation="Index 1 is the first element."
        )
        is_val_arr_bad, r_arr_bad = validate_mcq_candidate(cand_arr_bad, doc_chunks)
        assert not is_val_arr_bad, "Failed: Array question with incorrect output should be rejected!"

        # String immutability: valid candidate
        cand_str_ok = MCQCandidate(
            question='What will be the output of the following code?\n```java\nString s = "Java";\ns.concat(" SE");\nSystem.out.println(s);\n```',
            options={"A": "Java", "B": "Java SE", "C": "SE", "D": "Compilation Error"},
            correct="A",
            explanation="Strings are immutable."
        )
        is_val_str_ok, r_str_ok = validate_mcq_candidate(cand_str_ok, doc_chunks)
        assert is_val_str_ok, f"Failed: Correct String immutability question should be accepted! Reason: {r_str_ok}"

        # String immutability: wrong output candidate
        cand_str_bad = MCQCandidate(
            question='What will be the output of the following code?\n```java\nString s = "Java";\ns.concat(" SE");\nSystem.out.println(s);\n```',
            options={"A": "Java", "B": "Java SE", "C": "SE", "D": "Compilation Error"},
            correct="B",  # WRONG: s was not reassigned
            explanation="concat appends SE."
        )
        is_val_str_bad, r_str_bad = validate_mcq_candidate(cand_str_bad, doc_chunks)
        assert not is_val_str_bad, "Failed: String immutability question with wrong output should be rejected!"

        # Max/Min array search: Exact user reported failure (A=89 12 vs C=89 7)
        q_max_min = (
            "What will be the output of the following code snippet?\n"
            "```java\n"
            "int[] a = {12, 45, 7, 89, 23};\n"
            "int max = a[0], min = a[0];\n"
            "for (int x : a) {\n"
            "    if (x > max) max = x;\n"
            "    if (x < min) min = x;\n"
            "}\n"
            "System.out.println(max + \" \" + min);\n"
            "```"
        )
        opts_max_min = {"A": "89 12", "B": "12 89", "C": "89 7", "D": "7 89"}
        doc_chunks_code = ["Q3. Find largest and smallest element in array: int[] a = {12, 45, 7, 89, 23}; int max = a[0], min = a[0]; for(int x:a)... System.out.println(max + \" \" + min);"]

        # Case 1: Bad candidate claiming Option A (89 12)
        cand_maxmin_bad = MCQCandidate(
            question=q_max_min,
            options=opts_max_min,
            correct="A",  # WRONG: 12 is initial a[0], true min is 7
            explanation="The code snippet finds the maximum and minimum values in the 1D array a and prints them.",
            source_quote="int[] a = {12, 45, 7, 89, 23};"
        )
        is_val_mm_bad, r_mm_bad = validate_mcq_candidate(cand_maxmin_bad, doc_chunks_code)
        assert not is_val_mm_bad, f"Failed: Max/min question claiming 89 12 (Option A) MUST be rejected! Got accepted."
        assert "Code output mismatch" in r_mm_bad or "89 7" in r_mm_bad, f"Unexpected rejection reason: {r_mm_bad}"

        # Case 2: Good candidate claiming Option C (89 7)
        cand_maxmin_good = MCQCandidate(
            question=q_max_min,
            options=opts_max_min,
            correct="C",  # CORRECT: max is 89, min is 7
            explanation="The code snippet finds the maximum and minimum values in the 1D array a and prints them.",
            source_quote="int[] a = {12, 45, 7, 89, 23};"
        )
        improve_weak_explanation(cand_maxmin_good, doc_chunks_code)
        is_val_mm_good, r_mm_good = validate_mcq_candidate(cand_maxmin_good, doc_chunks_code)
        assert is_val_mm_good, f"Failed: Correct max/min candidate (Option C: 89 7) MUST be accepted! Reason: {r_mm_good}"
        assert "largest element (89)" in cand_maxmin_good.explanation and "smallest element (7)" in cand_maxmin_good.explanation

        # Case 3: Explanation contradicts displayed correct option
        cand_mm_contra = MCQCandidate(
            question=q_max_min,
            options=opts_max_min,
            correct="A",
            explanation="The loop finds that maximum is 89 and minimum is 7.",
            source_quote="int[] a = {12, 45, 7, 89, 23};"
        )
        is_val_contra, r_contra = validate_mcq_candidate(cand_mm_contra, doc_chunks_code)
        assert not is_val_contra, f"Failed: Candidate where explanation contradicts displayed answer must be rejected!"

        # 2D array row-by-row iteration print
        q_2d_print = (
            "What is the output of the following code snippet?\n"
            "```java\n"
            "int[][] m = {{1,2},{3,4}};\n"
            "for (int i=0;i<2;i++){\n"
            " for (int j=0;j<2;j++) System.out.print(m[i][j]+\" \");\n"
            " System.out.println();\n"
            "}\n"
            "```"
        )
        opts_2d = {"A": "1 2 3 4", "B": "1 2 4 3", "C": "4 3 2 1", "D": "2 1 4 3"}
        doc_2d = ["2D array: int[][] m = {{1,2},{3,4}}; System.out.print elements"]
        cand_2d_bad = MCQCandidate(question=q_2d_print, options=opts_2d, correct="B", explanation="Row print.")
        is_v_2d_bad, _ = validate_mcq_candidate(cand_2d_bad, doc_2d)
        assert not is_v_2d_bad, "Failed: 2D array with wrong printed order should be rejected!"
        cand_2d_good = MCQCandidate(question=q_2d_print, options=opts_2d, correct="A", explanation="Row print.")
        is_v_2d_good, r_2d = validate_mcq_candidate(cand_2d_good, doc_2d)
        assert is_v_2d_good, f"Failed: Correct 2D array print should be accepted! Reason: {r_2d}"

        # Array sum evaluation
        q_sum = (
            "What will be the output of the following code?\n"
            "```java\n"
            "int[] a = {1, 2, 3, 4, 5};\n"
            "int sum = 0;\n"
            "for (int x : a) sum += x;\n"
            "System.out.println(sum);\n"
            "```"
        )
        opts_sum = {"A": "10", "B": "15", "C": "5", "D": "20"}
        doc_sum = ["Array sum: int[] a = {1, 2, 3, 4, 5}; sum += x;"]
        cand_sum_bad = MCQCandidate(question=q_sum, options=opts_sum, correct="A", explanation="Sum of array.")
        is_v_sum_bad, _ = validate_mcq_candidate(cand_sum_bad, doc_sum)
        assert not is_v_sum_bad, "Failed: Array sum with wrong value should be rejected!"
        cand_sum_good = MCQCandidate(question=q_sum, options=opts_sum, correct="B", explanation="Sum of array.")
        is_v_sum_good, r_sum = validate_mcq_candidate(cand_sum_good, doc_sum)
        assert is_v_sum_good, f"Failed: Correct array sum should be accepted! Reason: {r_sum}"

        print("  [PASS] Java code-output semantics correctly verified: array indexing, String immutability, max/min loop, 2D matrix print, array sum, and contradiction guards.")
        results["Test 14: Java code-output semantic verification"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 14: Java code-output semantic verification"] = f"FAIL: {e}"

    # -------------------------------------------------------------
    # TEST 15: 1D vs 2D Array Distinction and Homogeneity Validation
    # -------------------------------------------------------------
    print("\n--- TEST 15: 1D vs 2D Array Distinction and Homogeneity Validation ---")
    try:
        doc_chunks = ["Array = same-type elements in contiguous memory. 1D array: int[] a = {10,20,30}; 2D array: int[][] m = {{1,2},{3,4}};"]
        
        # Heterogeneous array claim (invalid in Java)
        cand_hetero = MCQCandidate(
            question="What is the difference between a 1D array and a 2D array in Java?",
            options={
                "A": "A 1D array stores elements of the same type, while a 2D array stores elements of different types.",
                "B": "Both store only one element.",
                "C": "1D arrays cannot store integers.",
                "D": "There is no difference."
            },
            correct="A",
            explanation="A 2D array can store different data types."
        )
        is_val_hetero, r_hetero = validate_mcq_candidate(cand_hetero, doc_chunks)
        assert not is_val_hetero, "Failed: Candidate claiming 2D arrays store different types should be rejected!"

        # Valid 1D vs 2D array question with weak explanation -> auto-improved
        cand_valid_arr = MCQCandidate(
            question="What is the difference between a 1D array and a 2D array in Java?",
            options={
                "A": "A 1D array stores elements in a single row, while a 2D array stores elements in rows and columns.",
                "B": "A 1D array cannot be traversed.",
                "C": "A 2D array cannot hold numbers.",
                "D": "There is no difference."
            },
            correct="A",
            explanation="A 1D array stores elements in a single row, while a 2D array stores elements in rows and columns."
        )
        is_val_valid_arr, r_valid_arr = validate_mcq_candidate(cand_valid_arr, doc_chunks)
        assert is_val_valid_arr, f"Failed: Valid 1D vs 2D array question should be accepted! Reason: {r_valid_arr}"
        
        # Test explanation improvement
        improve_weak_explanation(cand_valid_arr, doc_chunks)
        exp_text = cand_valid_arr.explanation.lower()
        assert "array of 1d arrays" in exp_text or "ragged" in exp_text, f"Explanation was not properly enhanced! Got: {cand_valid_arr.explanation}"

        print("  [PASS] 1D vs 2D array homogeneity enforced and weak explanation successfully enhanced.")
        results["Test 15: 1D vs 2D array distinction and homogeneity"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 15: 1D vs 2D array distinction and homogeneity"] = f"FAIL: {e}"

    # -------------------------------------------------------------
    # TEST 16: Explanation Accuracy and Fallacy Rejection
    # -------------------------------------------------------------
    print("\n--- TEST 16: Explanation Accuracy and Fallacy Rejection ---")
    try:
        doc_chunks = ["Java interfaces define method declarations. interface Vehicle { void start(); }"]
        
        # Fallacy: "Interface methods have no return type"
        cand_fallacy1 = MCQCandidate(
            question="Which keyword is used to implement an interface in Java?",
            options={"A": "implements", "B": "extends", "C": "interface", "D": "abstract"},
            correct="A",
            explanation="The implements keyword is used. Interface methods are by default abstract and have no return type."
        )
        is_val_f1, r_f1 = validate_mcq_candidate(cand_fallacy1, doc_chunks)
        assert not is_val_f1, f"Failed: Fallacy 'have no return type' in explanation should be rejected! Reason: {r_f1}"

        # Fallacy: "Strings are mutable"
        cand_fallacy2 = MCQCandidate(
            question="Which method is used to compare two String objects for equality?",
            options={"A": "equals()", "B": "==", "C": "compareTo()", "D": "compare()"},
            correct="A",
            explanation="The equals method is used because Strings are mutable."
        )
        is_val_f2, r_f2 = validate_mcq_candidate(cand_fallacy2, doc_chunks)
        assert not is_val_f2, f"Failed: Fallacy 'Strings are mutable' in explanation should be rejected! Reason: {r_f2}"

        # Explanation referencing wrong option letter
        cand_wrong_ref = MCQCandidate(
            question="Which package is imported by default in every Java program?",
            options={"A": "java.lang", "B": "java.util", "C": "java.io", "D": "java.net"},
            correct="A",
            explanation="Option B is correct because java.lang is imported by default."
        )
        is_val_wref, r_wref = validate_mcq_candidate(cand_wrong_ref, doc_chunks)
        assert not is_val_wref, f"Failed: Explanation with mismatched option reference should be rejected! Reason: {r_wref}"

        # Fallacy 5: "implements keyword defines an interface"
        cand_fallacy_impl = MCQCandidate(
            question="Which keyword is used to implement an interface in Java?",
            options={"A": "implements", "B": "extends", "C": "interface", "D": "abstract"},
            correct="A",
            explanation="The implements keyword is used to define an interface and implement it in a class."
        )
        is_val_f_impl, r_f_impl = validate_explanation_accuracy(cand_fallacy_impl)
        assert not is_val_f_impl, f"Failed: Fallacy 'defines an interface' for implements should be rejected! Reason: {r_f_impl}"

        # Fallacy 6: Oversimplified class vs interface definition
        cand_fallacy_cls = MCQCandidate(
            question="What is the difference between a class and an interface in Java?",
            options={
                "A": "A class provides concrete implementations while an interface specifies abstract methods.",
                "B": "A class cannot have methods.",
                "C": "An interface can have instance variables.",
                "D": "There is no difference."
            },
            correct="A",
            explanation="A class is a blueprint for creating objects, while an interface is a blueprint for defining methods."
        )
        is_val_f_cls, r_f_cls = validate_explanation_accuracy(cand_fallacy_cls)
        assert not is_val_f_cls, f"Failed: Oversimplified class vs interface definition should be rejected! Reason: {r_f_cls}"

        # Test improvement for implements question
        cand_impl_improve = MCQCandidate(
            question="Which keyword is used by a Java class to implement an interface?",
            options={"A": "implements", "B": "extends", "C": "interface", "D": "abstract"},
            correct="A",
            explanation="The implements keyword is used to implement an interface."
        )
        improve_weak_explanation(cand_impl_improve, doc_chunks)
        assert "class uses the 'implements' keyword to declare that it implements an interface" in cand_impl_improve.explanation, f"Expected improved implements explanation, got: {cand_impl_improve.explanation}"
        assert "define an interface" not in cand_impl_improve.explanation

        # Test improvement for class vs interface question
        cand_cls_improve = MCQCandidate(
            question="What is the difference between a class and an interface in Java?",
            options={
                "A": "A class provides concrete implementations while an interface specifies abstract methods.",
                "B": "A class cannot have methods.",
                "C": "An interface can have instance variables.",
                "D": "There is no difference."
            },
            correct="A",
            explanation="A class is a blueprint for creating objects, while an interface is a blueprint for defining methods."
        )
        improve_weak_explanation(cand_cls_improve, doc_chunks)
        assert "Within the scope of the document" in cand_cls_improve.explanation, f"Expected document scope in explanation, got: {cand_cls_improve.explanation}"
        assert "contract" in cand_cls_improve.explanation.lower() or "abstract" in cand_cls_improve.explanation.lower()

        # Verify improved candidates pass validation
        is_val_imp1, _ = validate_explanation_accuracy(cand_impl_improve)
        assert is_val_imp1, f"Improved implements explanation should pass validation: {cand_impl_improve.explanation}"
        is_val_imp2, _ = validate_explanation_accuracy(cand_cls_improve)
        assert is_val_imp2, f"Improved class vs interface explanation should pass validation: {cand_cls_improve.explanation}"

        print("  [PASS] All explanation fallacies (no return type, mutable string, wrong option reference, implements defines interface, oversimplified class vs interface) correctly rejected and improved.")
        results["Test 16: Explanation accuracy and fallacy rejection"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 16: Explanation accuracy and fallacy rejection"] = f"FAIL: {e}"

    # -------------------------------------------------------------
    # TEST 17: Inheritance Hierarchy Grounding and Contradiction Rejection
    # -------------------------------------------------------------
    print("\n--- TEST 17: Inheritance Hierarchy Grounding & Contradiction Rejection ---")
    try:
        doc_chunks = [
            "Module 2 Q19. Vehicle → MotorVehicle → Car (multilevel)\n"
            "class Vehicle { String make,model; int year; Vehicle(String m,String mo,int y){ make=m; model=mo; year=y; } }\n"
            "class MotorVehicle extends Vehicle { double cc; String fuel; MotorVehicle(String m,String mo,int y,double cc,String f){ super(m,mo,y); this.cc=cc; fuel=f; } }\n"
            "class Car extends MotorVehicle { int doors; Car(String m,String mo,int y,double cc,String f,int d){ super(m,mo,y,cc,f); doors=d; } }"
        ]

        # Case 1: Contradictory candidate claiming Vehicle is an interface implemented by MotorVehicle
        cand_contra_interface = MCQCandidate(
            question="In the multilevel inheritance hierarchy presented in the document, what is the relationship between the `Vehicle`, `MotorVehicle`, and `Car` classes?",
            options={
                "A": "Vehicle is an interface implemented by MotorVehicle, and Car extends MotorVehicle.",
                "B": "Vehicle extends MotorVehicle, and MotorVehicle extends Car.",
                "C": "Car extends MotorVehicle, and MotorVehicle extends Vehicle.",
                "D": "All three classes are independent with no inheritance."
            },
            correct="A",
            explanation="Vehicle is an interface implemented by MotorVehicle.",
            source_quote="class MotorVehicle extends Vehicle"
        )
        is_val_inh1, r_inh1 = verify_inheritance_hierarchy_accuracy(cand_contra_interface, doc_chunks)
        assert not is_val_inh1, f"Failed: Option claiming Vehicle is an interface implemented by MotorVehicle must be rejected! Reason: {r_inh1}"
        is_val_full1, r_full1 = validate_mcq_candidate(cand_contra_interface, doc_chunks)
        assert not is_val_full1, f"Failed: validate_mcq_candidate should reject candidate claiming Vehicle is an interface! Reason: {r_full1}"

        # Case 2: Candidate with inverted hierarchy (Vehicle is a subclass of MotorVehicle)
        cand_inverted = MCQCandidate(
            question="In the document's multilevel inheritance hierarchy, what is the relationship between `Vehicle`, `MotorVehicle`, and `Car`?",
            options={
                "A": "Vehicle is a subclass of MotorVehicle, which extends Car.",
                "B": "Car extends MotorVehicle, and MotorVehicle extends Vehicle.",
                "C": "Car and Vehicle both extend MotorVehicle.",
                "D": "MotorVehicle is the top-level class with no subclasses."
            },
            correct="A",
            explanation="Vehicle is a subclass.",
            source_quote="class MotorVehicle extends Vehicle"
        )
        is_val_inh2, r_inh2 = verify_inheritance_hierarchy_accuracy(cand_inverted, doc_chunks)
        assert not is_val_inh2, f"Failed: Option with inverted hierarchy must be rejected! Reason: {r_inh2}"

        # Case 3: Explanation fallacy claiming MotorVehicle implements Vehicle
        cand_fallacy_exp = MCQCandidate(
            question="What is the inheritance structure between `Vehicle`, `MotorVehicle`, and `Car`?",
            options={
                "A": "Car extends MotorVehicle, and MotorVehicle extends Vehicle.",
                "B": "Car extends Vehicle directly.",
                "C": "MotorVehicle extends Car.",
                "D": "None of the above."
            },
            correct="A",
            explanation="In the document, MotorVehicle implements Vehicle, while Car extends MotorVehicle."
        )
        is_val_exp, r_exp = validate_explanation_accuracy(cand_fallacy_exp)
        assert not is_val_exp, f"Failed: Explanation claiming MotorVehicle implements Vehicle must be rejected! Reason: {r_exp}"

        # Case 4: Correct source-supported candidate (Car extends MotorVehicle, MotorVehicle extends Vehicle)
        cand_valid = MCQCandidate(
            question="In the document's multilevel inheritance hierarchy, what is the relationship between the `Vehicle`, `MotorVehicle`, and `Car` classes?",
            options={
                "A": "Car extends MotorVehicle, and MotorVehicle extends Vehicle.",
                "B": "Vehicle extends MotorVehicle, and MotorVehicle extends Car.",
                "C": "Car and MotorVehicle both directly extend Vehicle independently.",
                "D": "Vehicle is an interface implemented by MotorVehicle and Car."
            },
            correct="A",
            explanation="Vehicle is the base class, MotorVehicle extends Vehicle, and Car extends MotorVehicle.",
            source_quote="class MotorVehicle extends Vehicle ... class Car extends MotorVehicle"
        )
        is_val_inh4, r_inh4 = verify_inheritance_hierarchy_accuracy(cand_valid, doc_chunks)
        assert is_val_inh4, f"Failed: Valid inheritance candidate should be accepted! Reason: {r_inh4}"
        is_val_full4, r_full4 = validate_mcq_candidate(cand_valid, doc_chunks)
        assert is_val_full4, f"Failed: Valid candidate should pass full validate_mcq_candidate! Reason: {r_full4}"

        # Case 5: Explanation improvement for multilevel inheritance
        cand_weak_exp = MCQCandidate(
            question="What is the multilevel inheritance relationship between Vehicle, MotorVehicle, and Car?",
            options={
                "A": "Car extends MotorVehicle, and MotorVehicle extends Vehicle.",
                "B": "Vehicle extends MotorVehicle.",
                "C": "Car extends Vehicle directly.",
                "D": "MotorVehicle is an interface."
            },
            correct="A",
            explanation="Car extends MotorVehicle, and MotorVehicle extends Vehicle.",
            source_quote="class MotorVehicle extends Vehicle"
        )
        improve_weak_explanation(cand_weak_exp, doc_chunks)
        assert "Vehicle is the base class, MotorVehicle extends Vehicle, and Car extends MotorVehicle" in cand_weak_exp.explanation, f"Expected improved explanation, got: {cand_weak_exp.explanation}"
        is_val_improved, r_improved = validate_mcq_candidate(cand_weak_exp, doc_chunks)
        assert is_val_improved, f"Candidate with improved explanation should pass validation! Reason: {r_improved}"

        print("  [PASS] Inheritance hierarchy contradictions (Vehicle as interface, inverted hierarchy, MotorVehicle implements Vehicle) correctly rejected, and verified hierarchy accepted with accurate explanation.")
        results["Test 17: Inheritance hierarchy grounding and contradiction rejection"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["Test 17: Inheritance hierarchy grounding and contradiction rejection"] = f"FAIL: {e}"

    print("\n" + "="*50)
    print("FINAL REGRESSION AUDIT RESULTS:")
    print("="*50)
    for test_name, status in results.items():
        print(f"{test_name:60} : {status}")
    print("="*50)

if __name__ == "__main__":
    run_audit()
