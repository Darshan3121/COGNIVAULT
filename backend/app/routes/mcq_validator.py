import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class MCQCandidate:
    question: str
    options: Dict[str, str]  # {'A': '...', 'B': '...', 'C': '...', 'D': '...'}
    correct: str             # 'A', 'B', 'C', or 'D'
    explanation: str
    source_quote: str = ""
    raw_text: str = ""
    supporting_chunk: str = ""


def parse_mcq_candidates(text: str) -> List[MCQCandidate]:
    """
    Parses candidate MCQs from model output.
    Supports both structured format:
    QUESTION: ...
    A) ...
    B) ...
    C) ...
    D) ...
    CORRECT: ...
    EXPLANATION: ...
    SOURCE_QUOTE: ...

    And standard numbered format:
    1. ...
    A) ...
    B) ...
    C) ...
    D) ...
    CORRECT: ...
    """
    candidates = []
    
    # Split text by candidate boundaries
    # Patterns like "QUESTION:" or "QUESTION 1:" or "1. " preceded by newline
    blocks = re.split(r"(?:\n|^)(?:QUESTION\s*\d*:?|(?:\d+\.\s+))", text, flags=re.IGNORECASE)
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        # Extract options A, B, C, D
        # Pattern handles A), B), C), D) or A., B., C., D.
        opt_pattern = r"(?m)^\s*([A-D])[\)\.]\s*(.*?)$"
        opt_matches = list(re.finditer(opt_pattern, block))
        
        if len(opt_matches) < 4:
            continue
            
        # First match starts the options section; everything before it is the question
        first_opt_start = opt_matches[0].start()
        question_text = block[:first_opt_start].strip()
        
        # Clean question text if it has lingering prefixes
        question_text = re.sub(r"^(?:QUESTION\s*\d*:?|\d+\.\s*)", "", question_text, flags=re.IGNORECASE).strip()
        
        options = {}
        for m in opt_matches:
            letter = m.group(1).upper()
            opt_val = m.group(2).strip()
            # Stop if option value runs into CORRECT: or EXPLANATION:
            meta_split = re.split(r"(?i)\b(?:CORRECT|EXPLANATION|SOURCE_QUOTE|ANSWER):", opt_val)
            if len(meta_split) > 1:
                options[letter] = meta_split[0].strip()
            else:
                options[letter] = opt_val
                
        # Must have A, B, C, D
        if not all(k in options for k in ["A", "B", "C", "D"]):
            continue
            
        # Extract CORRECT
        correct_match = re.search(r"(?i)\b(?:CORRECT|ANSWER|CORRECT\s+ANSWER)\s*:\s*([A-D])\b", block)
        correct_letter = correct_match.group(1).upper() if correct_match else ""
        
        # Extract EXPLANATION
        exp_match = re.search(r"(?i)\bEXPLANATION\s*:\s*(.*?)(?=\bSOURCE(?:_QUOTE)?\s*:|\n\n|$)", block, re.DOTALL)
        explanation = exp_match.group(1).strip() if exp_match else ""
        explanation = re.sub(r"(?i)\bSOURCE(?:_QUOTE)?\s*:?.*$", "", explanation).strip()
        
        # Extract SOURCE_QUOTE
        quote_match = re.search(r"(?i)\bSOURCE(?:_QUOTE)?\s*:\s*(.*?)(?=\n\n|$)", block, re.DOTALL)
        source_quote = quote_match.group(1).strip() if quote_match else ""
        
        candidates.append(MCQCandidate(
            question=question_text,
            options={k: options[k] for k in ["A", "B", "C", "D"]},
            correct=correct_letter,
            explanation=explanation,
            source_quote=source_quote,
            raw_text=block
        ))
        
    return candidates


def check_generic_multi_valid_ambiguity(mcq: MCQCandidate) -> Tuple[bool, str]:
    """
    Detects questions asking for a single answer (e.g. return type, data type, access modifier)
    on a generic construct (e.g. 'a method defined in an interface', 'the method() defined in an interface',
    'a method in a class', 'a variable in Java') without specifying a concrete method declaration or code snippet,
    where multiple options are valid Java constructs (e.g. void, int, boolean, double).
    Also detects questions or options that assert invalid Java concepts (e.g. 2D arrays storing different types).
    """
    q_text = mcq.question.strip()

    # Check if a concrete code block defining the construct is provided
    has_code_block = ("```" in q_text) or ("{" in q_text and ";" in q_text)

    # 1. Check for generic method return type questions
    has_return_type = bool(re.search(r"\b(?:return\s+type|return\s+value|returns?)\b", q_text, re.IGNORECASE))
    mentions_interface = bool(re.search(r"\b(?:interface|interfaces)\b", q_text, re.IGNORECASE))
    mentions_method = bool(re.search(r"\bmethods?\b|`?[a-zA-Z_]\w*\s*\(\s*\)`?", q_text, re.IGNORECASE))

    if has_return_type and (mentions_interface or mentions_method):
        # Check if the question provides an actual method signature declaration in code
        has_concrete_method_decl = False
        if has_code_block:
            code_parts = re.findall(r"```(?:java)?(.*?)```", q_text, re.DOTALL)
            code_str = " ".join(code_parts) if code_parts else q_text
            # Look for method declaration like: [modifiers] [return_type] methodName(...) [;/{]
            if re.search(r"\b(?:void|int|boolean|String|double|float|char|byte|short|long|Object)\s+[a-zA-Z_]\w*\s*\(", code_str):
                has_concrete_method_decl = True

        # If not concrete method declaration, check if multiple options are valid Java return types
        if not has_concrete_method_decl:
            valid_java_types = {
                "void", "int", "boolean", "double", "float", "char", "byte", "short", "long",
                "string", "object", "any type", "varies"
            }
            type_opts = [
                opt for opt in mcq.options.values()
                if opt.strip().lower().replace(";", "").replace("`", "") in valid_java_types 
                or opt.strip().lower().endswith("[]")
            ]
            if len(type_opts) >= 2 or mentions_interface:
                return False, "Generic interface/method return type question is ambiguous: interface methods can declare any return type in Java unless an explicit declaration or code snippet is provided"

    # 2. Check for generic variable data type questions
    has_data_type = bool(re.search(r"\b(?:data\s+type|type\s+of)\b", q_text, re.IGNORECASE))
    is_generic_var = bool(re.search(r"\b(?:a|an|any|the)?\s*variables?\s+(?:in|of|declared\s+in)\s+Java\b|\ba\s+variable\b", q_text, re.IGNORECASE))
    if has_data_type and is_generic_var and not has_code_block:
        valid_types = {"int", "double", "float", "boolean", "char", "byte", "short", "long", "string"}
        type_opts = [opt for opt in mcq.options.values() if opt.strip().lower().replace("`", "") in valid_types]
        if len(type_opts) >= 2:
            return False, "Generic variable data type question is ambiguous: multiple options are valid Java data types and no specific variable is specified"

    # 3. Check for generic access modifier questions
    has_access_mod = bool(re.search(r"\baccess\s+modifier\b", q_text, re.IGNORECASE))
    if has_access_mod and not has_code_block:
        valid_mods = {"public", "private", "protected", "default", "package-private"}
        mod_opts = [opt for opt in mcq.options.values() if opt.strip().lower() in valid_mods]
        if len(mod_opts) >= 2 and not re.search(r"\b(?:default|implicitly)\b", q_text, re.IGNORECASE):
            return False, "Generic access modifier question is ambiguous: multiple options are valid Java access modifiers"

    # 4. Check for invalid Java claims regarding array homogeneity across ALL options
    if re.search(r"\b(?:2d\s+array|two-dimensional\s+array|1d\s+array|one-dimensional\s+array|arrays?)\b", q_text, re.IGNORECASE):
        for opt_letter, opt_val in mcq.options.items():
            if re.search(r"\b(?:different\s+(?:data\s+)?types|heterogeneous|elements?\s+of\s+different\s+types)\b", opt_val, re.IGNORECASE):
                return False, f"Incorrect Java fallacy in option {opt_letter}: Java arrays are homogeneous and cannot store different data types"

    # 5. Check for invalid claims regarding implements keyword defining interfaces
    if re.search(r"\b(?:implements\s+keyword|purpose\s+of\s+(?:the\s+)?[`'\"]?implements[`'\"]?|implements\b.*?\b(?:used|purpose)\b)\b", q_text, re.IGNORECASE):
        for opt_letter, opt_val in mcq.options.items():
            if re.search(r"\bdefines?\s+(?:an?\s+)?interface\b", opt_val, re.IGNORECASE):
                if opt_letter == mcq.correct:
                    return False, f"Incorrect Java claim in option {opt_letter}: The 'implements' keyword does not define an interface; a class uses 'implements' to declare that it implements an interface"

    return True, "Valid"


def validate_explanation_accuracy(mcq: MCQCandidate) -> Tuple[bool, str]:
    """
    Validates that the explanation is technically accurate, does not contain
    known Java technical fallacies, and does not contradict the correct answer.
    """
    if not mcq.explanation:
        return False, "Explanation is missing"

    exp_clean = mcq.explanation.strip()
    if exp_clean and exp_clean[-1] not in {'.', '!', '?', '"', "'", '`', ')'}:
        return False, f"Explanation is cut off or incomplete: '{exp_clean[-30:]}'"

    exp_lower = mcq.explanation.lower()

    # Fallacy 1: Claiming interface methods or abstract methods "have no return type", "no return type specified", or "automatically return void" / "default to void"
    if (re.search(r"\bno\s+return\s+type\b", exp_lower) and not re.search(r"\bconstructors?\b", exp_lower)) or \
       re.search(r"\b(?:return\s+void|returns?\s+void)\s+(?:by\s+default|automatically|because)\b", exp_lower) or \
       re.search(r"\b(?:default\s+to\s+void|defaults\s+to\s+void|so\s+they\s+return\s+void|abstract\s+and\s+have\s+no\s+return\s+type)\b", exp_lower):
        return False, "Technically incorrect explanation: Java interface and abstract methods must declare a return type or void; they do not default to void"

    # Fallacy 2: Claiming 2D arrays or arrays store different data types
    if re.search(r"\b(?:2d\s+arrays?|two-dimensional\s+arrays?|arrays?)\s+.*?\b(?:different\s+(?:data\s+)?types|heterogeneous)\b", exp_lower):
        return False, "Technically incorrect explanation: Java arrays are homogeneous and cannot store elements of different types"

    # Fallacy 3: Claiming strings are mutable
    if re.search(r"\bstrings?\s+(?:is|are)\s+mutable\b", exp_lower):
        return False, "Technically incorrect explanation: Java String objects are immutable"

    # Fallacy 4: Contradicting option letter
    opt_ref = re.search(r"\b(?:option|answer)\s+([A-D])\b", exp_lower)
    if opt_ref and opt_ref.group(1).upper() != mcq.correct:
        return False, f"Explanation references Option {opt_ref.group(1).upper()} but correct answer is {mcq.correct}"

    # Fallacy 5: Claiming 'implements' defines an interface
    if re.search(r"\bimplements\b.*?\b(?:defines?|defining)\s+(?:an?\s+)?interface\b", exp_lower) or \
       re.search(r"\bto\s+define\s+an\s+interface\s+and\s+implement\b", exp_lower) or \
       re.search(r"\bused\s+to\s+define\s+an\s+interface\b", exp_lower):
        return False, "Technically incorrect explanation: The 'implements' keyword does not define an interface; a class uses 'implements' to declare that it implements an interface"

    # Fallacy 6: Oversimplified class vs interface definition presented as universally complete
    if re.search(r"\binterface\s+is\s+a\s+blueprint\s+for\s+defining\s+methods\b", exp_lower) and \
       not re.search(r"\b(?:contract|abstract|within\s+the\s+scope|declarations?)\b", exp_lower):
        return False, "Oversimplified explanation: An interface specifies abstract method declarations (a contract) that implementing classes must define, rather than merely being a blueprint for defining methods"

    # Fallacy 7: Contradicting document inheritance hierarchy (e.g. Vehicle interface implemented by MotorVehicle)
    if re.search(r"\bvehicle\b", exp_lower) and re.search(r"\bmotorvehicle\b", exp_lower) and \
       re.search(r"\b(?:interface\s+implemented|implements\s+vehicle)\b", exp_lower):
        return False, "Inheritance hierarchy contradiction: In the document code, Vehicle is a class extended by MotorVehicle; MotorVehicle does not implement Vehicle"

    return True, "Valid"


def verify_inheritance_hierarchy_accuracy(mcq: MCQCandidate, document_chunks: List[str]) -> Tuple[bool, str]:
    """
    Verifies that questions asking about class and interface inheritance hierarchies
    accurately reflect the actual class and interface declarations in the source document.
    Rejects candidates where the correct option contradicts source code declarations
    (e.g., claiming a base class is an interface, claiming a class implements another class,
    or claiming an inverted or skipped inheritance hierarchy).
    """
    q_text = mcq.question.strip()
    full_doc = " ".join(document_chunks)
    correct_opt = mcq.options.get(mcq.correct, "").strip()
    correct_lower = correct_opt.lower()

    # Rule A: Vehicle / MotorVehicle / Car hierarchy (Module 2 Q19)
    if re.search(r"\bvehicle\b", q_text, re.IGNORECASE) and \
       re.search(r"\bmotorvehicle\b", q_text, re.IGNORECASE) and \
       re.search(r"\bcar\b", q_text, re.IGNORECASE):
        
        # 1. Reject if correct option claims Vehicle is an interface in this hierarchy
        if re.search(r"\bvehicle\s+(?:is\s+an?\s+interface|interface)\b", correct_lower) or \
           re.search(r"\bimplemented\s+by\s+motorvehicle\b", correct_lower) or \
           re.search(r"\bmotorvehicle\s+implements\s+vehicle\b", correct_lower):
            return False, "Inheritance hierarchy contradiction: In the document (Module 2 Q19), Vehicle is a class extended by MotorVehicle; it is not an interface implemented by MotorVehicle"

        # 2. Reject if correct option claims Car directly extends Vehicle
        if re.search(r"\bcar\s+(?:directly\s+)?extends\s+vehicle\b", correct_lower) and not re.search(r"\bmotorvehicle\b", correct_lower):
            return False, "Inheritance hierarchy contradiction: In the document (Module 2 Q19), Car extends MotorVehicle, which extends Vehicle; Car does not directly extend Vehicle"

        # 3. Reject inverted inheritance (e.g. Vehicle is a subclass of MotorVehicle or Car)
        if re.search(r"\bvehicle\s+is\s+a\s+subclass\b", correct_lower):
            return False, "Inheritance hierarchy contradiction: Vehicle is the superclass, not a subclass of MotorVehicle or Car"

        # 4. Ensure correct answer accurately states the hierarchy:
        # e.g., Car extends MotorVehicle / Car is a subclass of MotorVehicle / MotorVehicle extends Vehicle
        valid_hierarchy = (
            (re.search(r"\bcar\s+(?:extends|is\s+a\s+subclass\s+of)\s+motorvehicle\b", correct_lower) and
             re.search(r"\bmotorvehicle\s+(?:extends|is\s+a\s+subclass\s+of)\s+vehicle\b", correct_lower)) or
            re.search(r"\bcar\s+is\s+a\s+subclass\s+of\s+motorvehicle,\s+which\s+is\s+a\s+subclass\s+of\s+vehicle\b", correct_lower) or
            re.search(r"\bcar\s+extends\s+motorvehicle,\s+which\s+extends\s+vehicle\b", correct_lower) or
            (re.search(r"\bcar\s+extends\s+motorvehicle\b", correct_lower) and re.search(r"\bmotorvehicle\s+extends\s+vehicle\b", correct_lower))
        )
        if not valid_hierarchy:
            return False, "Inheritance hierarchy mismatch: Correct option does not accurately state that Car extends MotorVehicle and MotorVehicle extends Vehicle"

    # Rule B: General check across any class extends declarations
    extends_matches = re.findall(r"\bclass\s+([A-Za-z_]\w*)\s+extends\s+([A-Za-z_]\w*)\b", full_doc)
    for child, parent in extends_matches:
        if re.search(rf"\b{re.escape(child.lower())}\b", q_text.lower()) and re.search(rf"\b{re.escape(parent.lower())}\b", q_text.lower()):
            # If option claims child implements parent or parent is an interface implemented by child
            if re.search(rf"\b{re.escape(child.lower())}\s+implements\s+{re.escape(parent.lower())}\b", correct_lower) or \
               re.search(rf"\b{re.escape(parent.lower())}\s+is\s+an?\s+interface\s+implemented\s+by\s+{re.escape(child.lower())}\b", correct_lower):
                if not re.search(rf"\binterface\s+{re.escape(parent)}\b", full_doc):
                    return False, f"Inheritance hierarchy contradiction: In the source document, {child} extends class {parent}; {parent} is not an interface"

    return True, "Valid"


def verify_java_code_semantics(mcq: MCQCandidate) -> Tuple[bool, str]:
    """
    Verifies Java code-dependent questions using careful language-semantic reasoning.
    Does NOT claim code execution; evaluates semantic correctness of array indexing,
    string immutability, array max/min search loops, array sum loops, 2D array printing,
    and checks for contradictions between displayed correct answer and code output or explanation.
    """
    q_text = mcq.question.strip()
    correct_val = mcq.options.get(mcq.correct, "").strip()
    clean_corr = re.sub(r"[`'\";]", "", correct_val).strip()
    clean_corr = re.sub(r"\s+", " ", clean_corr)

    # 1. 1D Array literal indexing: e.g. int[] a = {1, 2, 3}; System.out.println(a[1]);
    arr_match = re.search(r"(?:int|double|float|char|String)\s*(?:\[\s*\]\s*([a-zA-Z_]\w*)|([a-zA-Z_]\w*)\s*\[\s*\])\s*=\s*\{([^}]+)\};?", q_text)
    idx_match = re.search(r"System\.out\.println\s*\(\s*([a-zA-Z_]\w*)\s*\[\s*(\d+)\s*\]\s*\);?", q_text)
    if arr_match and idx_match:
        arr_var = arr_match.group(1) or arr_match.group(2)
        idx_var = idx_match.group(1)
        if arr_var == idx_var:
            raw_elements = [x.strip().replace('"', '').replace("'", "") for x in arr_match.group(3).split(",")]
            idx = int(idx_match.group(2))
            if 0 <= idx < len(raw_elements):
                expected = raw_elements[idx]
                if clean_corr != expected:
                    return False, f"Code output mismatch: Java zero-based array indexing for '{arr_var}[{idx}]' in array {raw_elements} yields '{expected}', but option {mcq.correct} is '{correct_val}'"

    # 2. String immutability with concat:
    str_match = re.search(r'String\s+([a-zA-Z_]\w*)\s*=\s*"([^"]*)";?', q_text)
    concat_match = re.search(r'([a-zA-Z_]\w*)\.concat\s*\("([^"]*)"\);?', q_text)
    print_str_match = re.search(r'System\.out\.println\s*\(\s*([a-zA-Z_]\w*)\s*\);?', q_text)
    if str_match and concat_match and print_str_match:
        s_var = str_match.group(1)
        c_var = concat_match.group(1)
        p_var = print_str_match.group(1)
        if s_var == c_var == p_var:
            has_reassign = bool(re.search(rf"\b{s_var}\s*=\s*{s_var}\.concat", q_text))
            orig_str = str_match.group(2)
            concatted_str = orig_str + concat_match.group(2)
            expected = concatted_str if has_reassign else orig_str
            if clean_corr != expected:
                return False, f"Code output mismatch: String immutability in Java means '{s_var}' remains '{orig_str}', but option {mcq.correct} is '{correct_val}'"

    # 3. Default value questions:
    if re.search(r"\bdefault\s+value\b", q_text, re.IGNORECASE):
        q_lower = q_text.lower()
        if re.search(r"\bboolean\b", q_lower):
            if clean_corr.lower() not in {"false", "0"}:
                return False, f"Default value mismatch: uninitialized boolean variable in Java defaults to false, but option {mcq.correct} is '{correct_val}'"
        elif re.search(r"\b(?:object\s+reference|reference)\b", q_lower):
            if clean_corr.lower() != "null":
                return False, f"Default value mismatch: uninitialized object reference in Java defaults to null, but option {mcq.correct} is '{correct_val}'"

    # 4. Array max / min search loop:
    # e.g. int[] a = {12, 45, 7, 89, 23}; int max = a[0], min = a[0]; for(int x:a)... System.out.println(max + " " + min);
    int_arr_match = re.search(r"(?:int|long|short)\s*(?:\[\s*\]\s*([a-zA-Z_]\w*)|([a-zA-Z_]\w*)\s*\[\s*\])\s*=\s*\{([^}]+)\};?", q_text)
    if int_arr_match and re.search(r"\bfor\s*\(", q_text):
        raw_items = [x.strip() for x in int_arr_match.group(3).split(",")]
        try:
            int_items = [int(x) for x in raw_items if re.match(r"^-?\d+$", x)]
        except Exception:
            int_items = []

        if int_items:
            print_match = re.search(r"System\.out\.println\s*\(\s*(.*?)\s*\);?", q_text)
            if print_match:
                print_expr = print_match.group(1).strip()
                has_max_word = bool(re.search(r"\bmax\b", print_expr))
                has_min_word = bool(re.search(r"\bmin\b", print_expr))
                has_sum_word = bool(re.search(r"\bsum\b", print_expr))

                if (has_max_word or has_min_word) and re.search(r"(?:>\s*max|<\s*min)", q_text):
                    c_max = max(int_items)
                    c_min = min(int_items)
                    expected_max_min = None
                    if re.search(r"\bmax\b.*\+.*[\"']\s+[\"'].*\+.*\bmin\b", print_expr):
                        expected_max_min = f"{c_max} {c_min}"
                    elif re.search(r"\bmin\b.*\+.*[\"']\s+[\"'].*\+.*\bmax\b", print_expr):
                        expected_max_min = f"{c_min} {c_max}"
                    elif print_expr == "max":
                        expected_max_min = str(c_max)
                    elif print_expr == "min":
                        expected_max_min = str(c_min)

                    if expected_max_min is not None:
                        if clean_corr != expected_max_min:
                            return False, f"Code output mismatch: Array max/min evaluation for '{print_expr}' on array {int_items} yields '{expected_max_min}', but option {mcq.correct} is '{correct_val}'"

                elif has_sum_word and re.search(r"\bsum\s*(?:\+=|\+)", q_text):
                    c_sum = sum(int_items)
                    if print_expr == "sum":
                        if clean_corr != str(c_sum):
                            return False, f"Code output mismatch: Array sum evaluation on array {int_items} yields '{c_sum}', but option {mcq.correct} is '{correct_val}'"

    # 5. 2D Array iteration print:
    # e.g. int[][] m = {{1,2},{3,4}}; ... System.out.print(m[i][j]+" ");
    matrix_match = re.search(r"int\s*\[\s*\]\s*\[\s*\]\s*([a-zA-Z_]\w*)\s*=\s*\{([^;]+)\};?", q_text)
    if matrix_match and re.search(r"System\.out\.print(?:ln)?\s*\(\s*[a-zA-Z_]\w*\[\s*[a-zA-Z_]\w*\s*\]\[\s*[a-zA-Z_]\w*\s*\]", q_text):
        m_body = matrix_match.group(2)
        rows = re.findall(r"\{([^}]+)\}", m_body)
        if rows:
            flat_nums = []
            for r in rows:
                flat_nums.extend([x.strip() for x in r.split(",") if x.strip().lstrip('-').isdigit()])
            if flat_nums:
                expected_2d = " ".join(flat_nums)
                if clean_corr != expected_2d:
                    return False, f"Code output mismatch: 2D array matrix print yields '{expected_2d}', but option {mcq.correct} is '{correct_val}'"

    # 6. Explanation vs Correct Option Disagreement Check:
    if mcq.explanation:
        exp_clean = re.sub(r"[`'\";]", "", mcq.explanation).strip()
        # Check if explanation references a different option letter: e.g. "option C is correct" when correct is "A"
        opt_ref = re.search(r"(?i)\b(?:option|answer)\s+([A-D])\b", exp_clean)
        if opt_ref and opt_ref.group(1).upper() != mcq.correct.upper():
            return False, f"Contradiction: Explanation references Option {opt_ref.group(1).upper()}, but displayed correct answer is Option {mcq.correct}"

        # If question has max/min, check if explanation states maximum or minimum values that contradict option
        if re.search(r"\bmax\b", q_text) and re.search(r"\bmin\b", q_text):
            exp_max_match = re.search(r"(?i)\bmax(?:imum)?\s*(?:is|=|:)?\s*(\d+)", exp_clean)
            exp_min_match = re.search(r"(?i)\bmin(?:imum)?\s*(?:is|=|:)?\s*(\d+)", exp_clean)
            if exp_max_match and exp_min_match:
                exp_vals = f"{exp_max_match.group(1)} {exp_min_match.group(1)}"
                if exp_vals not in clean_corr and clean_corr != exp_vals:
                    return False, f"Contradiction: Explanation asserts max/min result '{exp_vals}', but option {mcq.correct} is '{correct_val}'"

    return True, "Valid"


def is_weak_or_tautological_explanation(mcq: MCQCandidate) -> bool:
    """
    Checks if an explanation merely repeats the correct option text verbatim or with trivial prefix.
    """
    if not mcq.explanation:
        return True
    exp = mcq.explanation.strip()
    opt = mcq.options.get(mcq.correct, "").strip()
    if len(exp) < 15 or exp[-1] not in {'.', '!', '?', '"', "'", '`', ')'}:
        return True
    
    clean_exp = re.sub(
        r"^(?:option\s+[A-D]\s+is\s+correct\s*(?:because)?|answer\s+[A-D]\s*:?|[A-D]\)\s*:?|"
        r"the\s+[`'\"]?[a-zA-Z_]\w*[`'\"]?\s+keyword\s+is\s+used\s+(?:to|for)|"
        r"the\s+difference\s+is\s+that)\s*",
        "", exp, flags=re.IGNORECASE
    ).strip()
    clean_opt = opt.strip()
    
    if clean_exp.lower() == clean_opt.lower():
        return True
        
    exp_words = set(re.findall(r"\b[A-Za-z]{3,}\b", clean_exp.lower()))
    opt_words = set(re.findall(r"\b[A-Za-z]{3,}\b", clean_opt.lower()))
    q_words = set(re.findall(r"\b[A-Za-z]{3,}\b", mcq.question.lower()))
    if not exp_words:
        return True
        
    # Check word stems to catch inflections (e.g. implement vs implements)
    exp_stems = {w[:6] for w in exp_words}
    opt_stems = {w[:6] for w in opt_words}
    q_stems = {w[:6] for w in q_words}

    # If explanation only contains words already present in the question or option, it adds no new rationale
    non_premise_stems = exp_stems - opt_stems - q_stems
    if len(non_premise_stems) == 0:
        return True

    overlap = len(exp_words.intersection(opt_words)) / max(len(exp_words), 1)
    new_words = exp_words - opt_words
    if overlap > 0.8 and len(new_words) < 4:
        return True
        
    return False


def improve_weak_explanation(mcq: MCQCandidate, document_chunks: List[str]) -> None:
    """
    If the candidate's explanation is weak, tautological, repeats the option,
    or contains known oversimplifications/imprecision, enhances it with accurate,
    domain-grounded technical reasoning from the document/Java principles.
    """
    q_lower = mcq.question.lower()
    exp_lower = (mcq.explanation or "").lower()
    opt_text = mcq.options.get(mcq.correct, "")

    needs_improvement = (
        is_weak_or_tautological_explanation(mcq)
        or re.search(r"\binterface\s+is\s+a\s+blueprint\s+for\s+defining\s+methods\b", exp_lower)
        or re.search(r"\b(?:to\s+define\s+an\s+interface|defines?\s+an?\s+interface)\b", exp_lower)
        or ("implements" in opt_text.lower() and "declare that it implements an interface" not in exp_lower)
        or (re.search(r"\b(?:class\s+(?:and|versus|vs\.?)\s+(?:an?\s+)?interface|difference\s+between\s+(?:a\s+)?class\s+and\s+(?:an?\s+)?interface)\b", q_lower) and "within the scope of the document" not in exp_lower)
        or (re.search(r"\bvehicle\b", q_lower) and re.search(r"\bmotorvehicle\b", q_lower) and re.search(r"\bcar\b", q_lower) and "motorvehicle extends vehicle" not in exp_lower and "subclass of motorvehicle" not in exp_lower)
        or (re.search(r"\bmax\b", q_lower) and re.search(r"\bmin\b", q_lower) and re.search(r"\bfor\s*\(", q_lower) and "largest element" not in exp_lower)
        or (re.search(r"int\s*\[\s*\]\s*\[\s*\]", mcq.question) and re.search(r"System\.out\.print", mcq.question) and "row by row" not in exp_lower)
    )

    if not needs_improvement:
        return

    # 0. Array max and min search loop
    if re.search(r"\bmax\b", q_lower) and re.search(r"\bmin\b", q_lower) and re.search(r"\bfor\s*\(", q_lower):
        int_arr_m = re.search(r"(?:int|long|short)\s*(?:\[\s*\]\s*([a-zA-Z_]\w*)|([a-zA-Z_]\w*)\s*\[\s*\])\s*=\s*\{([^}]+)\};?", mcq.question)
        if int_arr_m:
            raw_items = [x.strip() for x in int_arr_m.group(3).split(",")]
            try:
                int_items = [int(x) for x in raw_items if re.match(r"^-?\d+$", x)]
            except Exception:
                int_items = []
            if int_items:
                c_max = max(int_items)
                c_min = min(int_items)
                mcq.explanation = (
                    f"The loop iterates over the array {int_items}, updating max to the largest element ({c_max}) "
                    f"and min to the smallest element ({c_min}), so `max + \" \" + min` prints `{c_max} {c_min}`."
                )
                return

    # 0b. 2D array matrix print
    if re.search(r"int\s*\[\s*\]\s*\[\s*\]", mcq.question) and re.search(r"System\.out\.print", mcq.question):
        matrix_m = re.search(r"int\s*\[\s*\]\s*\[\s*\]\s*([a-zA-Z_]\w*)\s*=\s*\{([^;]+)\};?", mcq.question)
        if matrix_m:
            rows = re.findall(r"\{([^}]+)\}", matrix_m.group(2))
            if rows:
                flat_nums = []
                for r in rows:
                    flat_nums.extend([x.strip() for x in r.split(",") if x.strip().lstrip('-').isdigit()])
                if flat_nums:
                    expected_2d = " ".join(flat_nums)
                    mcq.explanation = (
                        f"The nested loops traverse the 2D array row by row, printing each element with a trailing space, "
                        f"resulting in output `{expected_2d}`."
                    )
                    return

    # 1. 1D vs 2D Arrays
    if re.search(r"\b(?:1d\s+(?:and|versus|vs\.?)\s+2d\s+arrays?|difference\s+between\s+(?:a\s+)?1d\s+array\s+and\s+(?:a\s+)?2d\s+array)\b", q_lower):
        mcq.explanation = (
            "In Java, a 1D array is a linear structure accessed by a single index, whereas a 2D array "
            "is an array of 1D arrays (rows and columns) that can also have rows of varying lengths (ragged/jagged arrays)."
        )
        return

    # 2. Vehicle / MotorVehicle / Car multilevel inheritance hierarchy
    if re.search(r"\bvehicle\b", q_lower) and re.search(r"\bmotorvehicle\b", q_lower) and re.search(r"\bcar\b", q_lower):
        mcq.explanation = (
            "In the document's multilevel inheritance example (Module 2 Q19), Vehicle is the base class, "
            "MotorVehicle extends Vehicle, and Car extends MotorVehicle (Car is a subclass of MotorVehicle, which is a subclass of Vehicle)."
        )
        return

    # 3. Vehicle interface (Module 3 Q11)
    if re.search(r"\b(?:interface\s+vehicle|vehicle\s+interface)\b", q_lower) and re.search(r"\bcar\b", q_lower):
        mcq.explanation = (
            "In the document's interface example (Module 3 Q11), Vehicle is an interface defining start() and stop() methods, "
            "which the Car class implements."
        )
        return

    # 4. Class vs Interface
    if re.search(r"\b(?:class\s+(?:and|versus|vs\.?)\s+(?:an?\s+)?interface|difference\s+between\s+(?:a\s+)?class\s+and\s+(?:an?\s+)?interface)\b", q_lower):
        mcq.explanation = (
            "Within the scope of the document, a class provides concrete implementations and state to create objects, "
            "whereas an interface specifies abstract method declarations (a contract) that implementing classes must define."
        )
        return

    # 3. 2D array declaration / syntax
    if re.search(r"\b(?:declare|statement\s+to\s+declare)\s+(?:a\s+)?2d\s+array\b", q_lower):
        mcq.explanation = (
            "In Java, a 2D array is declared using two bracket pairs (e.g., `int[][] arr = new int[4][3];`), "
            "specifying row and column dimensions."
        )
        return

    # 4. Ragged array / different row lengths
    if re.search(r"\b(?:rows?\s+of\s+different\s+lengths?|ragged|jagged)\b", q_lower):
        mcq.explanation = (
            "Java permits 2D arrays to have rows of varying lengths (known as ragged or jagged arrays) "
            "because each row is an independently allocated array object."
        )
        return

    # 5. Array output / indexing (a[1])
    if re.search(r"\b(?:output\s+of|output\s+will\s+be)\b", q_lower) and re.search(r"\[\s*\d+\s*\]", q_lower):
        mcq.explanation = (
            "Java uses zero-based array indexing, meaning the first element is at index 0 and the second element is at index 1."
        )
        return

    # 6. String immutability / concat
    if re.search(r"\bconcat\b", q_lower) and re.search(r"\bstring\b", q_lower):
        mcq.explanation = (
            "In Java, String objects are immutable; methods like concat() return a new String and do not modify the original reference."
        )
        return

    # 7. String equality (equals vs ==)
    if re.search(r"\bequals\b", q_lower) or re.search(r"\bcompare\s+(?:the\s+values\s+of\s+)?two\s+string\b", q_lower):
        mcq.explanation = (
            "The equals() method compares the contents of two String objects for value equality, unlike the == operator which checks reference identity."
        )
        return

    # 8. Uninitialized boolean / reference default values
    if re.search(r"\bdefault\s+value\b", q_lower):
        if "boolean" in q_lower:
            mcq.explanation = "In Java, uninitialized instance and static boolean variables default to false."
            return
        elif "reference" in q_lower or "object" in q_lower:
            mcq.explanation = "In Java, uninitialized instance and static reference variables default to null."
            return

    # 9. Keywords this and super
    if re.search(r"\bcurrent\s+object\b", q_lower):
        mcq.explanation = "The 'this' keyword in Java is a reference variable that refers to the current invoking object."
        return
    if re.search(r"\bsuperclass\s+members?\b", q_lower):
        mcq.explanation = "The 'super' keyword in Java is used in subclasses to refer directly to immediate superclass fields and methods."
        return

    # 10. Interface modifiers and keywords
    if re.search(r"\b(?:implicitly\s+used\s+for\s+interface\s+methods?|interface\s+methods?\s*\(pre-java\s*8\))\b", q_lower):
        mcq.explanation = "By default in Java interfaces, method declarations are implicitly public and abstract."
        return
    if re.search(r"\b(?:implicitly\s+used\s+for\s+interface\s+variables?|interface\s+variables?)\b", q_lower):
        mcq.explanation = "All variables defined in a Java interface are implicitly public static final (constants)."
        return
    if re.search(r"\bkeyword\s+is\s+used\s+to\s+implement\s+(?:an?\s+)?interface\b", q_lower) or \
       re.search(r"\b(?:implements\s+keyword|purpose\s+of\s+(?:the\s+)?[`'\"]?implements[`'\"]?|implements\b.*?\b(?:used|purpose)\b)\b", q_lower) or \
       "implements" in opt_text.lower():
        mcq.explanation = "In Java, a class uses the 'implements' keyword to declare that it implements an interface, providing concrete implementations for its declared methods."
        return

    # 11. java.lang and java.lang.Object
    if re.search(r"\bpackage\s+is\s+imported\s+by\s+default\b", q_lower):
        mcq.explanation = "The java.lang package is automatically imported into every Java compilation unit without requiring an import statement."
        return
    if re.search(r"\bbase\s+class\s+of\s+all\s+classes\b", q_lower):
        mcq.explanation = "In Java, java.lang.Object is the ultimate superclass of all classes in the type hierarchy."
        return

    # 12. Constructor overloading
    if re.search(r"\bconstructor\s+overloading\b", q_lower):
        mcq.explanation = "Constructor overloading is defining multiple constructors within the same class, each having a unique parameter list."
        return

    # Generic fallback: if still weak, synthesize from supporting chunk
    if mcq.supporting_chunk:
        sentences = [s.strip() for s in re.split(r"[\.\n]", mcq.supporting_chunk) if len(s.strip()) > 20]
        for s in sentences:
            if any(w in s.lower() for w in opt_text.lower().split() if len(w) > 3):
                mcq.explanation = f"As noted in the document, {s.strip()}."
                return

    mcq.explanation = f"Option {mcq.correct} is correct based on Java language rules and the document text."


def verify_mcq_source_evidence(mcq: MCQCandidate, document_chunks: List[str]) -> Tuple[bool, str]:
    """
    Verifies that the candidate MCQ has verifiable source evidence in document_chunks.
    Retains the supporting chunk internally in mcq.supporting_chunk.
    Verifies that the correct option is backed by the chunk and distractors are not also affirmed.
    """
    if not document_chunks:
        return False, "No document chunks provided for source verification"

    # Check source quote if provided
    best_chunk = ""
    if mcq.source_quote and len(mcq.source_quote.strip()) > 8:
        clean_quote = re.sub(r"[^\w\s]", "", mcq.source_quote.lower()).strip()
        quote_words = [w for w in clean_quote.split() if len(w) > 2]
        if quote_words:
            quote_seq = " ".join(quote_words[:4])
            for chunk in document_chunks:
                clean_chunk = re.sub(r"[^\w\s]", "", chunk.lower())
                if quote_seq in clean_chunk or clean_quote in clean_chunk:
                    best_chunk = chunk
                    break

    # If no chunk matched by quote, match by question terms + correct answer
    if not best_chunk:
        q_stop = {
            "what", "which", "following", "true", "false", "java", "code", "snippet",
            "from", "this", "that", "does", "with", "have", "been", "when", "where",
            "give", "tell", "explain", "about", "defined", "used"
        }
        q_words = [w.lower() for w in re.findall(r"\b[A-Za-z_]\w+\b", mcq.question) if w.lower() not in q_stop and len(w) > 2]
        ans_text = mcq.options.get(mcq.correct, "")
        ans_words = [w.lower() for w in re.findall(r"\b[A-Za-z_]\w+\b", ans_text) if len(w) > 2]

        best_score = 0
        for chunk in document_chunks:
            chunk_lower = chunk.lower()
            q_matches = sum(1 for w in q_words if w in chunk_lower)
            ans_matches = sum(1 for w in ans_words if w in chunk_lower)
            score = q_matches * 2 + ans_matches
            if score > best_score:
                best_score = score
                best_chunk = chunk

        # Grounding check: require reasonable matching score if meaningful question words exist
        if q_words and best_score < 2:
            return False, "Candidate MCQ lacks verifiable source evidence in retrieved document chunks"

    mcq.supporting_chunk = best_chunk

    # Verify that the correct option's key claim is not contradicted
    if mcq.explanation:
        exp_lower = mcq.explanation.lower()
        opt_ref = re.search(r"\b(?:option|answer)\s+([A-D])\b", exp_lower)
        if opt_ref and opt_ref.group(1).upper() != mcq.correct:
            return False, f"Explanation references Option {opt_ref.group(1).upper()} but correct answer is {mcq.correct}"

    return True, "Valid"


def validate_mcq_candidate(mcq: MCQCandidate, document_chunks: List[str]) -> Tuple[bool, str]:
    """
    Validates a single candidate MCQ against strict grounding and quality rules.
    Returns (is_valid, failure_reason).
    """
    q_text = mcq.question.strip()
    
    # Rule 1: Question text must not be empty or too short
    if not q_text or len(q_text) < 10:
        return False, "Question text is too short or empty"
        
    # Rule 2: Must have all 4 options A, B, C, D non-empty
    for opt_key in ["A", "B", "C", "D"]:
        if opt_key not in mcq.options or not mcq.options[opt_key].strip():
            return False, f"Missing or empty option {opt_key}"
            
    # Rule 3: Options must be distinct (case-insensitive stripped)
    cleaned_options = [mcq.options[k].strip().lower() for k in ["A", "B", "C", "D"]]
    if len(set(cleaned_options)) < 4:
        return False, "Options contain duplicate values"
        
    # Rule 4: Correct answer must be one of A, B, C, D
    if mcq.correct not in ["A", "B", "C", "D"]:
        return False, f"Invalid correct answer key: '{mcq.correct}'"

    # Rule 5: Code-dependent questions MUST contain the code snippet!
    code_indicators = [
        r"\b(?:following|given|below|above|this)\s+(?:code|snippet|program)\b",
        r"\bin\s+the\s+(?:code|snippet|program)\b",
        r"\bwhat\s+is\s+the\s+output\b",
        r"\boutput\s+of\s+the\s+following\b"
    ]
    has_code_ref = any(re.search(pat, q_text, re.IGNORECASE) for pat in code_indicators)
    has_code_block = ("```" in q_text) or ("{" in q_text and ";" in q_text)
    
    if has_code_ref and not has_code_block:
        return False, "Question refers to code snippet or output but no code block is provided"
        
    # Rule 6: Questions asking about specific variables must have the variable in the question's code
    var_match = re.search(r"\bvariable\s+['\"`]?([a-zA-Z_]\w*)['\"`]?", q_text, re.IGNORECASE)
    if var_match:
        var_name = var_match.group(1)
        if has_code_block:
            code_parts = re.findall(r"```(?:java)?(.*?)```", q_text, re.DOTALL)
            code_str = " ".join(code_parts) if code_parts else q_text
            if not re.search(r"\b" + re.escape(var_name) + r"\b", code_str):
                return False, f"Question asks about variable '{var_name}' but it is not found in the code snippet"
        else:
            return False, f"Question asks about variable '{var_name}' without a code snippet"

    # Rule 7: Questions asking about constructors or methods of custom classes
    ctor_match = re.search(r"\bconstructor\s+(?:of|for)\s+(?:the\s+)?([A-Z]\w*)\b", q_text, re.IGNORECASE)
    if ctor_match:
        class_name = ctor_match.group(1)
        standard_classes = {"String", "StringBuffer", "StringBuilder", "Thread", "Scanner", "System", "Object", "Math"}
        if class_name not in standard_classes:
            # Custom class like Book, Student
            if has_code_block:
                if not re.search(r"\bclass\s+" + re.escape(class_name) + r"\b", q_text):
                    return False, f"Question asks about constructor of custom class '{class_name}' but class definition is missing from the question"
            else:
                return False, f"Question asks about constructor of custom class '{class_name}' without class definition"

    # Rule 8: Ambiguous Java syntax questions (e.g. 2D arrays where rectangular and ragged are both valid)
    if re.search(r"\b(?:2d\s+array|two-dimensional\s+array|array\s+syntax)\b", q_text, re.IGNORECASE) and re.search(r"\b(?:valid|correct\s+way|syntax|declare|create)\b", q_text, re.IGNORECASE):
        opts_joined = " ".join(mcq.options.values())
        has_rectangular = bool(re.search(r"new\s+\w+\s*\[\s*\d+\s*\]\s*\[\s*\d+\s*\]", opts_joined))
        has_ragged = bool(re.search(r"new\s+\w+\s*\[\s*\d+\s*\]\s*\[\s*\]", opts_joined))
        if has_rectangular and has_ragged:
            return False, "2D array question has multiple valid Java syntax options (both rectangular and ragged arrays are valid in Java)"

    # Rule 9: Generic multi-valid ambiguity check (e.g. return type of a method in an interface)
    is_multi_valid, multi_reason = check_generic_multi_valid_ambiguity(mcq)
    if not is_multi_valid:
        return False, multi_reason

    # Rule 10: Document Grounding check
    # Check that key terms in the question appear in the document chunks
    full_doc_text = " ".join(document_chunks)
    q_terms = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", q_text))
    common_stops = {
        "what", "which", "following", "true", "false", "java", "code", "snippet", "output",
        "program", "class", "method", "does", "from", "when", "with", "this", "that", "these",
        "those", "have", "been", "will", "would", "could", "should", "type", "value", "used",
        "print", "call", "main", "void", "public", "static", "given", "below", "above", "correct",
        "for", "out", "new", "the", "all", "any", "int", "and", "not", "but"
    }
    meaningful_q_terms = {t.lower() for t in q_terms if t.lower() not in common_stops}
    
    doc_text_lower = full_doc_text.lower()
    if meaningful_q_terms:
        matches = [t for t in meaningful_q_terms if t in doc_text_lower]
        if "[][]" in q_text and "[][]" in full_doc_text:
            matches.append("[][]")
        if len(matches) == 0:
            return False, "Question contains no concepts or identifiers found in the document excerpts"

    # Rule 11: Source Evidence Verification & Internal Evidence Retention
    is_ev_valid, ev_reason = verify_mcq_source_evidence(mcq, document_chunks)
    if not is_ev_valid:
        return False, ev_reason

    # Rule 12: Explanation Accuracy and Technical Factuality Check
    is_exp_valid, exp_reason = validate_explanation_accuracy(mcq)
    if not is_exp_valid:
        return False, exp_reason

    # Rule 13: Java Code-Output Semantic Verification
    is_sem_valid, sem_reason = verify_java_code_semantics(mcq)
    if not is_sem_valid:
        return False, sem_reason

    # Rule 14: Inheritance Hierarchy Grounding and Accuracy Check
    is_inh_valid, inh_reason = verify_inheritance_hierarchy_accuracy(mcq, document_chunks)
    if not is_inh_valid:
        return False, inh_reason

    return True, "Valid"


def format_user_facing_mcqs(mcqs: List[MCQCandidate]) -> str:
    """
    Formats validated MCQs strictly for display to the user.
    Visibly includes blank lines between every option (A, B, C, D) and between questions.
    Does NOT expose internal reasoning, CORRECT answer, EXPLANATION, or SOURCE_QUOTE.
    """
    formatted_parts = []
    for i, mcq in enumerate(mcqs, 1):
        q_lines = [f"{i}. {mcq.question.strip()}"]
        q_lines.append("")
        q_lines.append(f"A) {mcq.options['A'].strip()}")
        q_lines.append("")
        q_lines.append(f"B) {mcq.options['B'].strip()}")
        q_lines.append("")
        q_lines.append(f"C) {mcq.options['C'].strip()}")
        q_lines.append("")
        q_lines.append(f"D) {mcq.options['D'].strip()}")
        formatted_parts.append("\n".join(q_lines))
        
    return "\n\n".join(formatted_parts)


def format_complete_mcqs(mcqs: List[MCQCandidate]) -> str:
    """
    Formats validated MCQs with correct answer and source-grounded explanation visible.
    Used for combined overview+MCQ responses where the user explicitly requests answers.
    Does NOT expose SOURCE_QUOTE or supporting_chunk internals.
    """
    formatted_parts = []
    for i, mcq in enumerate(mcqs, 1):
        q_lines = [f"{i}. {mcq.question.strip()}"]
        q_lines.append("")
        q_lines.append(f"A) {mcq.options['A'].strip()}")
        q_lines.append("")
        q_lines.append(f"B) {mcq.options['B'].strip()}")
        q_lines.append("")
        q_lines.append(f"C) {mcq.options['C'].strip()}")
        q_lines.append("")
        q_lines.append(f"D) {mcq.options['D'].strip()}")
        q_lines.append("")
        correct_letter = mcq.correct
        correct_text = mcq.options.get(correct_letter, "")
        q_lines.append(f"**Correct Answer: {correct_letter}) {correct_text}**")
        if mcq.explanation:
            clean_exp = re.sub(r"(?i)\bSOURCE(?:_QUOTE)?\s*:?.*$", "", mcq.explanation).strip()
            if clean_exp:
                if clean_exp[-1] not in {'.', '!', '?', '"', "'", '`', ')'}:
                    clean_exp += "."
                q_lines.append("")
                q_lines.append(f"**Explanation:** {clean_exp}")
        formatted_parts.append("\n".join(q_lines))

    return "\n\n".join(formatted_parts)


def format_verified_answers(mcqs: List[MCQCandidate]) -> str:
    """
    Formats the verified answer key and short explanations when requested by user.
    """
    lines = []
    for i, mcq in enumerate(mcqs, 1):
        correct_letter = mcq.correct
        correct_text = mcq.options.get(correct_letter, "")
        lines.append(f"{i}. Correct Answer: {correct_letter}) {correct_text}")
        if mcq.explanation:
            lines.append(f"Explanation: {mcq.explanation}")
        lines.append("")
    return "\n".join(lines).strip()


def sanitize_overview_claims(overview_text: str, document_chunks: List[str]) -> str:
    """
    Ensures that document overview claims (e.g. number of questions) are verified in the source text.
    If the overview claims a specific count of questions (e.g. 'contains 100 questions') but that
    number is not present anywhere in the document chunks, the unverified claim is omitted.
    """
    full_text = " ".join(document_chunks)
    
    # Check for claims of question counts, e.g. "contains 100 questions" or "100 questions"
    count_matches = list(re.finditer(r"\b(\d+)\s+questions\b", overview_text, re.IGNORECASE))
    for match in count_matches:
        num = match.group(1)
        # Check if the number appears in the document text near 'question' or as a total
        if not re.search(r"\b" + re.escape(num) + r"\b", full_text):
            # The number does not appear in the document at all! Remove the claim
            overview_text = re.sub(
                rf"(?:consisting\s+of|containing|contains|with)\s+{re.escape(num)}\s+questions[,\.]?",
                "",
                overview_text,
                flags=re.IGNORECASE
            )
            overview_text = re.sub(
                rf"{re.escape(num)}\s+questions[,\.]?",
                "questions",
                overview_text,
                flags=re.IGNORECASE
            )
            
    # Clean up double spaces or awkward punctuation
    overview_text = re.sub(r"\s+", " ", overview_text).strip()
    overview_text = re.sub(r"\s+([,\.])", r"\1", overview_text)
    overview_text = re.sub(r"\(\s*\)", "", overview_text)
    
    return overview_text
