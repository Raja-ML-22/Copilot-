"""
validator.py
Checks a benchmark for mistakes. Returns a LIST OF ERRORS (empty = good)
instead of using assert, so a failure tells you exactly what is wrong.
"""

import json

HIDDEN_KEYS = {"injected_fault", "expected_diagnosis", "expected_behavior"}

KNOWN_FAULTS = {"threshold_change", "missingness_spike"}

# Phrases that would reveal the cause if they appear in public text.
# Text is lowercased and "_" becomes a space before checking.
LEAK_PHRASES = {
    "threshold_change": [
        "threshold change",
        "threshold increased",
        "threshold was raised",
        "raised the threshold",
    ],
    "missingness_spike": [
        "missingness",
        "missing values increased",
        "missing value spike",
    ],
}


def validate_incident(public, hidden):
    errors = []

    # 1. required fields
    if "incident_id" not in public:
        errors.append("public: missing incident_id")

    for key in HIDDEN_KEYS:
        if key not in hidden:
            errors.append(f"hidden: missing {key}")

    # 2. hidden keys must never appear in public
    leaked_keys = HIDDEN_KEYS & set(public)
    if leaked_keys:
        errors.append(f"public contains hidden keys: {sorted(leaked_keys)}")

    # 3. label must be a known fault
    fault = hidden.get("injected_fault")
    if fault not in KNOWN_FAULTS:
        errors.append(f"unknown fault label: {fault}")

    # 4. the cause must not be spelled out inside public text
    public_text = json.dumps(public).lower().replace("_", " ")
    for phrase in LEAK_PHRASES.get(fault, []):
        if phrase in public_text:
            errors.append(f"cause leaked in public text: '{phrase}'")

    # 5. ids must match between public and hidden
    if public.get("incident_id") != hidden.get("incident_id"):
        errors.append("incident_id mismatch between public and hidden")

    # 6. label must match the details of what was injected
    details = hidden.get("details", {})
    if fault == "threshold_change":
        if details.get("baseline_threshold") == details.get("faulty_threshold"):
            errors.append("threshold_change label but thresholds are equal")
    if fault == "missingness_spike":
        if not details.get("actual_missing_rate", 0) > 0:
            errors.append("missingness_spike label but no missing values injected")

    return errors


def validate_benchmark(cases):
    errors = []
    seen_ids = set()

    for case in cases:
        public, hidden = case.public(), case.hidden()
        inc_id = public.get("incident_id", "?")

        if inc_id in seen_ids:
            errors.append(f"{inc_id}: duplicate incident_id")
        seen_ids.add(inc_id)

        for err in validate_incident(public, hidden):
            errors.append(f"{inc_id}: {err}")

    return errors

import re
from runbooks import RUNBOOKS

REQUIRED_TEXT_KEYS = ["runbook_id", "title"]
REQUIRED_LIST_KEYS = [
    "symptoms",
    "possible_causes",
    "evidence_to_check",
    "diagnostic_tests",
    "limitations",
]


def check_runbook(rb):
    errors = []
    # YOU WRITE (1): add one error message for every key in
    #   REQUIRED_TEXT_KEYS + REQUIRED_LIST_KEYS that is not in rb
    for k in REQUIRED_TEXT_KEYS + REQUIRED_LIST_KEYS:
        if k not in rb:
            errors.append(f"missing required key: {k}")

    # YOU WRITE (2): for every key in REQUIRED_LIST_KEYS that IS in rb,
    #   add an error if the value is not a list, or is an empty list
    for k in REQUIRED_LIST_KEYS:
        if k in rb:
            val = rb[k]
            if not isinstance(val, list) or len(val) == 0:
                errors.append(f"key '{k}' must be a non-empty list")

    # YOU WRITE (3): add an error if rb.get("runbook_id", "") is not in the
    #   form RB-000 (hint: re.fullmatch(r"RB-\d{3}", text), and text must be a string)
    rid = rb.get("runbook_id", "")
    if not isinstance(rid, str) or not re.fullmatch(r"RB-\d{3}", rid):
        errors.append(f"invalid runbook_id format: '{rid}' (expected RB-XXX)")

    return errors


def check_all_runbooks(runbooks):
    errors = []
    seen = set()
    for rb in runbooks:
        rid = rb.get("runbook_id", "?")
        if rid in seen:
            errors.append(f"{rid}: duplicate runbook_id")
        seen.add(rid)
        errors += [f"{rid}: {e}" for e in check_runbook(rb)]
    return errors

if __name__ == "__main__":
    print(check_all_runbooks(RUNBOOKS))
    bad = {
        "runbook_id": "RB-9",
        "title": "Bad",
        "symptoms": [],
        "possible_causes": "oops",
    }
    errs = check_runbook(bad)
    print(len(errs), errs)