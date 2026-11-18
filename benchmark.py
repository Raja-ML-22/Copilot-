"""
benchmark.py
Assembles the benchmark, validates it, proves the validator works,
and saves public + hidden files separately.

Run with:  python benchmark.py
"""

import copy
import json
from pathlib import Path

from incidents import build_benchmark
from validator import validate_benchmark, validate_incident

OUT_DIR = Path(__file__).parent


def save_benchmark(cases):
    """Public file = what the Copilot may load. Hidden file = scoring only."""
    public = [c.public() for c in cases]
    hidden = [c.hidden() for c in cases]

    (OUT_DIR / "incidents_public.json").write_text(json.dumps(public, indent=2))
    (OUT_DIR / "labels_hidden.json").write_text(json.dumps(hidden, indent=2))


def expect_rejected(name, public, hidden):
    errors = validate_incident(public, hidden)
    assert errors, f"validator FAILED to reject: {name}"
    print(f"  rejected '{name}' ->", errors)


if __name__ == "__main__":
    cases = build_benchmark()

    print("=== 1. VALID BENCHMARK ===")
    errors = validate_benchmark(cases)
    assert errors == [], errors
    print("Total incidents:", len(cases))
    for c in cases:
        print(" ", c.public()["incident_id"], "-", c.public()["description"])
    print("Validator passed with no errors.")

    print("\n=== 2. REPRODUCIBILITY (same seed -> identical) ===")
    again = build_benchmark()
    assert [c.public() for c in cases] == [c.public() for c in again]
    assert [c.hidden() for c in cases] == [c.hidden() for c in again]
    print("Two builds are identical: OK")

    print("\n=== 3. VALIDATOR MUST REJECT BROKEN INCIDENTS ===")
    good_public = cases[0].public()
    good_hidden = cases[0].hidden()

    leaky_text = copy.deepcopy(good_public)
    leaky_text["description"] = "Fraud recall dropped after a threshold change."
    expect_rejected("cause named in description", leaky_text, good_hidden)

    leaky_key = copy.deepcopy(good_public)
    leaky_key["injected_fault"] = "threshold_change"
    expect_rejected("hidden key inside public input", leaky_key, good_hidden)

    wrong_id = copy.deepcopy(good_hidden)
    wrong_id["incident_id"] = "INC-999"
    expect_rejected("id mismatch", good_public, wrong_id)

    bad_label = copy.deepcopy(good_hidden)
    bad_label["injected_fault"] = "made_up_fault"
    expect_rejected("unknown fault label", good_public, bad_label)

    mismatch = copy.deepcopy(good_hidden)
    mismatch["details"]["faulty_threshold"] = mismatch["details"]["baseline_threshold"]
    expect_rejected("label says threshold_change but thresholds equal", good_public, mismatch)

    print("\n=== 4. SAVING FILES ===")
    save_benchmark(cases)
    print("Saved incidents_public.json and labels_hidden.json")
