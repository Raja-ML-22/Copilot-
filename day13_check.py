import numpy as np
from diagnostics import simulate_missingness_fault, DEMO_Y_TRUE, DEMO_SCORES
from evidence import (threshold_evidence, missingness_evidence,
                      drift_evidence, check_evidence)


def stable_features(seed=1, n=500):
    # INC-001 assumption: the feature data did NOT change between periods
    rng = np.random.default_rng(seed)
    base = {"amount": rng.normal(100, 15, n).tolist(), "age": rng.normal(40, 8, n).tolist()}
    cur = {"amount": rng.normal(100, 15, n).tolist(), "age": rng.normal(40, 8, n).tolist()}
    return base, cur


def collect_evidence(y_true, scores, old, new, baseline_data, current_data):
    return {
        "threshold": threshold_evidence(y_true, scores, old, new),
        "missingness": missingness_evidence(baseline_data, current_data),
        "drift": drift_evidence(baseline_data, current_data),
    }


def flags(evidence):
    return {name: e["flagged"] for name, e in evidence.items()}


# Completed EXPECTED table
EXPECTED = {
    "INC-001": {"threshold": True, "missingness": False, "drift": False},
    "INC-002": {"threshold": False, "missingness": True, "drift": False},
}

if __name__ == "__main__":
    base, cur = stable_features()
    inc1 = collect_evidence(DEMO_Y_TRUE, DEMO_SCORES, 0.35, 0.70, base, cur)

    m = simulate_missingness_fault(seed=7)
    inc2 = collect_evidence(m["y_true"], m["scores"], 0.5, 0.5,
                            m["baseline_data"], m["current_data"])

    for name, ev in [("INC-001", inc1), ("INC-002", inc2)]:
        print(name, flags(ev))
        assert flags(ev) == EXPECTED[name], name
        for tool, e in ev.items():
            assert check_evidence(e) == [], (name, tool)
        if name == "INC-002":
            assert f"f{m['feature_idx']}" in inc2["missingness"]["numbers"]["flagged_features"]
    print("Right evidence, no false accusations: OK")
    