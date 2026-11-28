from day13_check import stable_features
from diagnostics import simulate_missingness_fault, DEMO_Y_TRUE, DEMO_SCORES


def load_bundle(incident_id):
    if incident_id == "INC-001":
        base, cur = stable_features()
        return {
            "incident_id": "INC-001",
            "y_true": DEMO_Y_TRUE,
            "scores": DEMO_SCORES,
            "old_threshold": 0.35,
            "new_threshold": 0.70,
            "baseline_data": base,
            "current_data": cur,
        }
    if incident_id == "INC-002":
        m = simulate_missingness_fault(seed=7)
        return {
            "incident_id": "INC-002",
            "y_true": m["y_true"],
            "scores": m["scores"],
            "old_threshold": 0.5,
            "new_threshold": 0.5,
            "baseline_data": m["baseline_data"],
            "current_data": m["current_data"],
        }
    raise ValueError(f"unknown incident: {incident_id}")


FORBIDDEN_KEYS = {
    "feature_idx",
    "actual_missing_rate",
    "seed",
    "injected_fault",
    "baseline",
    "current",
}

if __name__ == "__main__":
    for iid in ("INC-001", "INC-002"):
        b = load_bundle(iid)
        leaked = FORBIDDEN_KEYS & set(b)
        assert not leaked, f"{iid} leaks ground truth: {leaked}"
        print(iid, sorted(b))
    print("no ground truth in the bundles: OK")