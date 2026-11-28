from evidence import threshold_evidence, missingness_evidence, drift_evidence
from report import build_report
from incident_data import load_bundle

REQUIRES = {
    "threshold": ["y_true", "scores", "old_threshold", "new_threshold"],
    "missingness": ["baseline_data", "current_data"],
    "drift": ["baseline_data", "current_data"],
}

RUNNERS = {
    "threshold": lambda b: threshold_evidence(
        b["y_true"], b["scores"], b["old_threshold"], b["new_threshold"]
    ),
    "missingness": lambda b: missingness_evidence(
        b["baseline_data"], b["current_data"]
    ),
    "drift": lambda b: drift_evidence(b["baseline_data"], b["current_data"]),
}

# For scoring on Day 24: which injected fault each flagged tool points to
TOOL_TO_FAULT = {
    "threshold": "threshold_change",
    "missingness": "missingness_spike",
    "drift": "feature_drift",
}


def missing_keys(tool, bundle):
    return [k for k in REQUIRES[tool] if bundle.get(k) is None]


def run_investigation(bundle):
    evidence, not_run = {}, {}
    for tool in REQUIRES:
        missing = missing_keys(tool, bundle)
        if missing:
            not_run[tool] = missing
        else:
            evidence[tool] = RUNNERS[tool](bundle)
    return {
        "incident_id": bundle["incident_id"],
        "evidence": evidence,
        "not_run": not_run,
        "predicted_faults": [
            TOOL_TO_FAULT[t] for t, e in evidence.items() if e["flagged"]
        ],
        "report": build_report(bundle["incident_id"], evidence, not_run),
    }


if __name__ == "__main__":
    for iid in ("INC-001", "INC-002"):
        out = run_investigation(load_bundle(iid))
        print(iid, out["predicted_faults"], "not run:", out["not_run"])
    # same incident, but the feature data was never provided
    partial = load_bundle("INC-002")
    partial["baseline_data"] = None
    partial["current_data"] = None
    out = run_investigation(partial)
    print(out["report"])
    print("predicted:", out["predicted_faults"])