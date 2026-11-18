import numpy as np

REQUIRED_KEYS = ["tool", "finding", "numbers", "flagged", "limitation"]

def make_evidence(tool, finding, numbers, flagged, limitation):
    return {
        "tool": tool,
        "finding": finding,
        "numbers": numbers,
        "flagged": flagged,
        "limitation": limitation,
    }

def check_evidence(e):
    errors = []
    
    # 1. Check for missing required keys
    for key in REQUIRED_KEYS:
        if key not in e:
            errors.append(f"Missing required key: {key}")
            
    # 2. Check if finding is missing or not a valid non-empty string
    finding_val = e.get("finding")
    if not isinstance(finding_val, str) or finding_val.strip() == "":
        errors.append("Finding cannot be empty or invalid.")
        
    # 3. Check if limitation is missing or not a valid non-empty string
    lim_val = e.get("limitation")
    if not isinstance(lim_val, str) or lim_val.strip() == "":
        errors.append("Limitation cannot be empty or invalid.")
        
    return errors

from diagnostics import compare_confusion, evaluate_threshold, fraud_in_zone

LIMIT_THRESHOLD = (
    "A controlled threshold experiment can show that a threshold change is "
    "consistent with the recall drop, but production configuration history and "
    "logs are needed to learn why the threshold changed."
)

def threshold_evidence(y_true, scores, old, new, min_drop=0.1):
    b = evaluate_threshold(y_true, scores, old)
    c = evaluate_threshold(y_true, scores, new)
    cc = compare_confusion(y_true, scores, old, new)
    low, high = min(old, new), max(old, new)
    zone = fraud_in_zone(y_true, scores, low, high)

    numbers = {
        "baseline_recall": b["recall"],
        "current_recall": c["recall"],
        "extra_fn": cc["current"]["fn"] - cc["baseline"]["fn"],
        "fraud_in_zone": zone["fraud_in_zone"],
        "fp_baseline": cc["baseline"]["fp"],
        "fp_current": cc["current"]["fp"],
    }

    flagged = (c["recall"] - b["recall"]) <= -min_drop
    finding = f"Recall changed from {b['recall']} to {c['recall']}. {zone['fraud_in_zone']} fraud cases scored between the two thresholds ({low} and {high}), while false alarms shifted from {cc['baseline']['fp']} to {cc['current']['fp']}."
    return make_evidence("threshold_evidence", finding, numbers, flagged, LIMIT_THRESHOLD)

from diagnostics import check_missingness

LIMIT_MISSING = (
    "Higher missing-value rates show that data arrived emptier than before, but they "
    "do not show why; upstream sources, schema changes and pipeline history need "
    "separate investigation."
)

def missingness_evidence(baseline_data, current_data, flag_threshold=0.2):
    result = check_missingness(baseline_data, current_data, flag_threshold)
    
    # 1. Categorize the findings into distinct groups
    rising = [feat for feat, data in result.items() if data.get("flagged") and "missing_rate_change" in data]
    gone = [feat for feat, data in result.items() if data.get("status") == "column_missing_in_current"]
    new = [feat for feat, data in result.items() if data.get("status") == "column_new_in_current"]
    
    # 2. Build the sentence parts dynamically from non-empty groups
    parts = []
    if rising:
        parts.append(f"Missing-value rate rose by at least {flag_threshold} in {len(rising)} feature(s): {', '.join(sorted(rising))}.")
    if gone:
        parts.append(f"{len(gone)} column(s) present in the baseline are absent from the current data: {', '.join(sorted(gone))}.")
    if new:
        parts.append(f"{len(new)} new column(s) appeared in the current data: {', '.join(sorted(new))}.")
        
    finding = " ".join(parts) if parts else "No features or columns were flagged for missingness changes."
    
    # 3. Gather flagged features across all criteria for the numbers payload and boolean flag
    flagged_features = rising + gone + new
    numbers = {"flagged_features": sorted(flagged_features), "details": result}
    
    return make_evidence(
        "missingness_evidence", finding, numbers,
        len(flagged_features) > 0, LIMIT_MISSING,
    )

from diagnostics import feature_psi, is_missing, check_missingness, compare_confusion, evaluate_threshold, fraud_in_zone

LIMIT_DRIFT = (
    "A high PSI shows that a feature's values shifted between the two periods, but it "
    "does not prove that the shift caused the recall drop; PSI is also less reliable "
    "for features with few distinct values."
)

def drift_evidence(baseline_data, current_data, psi_threshold=0.25, min_values=5):
    psi_by_feature = {}
    skipped = []
    
    for feature in baseline_data:
        if feature not in current_data:
            skipped.append(feature)
            continue
            
        base = [v for v in baseline_data[feature] if not is_missing(v)]
        curr = [v for v in current_data[feature] if not is_missing(v)]
        
        # Skip if any value is not numeric OR if either list has fewer than min_values
        if not (all(isinstance(v, (int, float)) for v in base + curr) and len(base) >= min_values and len(curr) >= min_values):
            skipped.append(feature)
            continue
            
        psi_by_feature[feature] = feature_psi(base, curr)
        
    # 1. Identify drifted features based on threshold
    drifted = [f for f, p in psi_by_feature.items() if p >= psi_threshold]
    
    # 2. Build the sentence using the parts pattern
    parts = []
    computed_count = len(psi_by_feature)
    
    if drifted:
        drifted_str = ", ".join(sorted([f"{f} ({psi_by_feature[f]:.2f})" for f in drifted]))
        parts.append(f"PSI computed for {computed_count} feature(s); {len(drifted)} exceeded threshold of {psi_threshold}: {drifted_str}.")
    else:
        parts.append(f"PSI computed for {computed_count} feature(s); no features exceeded threshold of {psi_threshold}.")
        
    if skipped:
        parts.append(f"Skipped {len(skipped)} feature(s): {', '.join(sorted(skipped))}.")
        
    finding = " ".join(parts) if parts else "No features evaluated for drift."
    
    numbers = {"psi": psi_by_feature, "skipped": sorted(skipped), "threshold": psi_threshold}
    
    return make_evidence("drift_evidence", finding, numbers, len(drifted) > 0, LIMIT_DRIFT)

if __name__ == "__main__":
    good = make_evidence("demo", "Something was observed.", {"x": 1}, False, "This cannot show why.")
    print(check_evidence(good))
    
    no_limit = make_evidence("demo", "Something was observed.", {"x": 1}, False, "")
    print(check_evidence(no_limit))
    
    broken = {"tool": "demo", "finding": "Something."}
    print(check_evidence(broken))
    from diagnostics import DEMO_Y_TRUE, DEMO_SCORES
    e = threshold_evidence(DEMO_Y_TRUE, DEMO_SCORES, 0.35, 0.70)
    print(e)
    print(check_evidence(e))
    base = {
        "amount":  [120.0, 80.5, None, 45.0, 230.0, 99.9, 60.0, 150.0, 75.0, 310.0],
        "age":     [34, 29, 41, 52, 38, 27, 45, 33, 60, 31],
        "country": ["IN", "IN", "US", None, "IN", "UK", "IN", "US", "IN", "UK"],
    }
    cur = {
        "amount":  [None, 85.0, None, None, 210.0, None, 58.0, None, 72.0, 295.0],
        "age":     [36, 30, None, 50, 39, 28, 44, 35, 58, 32],
        "country_code": ["IN", "US", None, "IN", "UK", "IN", "IN", "US", "UK", "IN"],
    }
    e = missingness_evidence(base, cur)
    print(e["finding"])
    print(check_evidence(e))
    print(missingness_evidence(base, base)["finding"])
    rng = np.random.default_rng(0)
    base_d = {"amount": list(rng.normal(100, 15, 500)), "age": list(rng.normal(40, 8, 500)), "country": ["IN"] * 500}
    cur_d  = {"amount": list(rng.normal(120, 15, 500)), "age": list(rng.normal(40, 8, 500)), "country": ["IN"] * 500}
    e = drift_evidence(base_d, cur_d)
    print(e["finding"])
    print(e["numbers"])
    print(check_evidence(e))
    print(drift_evidence(base_d, base_d)["finding"])