"""
diagnostics.py
Pure measurement functions. This file does NOT build incidents and does NOT
print anything when imported. Other files import from here.
"""

import numpy as np
import math
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import recall_score, precision_score, confusion_matrix


# ---------------------------------------------------------------
# Demo data for the threshold incident (kept in ONE place only)
# ---------------------------------------------------------------
DEMO_Y_TRUE = [1] * 10 + [0] * 10

DEMO_SCORES = [
    0.95, 0.90, 0.85, 0.80, 0.75,
    0.68, 0.60, 0.50, 0.40, 0.30,
    0.90, 0.70, 0.55, 0.40, 0.30,
    0.20, 0.10, 0.08, 0.05, 0.01,
]


# ---------------------------------------------------------------
# Threshold fault (Day 2)
# ---------------------------------------------------------------
def evaluate_threshold(y_true, scores, threshold):
    """Turn scores into 0/1 predictions at a threshold, then measure."""
    predictions = [int(score >= threshold) for score in scores]

    return {
        "threshold": threshold,
        "recall": round(
            float(recall_score(y_true, predictions, zero_division=0)), 4
        ),
        "precision": round(
            float(precision_score(y_true, predictions, zero_division=0)), 4
        ),
    }


def compare_thresholds(y_true, scores, old, new):
    return {
        "baseline": evaluate_threshold(y_true, scores, old),
        "current": evaluate_threshold(y_true, scores, new),
    }


def compare_metrics(baseline, current, metric_keys):
    """Change (current - baseline) for explicitly chosen metrics only."""
    changes = {}
    for key in metric_keys:
        if key in baseline and key in current:
            changes[f"{key}_change"] = round(current[key] - baseline[key], 4)
    return changes


def compare_multiple_thresholds(y_true, scores, thresholds):
    return [evaluate_threshold(y_true, scores, t) for t in thresholds]


def simulate_threshold_fault(
    y_true, scores, baseline_threshold, faulty_threshold
):
    """Raw measurements only. incidents.py turns these into an incident."""
    result = compare_thresholds(
        y_true, scores, old=baseline_threshold, new=faulty_threshold
    )
    baseline, current = result["baseline"], result["current"]

    return {
        "baseline": baseline,
        "current": current,
        "metric_changes": compare_metrics(
            baseline, current, ["recall", "precision"]
        ),
    }


# ---------------------------------------------------------------
# Missingness fault (Day 3) - REAL injection into a real dataset
# ---------------------------------------------------------------
def columns_to_dict(X):
    return {f"f{i}": X[:, i].tolist() for i in range(X.shape[1])}

def simulate_missingness_fault(seed=7, missing_rate=0.30):
    """
    1. Make a dataset and train a model on clean data.
    2. Measure baseline recall/precision on clean test data.
    3. Blank out `missing_rate` of the model's most important feature
       in the test data, fill the blanks the way a pipeline would
       (training mean), and measure again.
    Same seed -> identical result.
    """
    rng = np.random.default_rng(seed)

    X, y = make_classification(
        n_samples=3000,
        n_features=6,
        n_informative=4,
        weights=[0.8, 0.2],
        random_state=seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    # The fault hits the feature the model relies on most
    feature_idx = int(np.argmax(np.abs(model.coef_[0])))
    fill_value = X_train[:, feature_idx].mean()  # pipeline's null handling

    def evaluate(X_eval):
        preds = model.predict(X_eval)
        return {
            "recall": round(
                float(recall_score(y_test, preds, zero_division=0)), 4
            ),
            "precision": round(
                float(precision_score(y_test, preds, zero_division=0)), 4
            ),
        }

    baseline = evaluate(X_test)

    X_faulty = X_test.copy()
    mask = rng.random(len(X_faulty)) < missing_rate
    X_faulty[mask, feature_idx] = np.nan
    X_raw = X_faulty.copy()
    X_faulty[:, feature_idx] = np.where(
        np.isnan(X_faulty[:, feature_idx]),
        fill_value,
        X_faulty[:, feature_idx],
    )

    current = evaluate(X_faulty)

    return {
        "baseline": baseline,
        "current": current,
        "metric_changes": compare_metrics(baseline, current, ["recall", "precision"]),
        "actual_missing_rate": round(float(mask.mean()), 4),
        "feature_idx": feature_idx,
        "seed": seed,
        "y_true": y_test.tolist(),
        "scores": model.predict_proba(X_faulty)[:, 1].tolist(),
        "baseline_data": columns_to_dict(X_test),
        "current_data": columns_to_dict(X_raw),
    }
def label_pair(actual, predicted):
    if actual == 1 and predicted == 1:
        return "tp"
    elif actual == 1 and predicted == 0:
        return "fn"
    elif actual == 0 and predicted == 1:
        return "fp"
    elif actual == 0 and predicted == 0:
        return "tn"

def confusion_counts(y_true, predictions):
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for actual, pred in zip(y_true, predictions):
        label = label_pair(actual, pred)
        counts[label] += 1
    return counts

def compare_confusion(y_true, scores, old, new):
    preds_old = [1 if score >= old else 0 for score in scores]
    counts_old = confusion_counts(y_true, preds_old)
    
    preds_new = [1 if score >= new else 0 for score in scores]
    counts_new = confusion_counts(y_true, preds_new)
    
    return {
        "baseline": counts_old,
        "current": counts_new
    }

def safe_divide(a, b):
    if b == 0:
        return 0.0
    return a / b

def is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))

def missing_rate(values):
    nones = sum(1 for v in values if is_missing(v))
    return safe_divide(nones, len(values))

def missing_rate_change(baseline_values, current_values):
    baseline_rate = missing_rate(baseline_values)
    current_rate = missing_rate(current_values)
    return {
        "baseline_rate": baseline_rate,
        "current_rate": current_rate,
        "missing_rate_change": current_rate - baseline_rate
    }

def check_missingness(baseline_data, current_data, flag_threshold=0.2):
    results = {}
    for feature in baseline_data:
        if feature not in current_data:
            results[feature] = {
                "status": "column_missing_in_current",
                "flagged": True
            }
            continue
        b_vals = baseline_data[feature]
        c_vals = current_data[feature]
        
        # Get baseline, current, and change metrics
        metrics = missing_rate_change(b_vals, c_vals)
        
        # Determine if the feature is flagged based on the threshold
        metrics["flagged"] = metrics["missing_rate_change"] >= flag_threshold
        
        results[feature] = metrics

    new_columns = sorted(set(current_data) - set(baseline_data))
    for feature in new_columns:
        results[feature] = {
            "status": "column_new_in_current",
            "flagged": True
        }
    return results
def psi(baseline_pcts, current_pcts):
    total = 0.0
    eps = 0.0001
    for b, c in zip(baseline_pcts, current_pcts):
        b = max(b, eps)
        c = max(c, eps)
        total += (c - b) * math.log(c / b)
    return round(total, 4)

def bin_shares(values, edges):
    idx = np.searchsorted(edges, values, side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return (counts / len(values)).tolist()

def feature_psi(baseline_values, current_values, n_bins=5):
    # drop missing values first (reusing your Day 9 helper)
    base = [v for v in baseline_values if not is_missing(v)]
    curr = [v for v in current_values if not is_missing(v)]
    
    # bin edges from BASELINE quantiles: n_bins equal-sized groups
    edges = np.quantile(base, np.linspace(0, 1, n_bins + 1)[1:-1])
    
    # compute shares and calculate PSI score
    base_shares = bin_shares(base, edges)
    curr_shares = bin_shares(curr, edges)
    return psi(base_shares, curr_shares)

def score_bands(y_true, scores, edges=(0.2, 0.4, 0.6, 0.8)):
    fraud_scores = [s for s, y in zip(scores, y_true) if y == 1]
    normal_scores = [s for s, y in zip(scores, y_true) if y == 0]
    return {
        "fraud": bin_shares(fraud_scores, list(edges)),
        "normal": bin_shares(normal_scores, list(edges)),
    }

def threshold_sweep(y_true, scores, thresholds):
    rows = []
    for t in thresholds:
        preds = [1 if s >= t else 0 for s in scores]
        c = confusion_counts(y_true, preds)
        
        # calculate recall and precision safely
        recall = safe_divide(c["tp"], c["tp"] + c["fn"])
        precision = safe_divide(c["tp"], c["tp"] + c["fp"])
        
        rows.append({
            "threshold": t,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "fn": c["fn"],
            "fp": c["fp"],
        })
    return rows
def fraud_in_zone(y_true, scores, low, high):
    fraud = sum(1 for y, s in zip(y_true, scores) if y == 1 and low <= s < high)
    normal = sum(1 for y, s in zip(y_true, scores) if y == 0 and low <= s < high)
    return {"fraud_in_zone": fraud, "normal_in_zone": normal}

# ---------------------------------------------------------------
# Quick manual tests - run only with: python diagnostics.py
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=== THRESHOLD FAULT ===")
    print(simulate_threshold_fault(DEMO_Y_TRUE, DEMO_SCORES, 0.35, 0.70))

    print("\n=== MULTIPLE THRESHOLDS ===")
    for item in compare_multiple_thresholds(
        DEMO_Y_TRUE, DEMO_SCORES, [0.35, 0.50, 0.70]
    ):
        print(item)

    print("\n=== ZERO-VALUE TEST ===")
    print(compare_metrics({"recall": 0.0}, {"recall": 0.5}, ["recall"]))

    print("\n=== MISSINGNESS FAULT ===")
    a = simulate_missingness_fault(seed=7)
    b = simulate_missingness_fault(seed=7)
    
    # Print a clean summary instead of dumping raw multi-thousand item lists
    print("Baseline Metrics:", a["baseline"])
    print("Current Metrics:", a["current"])
    print("Metric Changes:", a["metric_changes"])
    print("Actual Missing Rate:", a["actual_missing_rate"])
    print("Faulty Feature Index:", a["feature_idx"])

    assert a["baseline"] == b["baseline"]
    assert a["current"] == b["current"]
    assert a["actual_missing_rate"] == b["actual_missing_rate"] 
    print("Reproducible with same seed: OK")
    
    assert a["current"]["recall"] < a["baseline"]["recall"]
    print("Recall actually dropped after injection: OK")

    print("\n=== CONFUSION MATRIX ===")
    result = compare_confusion(DEMO_Y_TRUE, DEMO_SCORES, old=0.35, new=0.70)
    print(result)

    for name, thresh in [("baseline", 0.35), ("current", 0.70)]:
        preds = [1 if s >= thresh else 0 for s in DEMO_SCORES]
        sk_tn, sk_fp, sk_fn, sk_tp = confusion_matrix(
            DEMO_Y_TRUE, preds, labels=[0, 1]
        ).ravel()
        c = result[name]
        assert (c["tn"], c["fp"], c["fn"], c["tp"]) == (sk_tn, sk_fp, sk_fn, sk_tp), name
    print("Confusion counts match sklearn at both thresholds: OK")
    print("\n=== CONFUSION CHANGES ===")
    print(compare_metrics(result["baseline"], result["current"], ["tp", "fp", "fn", "tn"]))

    real_positives = sum(DEMO_Y_TRUE)
    for name in ["baseline", "current"]:
        c = result[name]
        assert c["tp"] + c["fn"] ==real_positives, name
    print("tp + fn equals the real positive count: OK")

    print("\n=== MISSINGNESS CHECK (regression) ===")
    baseline_data = {
        "amount":  [120.0, 80.5, None, 45.0, 230.0, 99.9, 60.0, 150.0, 75.0, 310.0],
        "age":     [34, 29, 41, 52, 38, 27, 45, 33, 60, 31],
        "country": ["IN", "IN", "US", None, "IN", "UK", "IN", "US", "IN", "UK"],
    }
    current_data = {
        "amount":  [None, 85.0, None, None, 210.0, None, 58.0, None, 72.0, 295.0],
        "age":     [36, 30, None, 50, 39, 28, 44, 35, 58, 32],
        "country_code": ["IN", "US", None, "IN", "UK", "IN", "IN", "US", "UK", "IN"],
    }
    print(check_missingness(baseline_data, current_data))
    print("\n=== NAN SUPPORT ===")
    nan = float("nan")
    print(is_missing(None), is_missing(nan), is_missing(0.0), is_missing("IN"))
    print(missing_rate([1.0, nan, None, 4.0]))
    print("\n=== PSI ===")
    print(psi([0.5, 0.3, 0.2], [0.2, 0.3, 0.5]))
    print(psi([0.5, 0.3, 0.2], [0.5, 0.3, 0.2]))
    print("\n=== BIN SHARES ===")
    print(bin_shares([5, 12, 25, 18, 10], [10, 20]))
    print("\n=== FEATURE PSI ===")
    rng = np.random.default_rng(0)
    base = rng.normal(100, 15, 1000)
    same = rng.normal(100, 15, 1000)
    shifted = rng.normal(120, 15, 1000)
    print(feature_psi(base, same))
    print(feature_psi(base, shifted))
    print("\n=== SCORE BANDS ===")
    print(score_bands(DEMO_Y_TRUE, DEMO_SCORES))
    print("\n=== THRESHOLD SWEEP ===")
    for row in threshold_sweep(DEMO_Y_TRUE, DEMO_SCORES, [0.2, 0.35, 0.5, 0.7, 0.85]):
        print(row)
    print("\n=== FRAUD IN ZONE ===")
    zone = fraud_in_zone(DEMO_Y_TRUE, DEMO_SCORES, 0.35, 0.70)
    print(zone)
    cc = compare_confusion(DEMO_Y_TRUE, DEMO_SCORES, old=0.35, new=0.70)
    assert zone["fraud_in_zone"] == cc["current"]["fn"] - cc["baseline"]["fn"]
    print("zone count matches the extra misses: OK")