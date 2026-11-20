"""
incidents.py
Defines what an incident IS (dataclasses) and builds the benchmark cases.
Numbers come from diagnostics.py, so there is only one source of truth.
"""

from dataclasses import dataclass, field, asdict

from diagnostics import (
    DEMO_Y_TRUE,
    DEMO_SCORES,
    simulate_threshold_fault,
    simulate_missingness_fault,
)

# One vocabulary for fault labels, used everywhere
FAULT_THRESHOLD = "threshold_change"
FAULT_MISSINGNESS = "missingness_spike"

EXPECTED_BEHAVIOR = "investigate_and_report_uncertainty"


# ---------------------------------------------------------------
# Schema (Day 1)
# ---------------------------------------------------------------
@dataclass
class Incident:
    """Everything the Copilot is ALLOWED to see."""
    incident_id: str
    incident_type: str
    model_type: str
    description: str                 # symptom only - never the cause
    observed_metrics: dict
    available_evidence: list = field(default_factory=list)
    missing_information: list = field(default_factory=list)


@dataclass
class GroundTruth:
    """Evaluation only. Never sent to the Copilot."""
    injected_fault: str
    expected_diagnosis: str
    expected_behavior: str
    details: dict = field(default_factory=dict)


@dataclass
class BenchmarkCase:
    incident_input: Incident
    ground_truth: GroundTruth

    def public(self):
        return asdict(self.incident_input)

    def hidden(self):
        return {
            "incident_id": self.incident_input.incident_id,
            **asdict(self.ground_truth),
        }


# ---------------------------------------------------------------
# INC-001: threshold fault
# ---------------------------------------------------------------
def build_inc_001():
    baseline_threshold = 0.35
    faulty_threshold = 0.70

    m = simulate_threshold_fault(
        DEMO_Y_TRUE, DEMO_SCORES, baseline_threshold, faulty_threshold
    )
    b, c = m["baseline"], m["current"]

    incident = Incident(
        incident_id="INC-001",
        incident_type="fraud_detection",
        model_type="binary_classification",
        # symptom only - does NOT name the cause
        description=f"Fraud recall dropped from {b['recall']} to {c['recall']}.",
        observed_metrics={
            "baseline_recall": b["recall"],
            "current_recall": c["recall"],
            "baseline_precision": b["precision"],
            "current_precision": c["precision"],
            # Easy case: thresholds are visible. A harder INC-003 can hide them.
            "baseline_threshold": baseline_threshold,
            "current_threshold": faulty_threshold,
        },
        available_evidence=[],
        missing_information=[
            "confusion_matrix",
            "score_distribution",
            "config_change_history",
            "feature_drift",
            "feature_missing_rates",
        ],
    )

    truth = GroundTruth(
        injected_fault=FAULT_THRESHOLD,
        expected_diagnosis="threshold_change_is_plausible",
        expected_behavior=EXPECTED_BEHAVIOR,
        details={
            "baseline_threshold": baseline_threshold,
            "faulty_threshold": faulty_threshold,
            "discriminating_test": (
                "Evaluate both thresholds on the same fixed scores "
                "and labels; compare recall, precision, false negatives."
            ),
        },
    )

    return BenchmarkCase(incident, truth)


# ---------------------------------------------------------------
# INC-002: missingness fault
# ---------------------------------------------------------------
def build_inc_002(seed=7, missing_rate=0.30):
    m = simulate_missingness_fault(seed=seed, missing_rate=missing_rate)
    b, c = m["baseline"], m["current"]

    incident = Incident(
        incident_id="INC-002",
        incident_type="fraud_detection",
        model_type="binary_classification",
        # symptom only - the Copilot must discover the missing values itself
        description=f"Fraud recall dropped from {b['recall']} to {c['recall']}.",
        observed_metrics={
            "baseline_recall": b["recall"],
            "current_recall": c["recall"],
            "baseline_precision": b["precision"],
            "current_precision": c["precision"],
        },
        available_evidence=[],
        missing_information=[
            "confusion_matrix",
            "score_distribution",
            "pipeline_change_history",
            "feature_drift",
            "feature_missing_rates",
        ],
    )

    truth = GroundTruth(
        injected_fault=FAULT_MISSINGNESS,
        expected_diagnosis="missingness_spike_is_plausible",
        expected_behavior=EXPECTED_BEHAVIOR,
        details={
            "seed": m["seed"],
            "feature_idx": m["feature_idx"],
            "actual_missing_rate": m["actual_missing_rate"],
            "discriminating_test": (
                "Inject missing values into the suspect feature with the "
                "model and examples fixed; compare against baseline."
            ),
        },
    )

    return BenchmarkCase(incident, truth)


def build_benchmark():
    return [build_inc_001(), build_inc_002()]


if __name__ == "__main__":
    for case in build_benchmark():
        print(case.public())
        print(case.hidden())
        print()
