"""
runbooks.py
Your RB-001 and RB-002 (content unchanged) plus a RUNBOOKS list at the
bottom so retriever.py can import them.
"""

recall_drop_runbook = {
    "runbook_id": "RB-001",
    "title": "Investigating a Recall Drop",
    "symptoms": [
        "Recall drops significantly compared with the baseline",
        "False negatives increase on a comparable labeled evaluation dataset",
        "Precision remains stable or increases while recall declines",
    ],
    "possible_causes": [
        "Decision threshold change",
        "Missingness spike",
        "Feature distribution drift",
    ],
    "evidence_to_check": [
        "Baseline and current confusion matrices from comparable labeled datasets",
        "Baseline and current classification thresholds",
        "Feature-level PSI measurements comparing reference and current data",
    ],
    "diagnostic_tests": [
        "Evaluate baseline and current thresholds on the same fixed model scores and labeled dataset; compare recall, precision, and false negatives.",
        "Calculate feature-level PSI between reference and current data, investigate substantial shifts, and evaluate performance on labeled current data when available.",
    ],
    "limitations": [
        "High PSI indicates a feature distribution shift but does not prove that the shift caused model performance degradation.",
        "A controlled threshold experiment can show whether the threshold change plausibly contributed to the recall decline, but production configuration history and logs are needed to investigate why the threshold changed.",
    ],
}

missingness_spike_runbook = {
    "runbook_id": "RB-002",
    "title": "Investigating a Missingness Spike",
    "symptoms": [
        "A sudden increase in the percentage of missing values in one or more production features",
        "Model performance deteriorates after a data pipeline or feature-processing change",
    ],
    "possible_causes": [
        "An upstream source stops supplying one or more expected fields",
        "The upstream system begins providing incomplete records",
        "A column is renamed, its data type changes, or a transformation unexpectedly produces null values",
    ],
    "evidence_to_check": [
        "Compare baseline and current missing-value rates for each feature",
        "Inspect pipeline changes, data schemas, data types, and upstream data availability",
        "Compare model performance under baseline and current missingness conditions using comparable labeled data",
    ],
    "diagnostic_tests": [
        "Compare baseline and current missingness rates per feature, identify substantial increases, and inspect the data pipeline, schema, preprocessing, and null-value handling.",
        "Inject missing values into individual features independently while keeping the model and evaluation examples fixed; apply the same preprocessing pipeline and compare recall, precision, and false negatives against baseline performance.",
    ],
    "limitations": [
        "An injected missingness experiment can demonstrate that missing values plausibly contribute to performance degradation, but it does not prove they caused the actual production incident.",
        "A pipeline that handles missing values correctly does not explain why missingness increased; upstream data sources, schema changes, collection processes, and pipeline history require further investigation.",
    ],
}

RUNBOOKS = [recall_drop_runbook, missingness_spike_runbook]
