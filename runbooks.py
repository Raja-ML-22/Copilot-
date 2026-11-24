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
    "Precision remains stable or increases while recall declines"
    ],
    "possible_causes": [
        "Decision threshold change",
        "Missingness spike",
        "Feature distribution drift"
    ],

    "evidence_to_check": [
    "Baseline and current confusion matrices from comparable labeled datasets",
    "Baseline and current classification thresholds",
    "Feature-level PSI measurements comparing reference and current data"
    ],
    "diagnostic_tests": [
    "Evaluate baseline and current thresholds on the same fixed model scores and labeled dataset; compare recall, precision, and false negatives.",
    "Calculate feature-level PSI between reference and current data, investigate substantial shifts, and evaluate performance on labeled current data when available."
    ],
    "limitations": [
    "High PSI indicates a feature distribution shift but does not prove that the shift caused model performance degradation.",
    "A controlled threshold experiment can show whether the threshold change plausibly contributed to the recall decline, but production configuration history and logs are needed to investigate why the threshold changed."
    ]
}

missingness_spike_runbook = {
    "runbook_id": "RB-002",
    "title": "Investigating a Missingness Spike",
    "symptoms": [
        'A sudden increase in the percentage of missing values in one or more production features',
        'Model performance deteriorates after a data pipeline or feature-processing change'
    ],
    "possible_causes": [
        'An upstream source stops supplying one or more expected fields',
        'The upstream system begins providing incomplete records',
        'A transformation or join in the feature pipeline unexpectedly produces null values for a feature that still exists'
    ],
    "evidence_to_check": [
    "Compare baseline and current missing-value rates for each feature",
    "Inspect pipeline changes, null-handling code and upstream data availability. If a column was renamed, removed or retyped, follow RB-004 instead",
    "Compare model performance under baseline and current missingness conditions using comparable labeled data"
    ],
    "diagnostic_tests": [
    "Compare baseline and current missingness rates per feature, identify substantial increases, and inspect the data pipeline, schema, preprocessing, and null-value handling.",
    "Inject missing values into individual features independently while keeping the model and evaluation examples fixed; apply the same preprocessing pipeline and compare recall, precision, and false negatives against baseline performance."
    ],
    "limitations": [
    "An injected missingness experiment can demonstrate that missing values plausibly contribute to performance degradation, but it does not prove they caused the actual production incident.",
    "A pipeline that handles missing values correctly does not explain why missingness increased; upstream data sources, schema changes, collection processes, and pipeline history require further investigation."
    ]
}
feature_drift_runbook = {
    "runbook_id": "RB-003",
    "title": "Investigating Feature Distribution Drift",
    "symptoms": [
        "A feature's Population Stability Index (PSI) exceeds the critical monitoring threshold between baseline and current periods.",
        "Model predictions or scores shift across score bands without any manual change to the decision threshold.",
    ],
    "possible_causes": [
        "Seasonal shifts or demographic changes in the incoming user population over time.",
        "A new merchant category, customer segment or promotion changes the typical values of transaction features.",
    ],
    "evidence_to_check": [
        "Feature-level PSI values and the list of skipped features produced by drift_evidence.",
        "Summary statistics (mean, spread, range) for each flagged feature, baseline versus current. The current tools do not produce these, so compute them from the raw data.",
    ],
    "diagnostic_tests": [
        "Run drift_evidence on baseline and current data to get a PSI value for each numeric feature, and review the features at or above the threshold (0.25 by default).",
        "Check the skipped list: text features and features with too few values are not scored and need a separate check.",
    ],
    "limitations": [
        "PSI flags that a distribution has shifted but cannot explain the underlying root cause.",
        "A high PSI value does not automatically prove that model classification performance has degraded.",
        "PSI is computed only for numeric features with enough values; text and sparse features are skipped and reported.",
    ],
}
schema_change_runbook = {
    "runbook_id": "RB-004",
    "title": "Investigating an Upstream Schema or Data-Type Change",
    "symptoms": [
        "A feature column that existed in the baseline is absent from current data, or an unexpected new column appears",
        "A numeric feature starts arriving as text, or its values change scale or units",
        "Model performance deteriorates shortly after a data pipeline or source-system release",
    ],
    "possible_causes": [
        "A column is renamed, removed or added by an upstream team",
        "A data type changes (for example number to string) and a transformation turns values into nulls",
        "A unit or scale changes (for example rupees to paise, or seconds to milliseconds)",
    ],
    "evidence_to_check": [
        "Column lists for baseline and current data, including columns missing from current and new columns in current",
        "Data types and value ranges per feature, baseline versus current. The current tools do not produce these, so check the raw data or schema definitions.",
        "Pipeline, schema and release history for the period when performance changed. This is outside information, so ask the data engineering or platform team.",
    ],
    "diagnostic_tests": [
        "Compare baseline and current column sets and report columns that vanished or appeared, without assuming a rename.",
        "Compare per-feature missing rates and PSI; a unit or scale change typically appears as a large PSI shift on that feature while missing rates stay unchanged.",
    ],
    "limitations": [
        "A column-set difference shows that the schema changed but not why; a rename, a bug and two unrelated changes look identical.",
        "PSI can flag a scale shift but cannot tell a unit change from a genuine change in the population; release and pipeline history are needed.",
    ],
}
RUNBOOKS = [
    recall_drop_runbook,
    missingness_spike_runbook,
    feature_drift_runbook,
    schema_change_runbook,
]