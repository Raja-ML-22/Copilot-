# Data only. Each item: the query, the right runbook id (None if no runbook fits), and a kind.

EVAL_SET = [
    # RB-001: Decision threshold shift / recall drop (5 items: 1 plain, 3 paraphrase, 1 tricky)
    {"query": "recall dropped after the threshold was raised", "expected": "RB-001", "kind": "plain"},
    {"query": "we are catching far fewer fraud cases than last month", "expected": "RB-001", "kind": "paraphrase"},
    {"query": "alerting way less often now, slipping right past our detector", "expected": "RB-001", "kind": "paraphrase"},
    {"query": "cutoffs got stricter overnight and false negatives shot up", "expected": "RB-001", "kind": "paraphrase"},
    {"query": "positives tanked right after a config push and score spread narrowed", "expected": "RB-001", "kind": "tricky"},

    # RB-002: Missingness spike / null values (5 items: 1 plain, 3 paraphrase, 1 tricky)
    {"query": "missing values increased in the input features", "expected": "RB-002", "kind": "plain"},
    {"query": "half the incoming rows are blank on credit score today", "expected": "RB-002", "kind": "paraphrase"},
    {"query": "empty fields everywhere on the checkout form since morning", "expected": "RB-002", "kind": "paraphrase"},
    {"query": "a required column is coming through with nothing in it", "expected": "RB-002", "kind": "paraphrase"},
    {"query": "nulls jumped on income right after an upstream schema edit", "expected": "RB-002", "kind": "tricky"},

    # RB-003: Feature distribution drift / PSI (5 items: 1 plain, 4 paraphrase)
    {"query": "feature distribution drift detected in production", "expected": "RB-003", "kind": "plain"},
    {"query": "customers are behaving differently lately and input histograms look off", "expected": "RB-003", "kind": "paraphrase"},
    {"query": "age and transaction amounts drifted slowly over the past few weeks", "expected": "RB-003", "kind": "paraphrase"},
    {"query": "the population profile shifted away from what we saw at training", "expected": "RB-003", "kind": "paraphrase"},
    {"query": "score buckets are moving gradually as user habits change", "expected": "RB-003", "kind": "paraphrase"},

    # RB-004: Upstream schema / data-type / unit or scale change (5 items: 1 plain, 4 paraphrase)
    {"query": "upstream schema or data type changed", "expected": "RB-004", "kind": "plain"},
    {"query": "latency values suddenly jumped by 1000x after a backend deploy", "expected": "RB-004", "kind": "paraphrase"},
    {"query": "a column name changed upstream and broke our feature pipeline", "expected": "RB-004", "kind": "paraphrase"},
    {"query": "numeric amounts are arriving as strings now and casting failed", "expected": "RB-004", "kind": "paraphrase"},
    {"query": "an extra field showed up in the payload and a known column disappeared", "expected": "RB-004", "kind": "paraphrase"},

    # Out of scope (4 items: expected=None)
    {"query": "gpu ran out of memory during training", "expected": None, "kind": "out_of_scope"},
    {"query": "docker build failed because apt-get returned 404", "expected": None, "kind": "out_of_scope"},
    {"query": "css grid layout is overlapping on mobile screens", "expected": None, "kind": "out_of_scope"},
    {"query": "kubernetes pod keeps crashing with exit code 137", "expected": None, "kind": "out_of_scope"},
        # Near-domain but NOT covered by any runbook (a good retriever finds no good match)
    {"query": "ground truth labels arrive two weeks late so we cannot compute recall yet", "expected": None, "kind": "out_of_scope_near"},
    {"query": "predictions take twice as long to return since the last release", "expected": None, "kind": "out_of_scope_near"},
    {"query": "the nightly retraining job failed and the model is three months stale", "expected": None, "kind": "out_of_scope_near"},
]

from collections import Counter
from eval_set import EVAL_SET
print(len(EVAL_SET), Counter(item["expected"] for item in EVAL_SET))