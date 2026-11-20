"""
retriever.py
Keyword-overlap retriever over the REAL runbooks in runbooks.py.
(Week 4 replaces this with proper RAG retrieval.)
"""

import re

from runbooks import RUNBOOKS

STOPWORDS = {
    "a", "an", "the", "in", "on", "of", "to", "and", "or", "is", "are",
    "was", "were", "for", "with", "that", "this", "after", "from", "at", "by",
}


def tokenize(text):
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def runbook_to_text(runbook):
    """Join title + every list field into one searchable string."""
    parts = [runbook["title"]]
    for field_name in (
        "symptoms",
        "possible_causes",
        "evidence_to_check",
        "diagnostic_tests",
        "limitations",
    ):
        parts.extend(runbook.get(field_name, []))
    return " ".join(parts)


def score_runbook(query, runbook_text):
    return len(set(tokenize(query)) & set(tokenize(runbook_text)))


def rank_runbooks(query, runbooks=None):
    runbooks = RUNBOOKS if runbooks is None else runbooks

    scored = []
    for runbook in runbooks:
        score = score_runbook(query, runbook_to_text(runbook))
        if score > 0:  # drop runbooks with no overlap at all
            scored.append((runbook["runbook_id"], runbook["title"], score))

    scored.sort(key=lambda item: item[2], reverse=True)
    return scored

if __name__ == "__main__":
    for q in [
        "recall drop threshold change",
        "missing values increased",
        "feature distribution shift psi",
        "column renamed upstream",
        "unit changed from seconds to milliseconds",
    ]:
        print(q, "->", rank_runbooks(q))