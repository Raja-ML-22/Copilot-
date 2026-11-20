from runbooks import RUNBOOKS
from retriever import tokenize

SECTIONS = [
    "symptoms",
    "possible_causes",
    "evidence_to_check",
    "diagnostic_tests",
    "limitations",
]


def make_chunks(runbooks):
    chunks = []
    for rb in runbooks:
        for section in SECTIONS:
            for i, item in enumerate(rb[section], start=1):
                text = f"{rb['title']} | {section}: {item}"
                chunk_id = f"{rb['runbook_id']}:{section}:{i}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "runbook_id": rb["runbook_id"],
                    "title": rb["title"],
                    "section": section,
                    "text": text,
                })
    return chunks


def check_chunks(chunks, runbooks):
    errors = []
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        errors.append("duplicate chunk_id")
    for c in chunks:
        if not c["text"].strip():
            errors.append(f"{c['chunk_id']}: empty text")
    expected = sum(len(rb[s]) for rb in runbooks for s in SECTIONS)
    if len(chunks) != expected:
        errors.append(f"expected {expected} chunks, got {len(chunks)}")
    return errors


def rank_chunks(query, chunks, top_k=3):
    q = set(tokenize(query))
    scored = []
    for c in chunks:
        score = len(q & set(tokenize(c["text"])))
        if score > 0:
            scored.append((c["chunk_id"], score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    chunks = make_chunks(RUNBOOKS)
    print("chunks:", len(chunks))
    print(chunks[0])
    print("errors:", check_chunks(chunks, RUNBOOKS))
    for q in [
        "recall drop threshold change",
        "missing values increased",
        "feature distribution shift psi",
        "column renamed upstream",
        "unit changed from seconds to milliseconds",
        "numbers suddenly 100 times larger",
        "nulls appearing in a feature",
    ]:
        print(q, "->", rank_chunks(q, chunks))