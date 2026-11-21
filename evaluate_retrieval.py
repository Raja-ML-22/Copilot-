from collections import defaultdict

from eval_set import EVAL_SET
from runbooks import RUNBOOKS
from chunker import make_chunks, rank_chunks

chunks = make_chunks(RUNBOOKS)


def runbook_ranking(chunk_results):
    seen = []
    for chunk_id, _ in chunk_results:
        rid = chunk_id.split(":")[0]
        if rid not in seen:
            seen.append(rid)
    return seen


def hit_at_k(ranking, expected, k):
    if expected is None:
        return None
    return expected in ranking[:k]


def evaluate(rank_fn, name):
    in_scope = [it for it in EVAL_SET if it["expected"] is not None]
    out_scope = [it for it in EVAL_SET if it["expected"] is None]

    h1 = h3 = 0
    by_kind = defaultdict(lambda: [0, 0])
    misses = []
    for it in in_scope:
        ranking = runbook_ranking(rank_fn(it["query"]))
        a = hit_at_k(ranking, it["expected"], 1)
        b = hit_at_k(ranking, it["expected"], 3)
        h1 += a
        h3 += b
        by_kind[it["kind"]][0] += a
        by_kind[it["kind"]][1] += 1
        if not a:
            misses.append((it["query"], it["expected"], ranking[:3]))

    n = len(in_scope)
    print(f"\n=== {name} ===")
    print(f"hit@1: {h1}/{n} = {h1 / n:.2f}   hit@3: {h3}/{n} = {h3 / n:.2f}")
    for kind, (ok, total) in by_kind.items():
        print(f"  {kind}: hit@1 {ok}/{total}")
    print("misses at rank 1:")
    for q, exp, got in misses:
        print(f"  {q!r}: expected {exp}, got {got}")
    print("out-of-scope: best chunk found:")
    for it in out_scope:
        print(f"  {it['query']!r} -> {rank_fn(it['query'])[:1]}")


if __name__ == "__main__":
    # self-tests for your two functions
    assert runbook_ranking([("RB-003:symptoms:1", 3), ("RB-001:limitations:1", 2),
                            ("RB-003:limitations:2", 2)]) == ["RB-003", "RB-001"]
    assert hit_at_k(["RB-003", "RB-001"], "RB-001", 1) is False
    assert hit_at_k(["RB-003", "RB-001"], "RB-001", 2) is True
    assert hit_at_k([], "RB-001", 3) is False
    assert hit_at_k(["RB-001"], None, 3) is None
    print("self-tests passed")

    evaluate(lambda q: rank_chunks(q, chunks, top_k=len(chunks)), "keyword")

    from semantic_search import rank_semantic
    evaluate(lambda q: rank_semantic(q, top_k=len(chunks)), "semantic")