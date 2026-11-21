from eval_set import EVAL_SET
from evaluate_retrieval import runbook_ranking, hit_at_k, chunks
from semantic_search import rank_semantic

for it in EVAL_SET:
    if it["expected"] is None:
        continue
    results = rank_semantic(it["query"], top_k=len(chunks))
    if hit_at_k(runbook_ranking(results), it["expected"], 1):
        continue
    print("\n", it["query"], "| expected", it["expected"])
    for chunk_id, score in results[:5]:
        print("   ", chunk_id, score)