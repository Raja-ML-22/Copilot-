from day13_check import stable_features, collect_evidence
from diagnostics import simulate_missingness_fault, DEMO_Y_TRUE, DEMO_SCORES
from evaluate_retrieval import runbook_ranking
from semantic_search import rank_semantic, chunks


def evidence_to_query(evidence):
    parts = [
        item["finding"].strip()
        for item in evidence.values()
        if item.get("flagged") and item.get("finding")
    ]
    return " ".join(parts)


def investigate(evidence, top_k_runbooks=3):
    query = evidence_to_query(evidence)
    if not query:
        return {
            "query": "",
            "runbooks": [],
            "note": "No tool flagged anything; nothing to retrieve.",
        }
    results = rank_semantic(query, top_k=len(chunks))
    ranking = runbook_ranking(results)
    return {
        "query": query,
        "runbooks": ranking[:top_k_runbooks],
        "top_chunks": results[:3],
    }

def investigate_per_finding(evidence):
    matches = []
    for tool, item in evidence.items():
        if not (item.get("flagged") and item.get("finding")):
            continue
        results = rank_semantic(item["finding"], top_k=len(chunks))
        ranking = runbook_ranking(results)
        if not ranking:
            continue

        best = ranking[0]
        cites = [
            (cid, score)
            for cid, score in results
            if cid.startswith(best + ":")
        ][:2]
        limits = [
            cid
            for cid, _ in results
            if cid.startswith(best + ":") and ":limitations:" in cid
        ][:1]

        matches.append({
            "tool": tool,
            "runbook": best,
            "cites": cites,
            "limits": limits,
        })
    return matches

if __name__ == "__main__":
    base, cur = stable_features()
    inc1 = collect_evidence(DEMO_Y_TRUE, DEMO_SCORES, 0.35, 0.70, base, cur)
    m = simulate_missingness_fault(seed=7)
    inc2 = collect_evidence(
        m["y_true"], m["scores"], 0.5, 0.5, m["baseline_data"], m["current_data"]
    )
    clean = collect_evidence(DEMO_Y_TRUE, DEMO_SCORES, 0.5, 0.5, base, base)
    for name, ev in [("INC-001", inc1), ("INC-002", inc2), ("CLEAN", clean)]:
        print(name, investigate(ev))