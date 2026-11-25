from runbooks import RUNBOOKS
from investigate import investigate, investigate_per_finding

RB = {rb["runbook_id"]: rb for rb in RUNBOOKS}


def chunk_text(chunk_id):
    rid, section, idx = chunk_id.split(":")
    return RB[rid][section][int(idx) - 1]


def build_report(name, evidence):
    flagged = {t: e for t, e in evidence.items() if e.get("flagged")}
    lines = [f"INVESTIGATION REPORT: {name}", ""]

    if not flagged:
        lines.append(
            "- No tool flagged anything, so there is not enough evidence to point to a runbook."
        )
        lines.append(
            "- No cause is suggested because no active anomaly was observed."
        )
        return "\n".join(lines)

    matches = investigate_per_finding(evidence)

    lines.append("1. WHAT WAS OBSERVED")
    for tool, e in flagged.items():
        lines.append(f"- {e['finding']} [evidence: {tool}]")

    lines += ["", "2. RUNBOOKS THAT MATCH (consistent with these findings, not proof of cause)"]
    if not matches:
        lines.append("- No runbook matched the findings.")
    for m in matches:
        rb = RB[m["runbook"]]
        lines.append(f"- {rb['runbook_id']} {rb['title']} (matched on the {m['tool']} finding)")
        for cid, score in m["cites"]:
            lines.append(f"    [{cid}] (score {score}) {chunk_text(cid)}")

    lines += ["", "3. WHAT THIS EVIDENCE CANNOT SHOW"]
    for tool, e in flagged.items():
        lines.append(f"- {e['limitation']} [evidence: {tool}]")
    for m in matches:
        for i, text in enumerate(RB[m["runbook"]]["limitations"], start=1):
            lines.append(f"- {text} [{m['runbook']}:limitations:{i}]")

    lines += ["", "4. SUGGESTED NEXT CHECKS (from the matched runbooks)"]
    for m in matches:
        for i, text in enumerate(RB[m["runbook"]]["diagnostic_tests"], start=1):
            lines.append(f"- {text} [{m['runbook']}:diagnostic_tests:{i}]")
    return "\n".join(lines)


if __name__ == "__main__":
    from day13_check import stable_features, collect_evidence
    from diagnostics import simulate_missingness_fault, DEMO_Y_TRUE, DEMO_SCORES

    base, cur = stable_features()
    inc1 = collect_evidence(DEMO_Y_TRUE, DEMO_SCORES, 0.35, 0.70, base, cur)
    m = simulate_missingness_fault(seed=7)
    inc2 = collect_evidence(m["y_true"], m["scores"], 0.5, 0.5,
                            m["baseline_data"], m["current_data"])
    clean = collect_evidence(DEMO_Y_TRUE, DEMO_SCORES, 0.5, 0.5, base, base)
    # pretend ONE incident where both the threshold and missingness tools flagged
    mixed = {"threshold": inc1["threshold"], "missingness": inc2["missingness"],
             "drift": inc1["drift"]}

    for name, ev in [("INC-001", inc1), ("INC-002", inc2), ("CLEAN", clean), ("MIXED", mixed)]:
        print(build_report(name, ev))
        print("-" * 60)

    print("combined query ->", investigate(mixed)["runbooks"])
    print("per-finding    ->", [x["runbook"] for x in investigate_per_finding(mixed)])