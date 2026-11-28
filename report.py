from runbooks import RUNBOOKS
from investigate import investigate, investigate_per_finding

RB = {rb["runbook_id"]: rb for rb in RUNBOOKS}

def chunk_text(chunk_id):
    rid, section, idx = chunk_id.split(":")
    return RB[rid][section][int(idx) - 1]

def not_checked_lines(not_run):
    if not not_run:
        return []
    out = ["", "NOT CHECKED (data was not provided)"]
    for tool, keys in not_run.items():
        out.append(f"- {tool}: needs {', '.join(keys)}")
    return out


def build_report(name, evidence, not_run=None):
    flagged = {t: e for t, e in evidence.items() if e.get("flagged")}
    lines = [f"INVESTIGATION REPORT: {name}", ""]

    if not flagged:
        lines.append(
            "- No tool flagged anything, so there is not enough evidence to point to a runbook."
        )
        lines.append(
            "- No cause is suggested. This does not mean nothing is wrong: the tools only cover thresholds, missing values and drift."
        )
        lines += not_checked_lines(not_run)
        return "\n".join(lines)

    matches = investigate_per_finding(evidence)

    lines.append("1. WHAT WAS OBSERVED")
    for tool, e in flagged.items():
        lines.append(f"- {e['finding']} [evidence: {tool}]")

    unflagged = {t: e for t, e in evidence.items() if not e.get("flagged")}
    if unflagged:
        lines += ["", "2. CHECKED AND NOT FLAGGED"]
        for tool, e in unflagged.items():
            lines.append(f"- {e['finding']} [evidence: {tool}]")

    lines += [
        "",
        "3. RUNBOOKS THAT MATCH (consistent with these findings, not proof of cause)",
    ]
    if not matches:
        lines.append("- No runbook matched the findings.")
    for m in matches:
        rb = RB[m["runbook"]]
        lines.append(
            f"- {rb['runbook_id']} {rb['title']} (matched on the {m['tool']} finding)"
        )
        for cid, score in m["cites"]:
            lines.append(f"    [{cid}] (score {score}) {chunk_text(cid)}")

    lines += ["", "4. WHAT THIS EVIDENCE CANNOT SHOW"]
    for tool, e in flagged.items():
        lines.append(f"- {e['limitation']} [evidence: {tool}]")
    for m in matches:
        for cid in m["limits"]:
            lines.append(f"- {chunk_text(cid)} [{cid}]")

    lines += ["", "5. SUGGESTED NEXT CHECKS (from the matched runbooks)"]
    for m in matches:
        for i, text in enumerate(RB[m["runbook"]]["diagnostic_tests"], start=1):
            lines.append(f"- {text} [{m['runbook']}:diagnostic_tests:{i}]")

    lines += not_checked_lines(not_run)
    return "\n".join(lines)