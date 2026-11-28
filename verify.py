import re

TAG = re.compile(r"\[([^\]\n]+)\]")
NUM = re.compile(r"\d+(?:\.\d+)?")

# Phrases that state or prove a cause. The tools only observe, so these are not allowed.
BANNED = [
    "caused by",
    "the cause is",
    "root cause is",
    "proves",
    "proven",
    "definitely",
    "because of",
    "due to",
]


def normalise(tag):
    return tag.replace(" ", "")


def sources_from_report(report):
    # maps each tag found in the report to the text of the line(s) it appears on
    sources = {}
    for line in report.splitlines():
        for tag in TAG.findall(line):
            key = normalise(tag)
            sources[key] = sources.get(key, "") + " " + TAG.sub("", line)
    return sources


def verify_summary(summary, report):
    sources = sources_from_report(report)
    problems = []
    lines = [l for l in summary.splitlines() if l.strip()]
    for n, line in enumerate(lines, start=1):
        if not line.lstrip().startswith("- "):
            problems.append(f"line {n}: every line must be a '- ' bullet")
        tags = [normalise(t) for t in TAG.findall(line)]
        if not tags:
            problems.append(f"line {n}: no source tag")
        for t in tags:
            if t not in sources:
                problems.append(f"line {n}: unknown source tag [{t}]")
        # every number in the claim must appear in the sources the line cites
        claim = re.sub(r"RB-\d{3}", "", TAG.sub("", line))
        src_numbers = NUM.findall(" ".join(sources.get(t, "") for t in tags))
        for num in NUM.findall(claim):
            if num not in src_numbers:
                problems.append(f"line {n}: number {num} is not in the cited sources")
        for phrase in BANNED:
            if phrase in line.lower():
                problems.append(f"line {n}: forbidden phrase '{phrase}'")
    return problems


REPORT = """INVESTIGATION REPORT: INC-001

1. WHAT WAS OBSERVED
- Recall changed from 0.9 to 0.5. 4 fraud cases scored between the two thresholds (0.35 and 0.7), while false alarms shifted from 4 to 2. [evidence: threshold]

2. CHECKED AND NOT FLAGGED
- No features or columns were flagged for missingness changes. [evidence: missingness]
- PSI computed for 2 feature(s); no features exceeded threshold of 0.25. [evidence: drift]

3. RUNBOOKS THAT MATCH (consistent with these findings, not proof of cause)
- RB-001 Investigating a Recall Drop (matched on the threshold finding)
    [RB-001:possible_causes:1] (score 0.631) Decision threshold change
    [RB-001:evidence_to_check:2] (score 0.6) Baseline and current classification thresholds

4. WHAT THIS EVIDENCE CANNOT SHOW
- A controlled threshold experiment can show that a threshold change is consistent with the recall drop, but production configuration history and logs are needed to learn why the threshold changed. [evidence: threshold]
- A controlled threshold experiment can show whether the threshold change plausibly contributed to the recall decline, but production configuration history and logs are needed to investigate why the threshold changed. [RB-001:limitations:2]

5. SUGGESTED NEXT CHECKS (from the matched runbooks)
- Evaluate baseline and current thresholds on the same fixed model scores and labeled dataset; compare recall, precision, and false negatives. [RB-001:diagnostic_tests:1]
"""

GOOD = """- Recall changed from 0.9 to 0.5, and 4 fraud cases scored between the two thresholds (0.35 and 0.7). [evidence: threshold]
- Missing values and drift were checked and not flagged. [evidence: missingness] [evidence: drift]
- A threshold change is one possible explanation. [RB-001:possible_causes:1]
- Production configuration history is needed to learn why the threshold changed. [evidence: threshold]
"""

BAD_NUMBER = "- Recall fell by 60 percent after the change. [evidence: threshold]\n"
BAD_TAG = "- Check the schema release notes. [RB-009:symptoms:1]\n"
BAD_NOTAG = "- The threshold was probably raised.\n"
BAD_CAUSE = "- The threshold was raised by mistake, which caused by the drop. [evidence: threshold]\n"
BAD_FORMAT = "Summary: recall dropped. [evidence: threshold]\n"

if __name__ == "__main__":
    for name, s in [
        ("GOOD", GOOD),
        ("BAD_NUMBER", BAD_NUMBER),
        ("BAD_TAG", BAD_TAG),
        ("BAD_NOTAG", BAD_NOTAG),
        ("BAD_CAUSE", BAD_CAUSE),
        ("BAD_FORMAT", BAD_FORMAT),
    ]:
        print(name, verify_summary(s, REPORT))