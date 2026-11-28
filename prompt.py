from verify import REPORT

RULES = [
    "Use only facts explicitly present in the REPORT; do not add outside knowledge or assumptions.",
    "Write every output line as a bullet starting with '- '.",
    "End every bullet with one or more source tags copied verbatim from the REPORT.",
    "Copy every number and decimal (such as 0.9, 0.5, or 4) exactly as written in the REPORT; never calculate or estimate new numbers.",
    "Report observations and possibilities using neutral wording such as 'consistent with' or 'one possible explanation'; never state or prove a cause.",
    "If no tool flagged an issue or any tool appears under NOT CHECKED, state that plainly and note what was not checked.",
]


def build_prompt(report):
    rules = "\n".join(f"{i}. {r}" for i, r in enumerate(RULES, start=1))
    system = (
        "You rewrite an ML incident investigation report in clearer language. "
        "Follow these rules exactly:\n" + rules
    )
    user = "REPORT:\n" + report + "\n\nWrite the summary now."
    return system, user


if __name__ == "__main__":
    system, user = build_prompt(REPORT)
    print(system)
    print("-" * 60)
    print(user)