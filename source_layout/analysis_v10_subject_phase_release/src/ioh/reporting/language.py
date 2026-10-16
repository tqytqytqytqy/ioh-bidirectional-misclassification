from __future__ import annotations

_valid_word = "valid" + "ated"
_renal_abbrev = "a" + "ki"

FORBIDDEN_ALWAYS = [
    _valid_word + " hidden burden",
    "hidden burden captured",
    _renal_abbrev + " reduction",
    "continuous monitoring reduces " + _renal_abbrev,
    "error-free truth",
]


def assert_no_forbidden_language(text: str, mover_allowed: bool) -> None:
    lowered = text.lower()
    findings = [phrase for phrase in FORBIDDEN_ALWAYS if phrase in lowered]
    if not mover_allowed:
        valid_word = "valid" + "ation"
        for phrase in ["external " + valid_word, "external waveform " + valid_word, "replication"]:
            if phrase in lowered:
                findings.append(phrase)
    if findings:
        raise ValueError("; ".join(dict.fromkeys(findings)))
