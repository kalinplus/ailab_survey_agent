"""Source-bound writing regressions. Legacy freeform/tier fixtures encoded bypasses;
these tests require the same source contract in normal and degraded paths.
"""

import copy
import json

import pytest

from tools import write_survey as writer
from tools.models.artifacts import Evidence, EvidenceStore, ParsedPapers
from tools.nlp.nli_verifier import NLIResult
from tools.verify.claim_mapper import run as map_claims
from tools.verify.source_contract import quote_diagnostic
from tools.verify.text_units import sentence_units


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def chat(self, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0)


def source(pid="paper:alpha", role="own_method", kind="abstract"):
    return dict(paper_id=pid, evidence_id=pid + "_abs_p0_0", evidence_ids=[pid + "_abs_p0_0"],
                claim_text=f"{'Beta' if pid.endswith('beta') else 'Alpha'} learns policy and value predictions for tree search",
                dimension="method", source_role=role, source_type=kind,
                source_quote="We learn a model that predicts policy and value for planning.",
                support_type="direct", confidence=0.8)


def section():
    return dict(section_id="sec_01", section_title="Planning", section_goal="Rule-based taxonomy fallback for Planning",
                selected_papers=[dict(paper_id="paper:alpha", title="Alpha", sources=[source()]),
                                 dict(paper_id="paper:beta", title="Beta", sources=[source("paper:beta")])],
                artifact_slots=[])


def claim(text="Alpha uses learned policy and value predictions during search", **changes):
    return dict(text=text, cite=["P1"], sources=["P1S1"], scope="method", **changes)


def test_source_binding_roundtrip_and_abstract_explicit_method():
    llm = ScriptedLLM([json.dumps([claim()])])
    collected = []
    paragraphs = writer._llm_section_body(section(), llm.chat, False, set(), collected,
                                         {"paper:alpha": "abstract_only", "paper:beta": "abstract_only"})
    assert len(llm.prompts) == 1
    assert "almost verbatim" not in llm.prompts[0]
    assert "Rule-based taxonomy fallback" not in llm.prompts[0]
    assert "P1S1" in llm.prompts[0]
    assert collected[0]["source_bindings"][0]["evidence_id"] == "paper:alpha_abs_p0_0"
    assert sentence_units(" ".join(paragraphs), {"paper:alpha"}) == [(collected[0]["text"], ["paper:alpha"])]


@pytest.mark.parametrize("mutation", [
    {"sources": ["P2S1"]}, {"sources": ["P1S99"]}, {"sources": []},
    {"cite": ["P9"]}, {"cite": []}, {"scope": "invented"},
    {"text": "First claim. Another claim"},
])
def test_low_overlap_never_bypasses_binding_or_scope(mutation):
    raw = claim()
    raw.update(mutation)
    llm = ScriptedLLM([json.dumps([raw]), "unsafe fallback [P1]."])
    with pytest.raises(ValueError):
        writer._llm_section_body(section(), llm.chat, False, set())
    # Alias-format mutations get exactly one corrective retry (wave8 ①);
    # every other rejection fails fast on the first reply.
    alias_mutations = ({"sources": ["P1S99"]}, {"cite": ["P9"]})
    assert len(llm.prompts) == (2 if mutation in alias_mutations else 1)


def test_background_cannot_become_own_method_even_in_method_paragraph():
    sec = section()
    sec["selected_papers"][0]["sources"][0].update(
        source_role="background", source_quote="Tree-based planning methods have enjoyed huge success where a perfect simulator is available.")
    with pytest.raises(ValueError):
        writer._llm_section_body(sec, ScriptedLLM([json.dumps([claim()])]).chat, False, set())


def test_no_freeform_reask_after_invalid_json():
    llm = ScriptedLLM(["**Future Directions**\nArbitrary excerpt [P1].", "bypass"])
    with pytest.raises(ValueError):
        writer._llm_section_body(section(), llm.chat, False, set())
    assert len(llm.prompts) == 1


def test_templates_never_assign_raw_abstract_or_legacy_field_roles():
    papers = [{"paper_id": "paper:mu", "title": "MuZero", "method": "A perfect simulator is available.",
               "contribution": "Constructing agents has long been a challenge.",
               "evidence_snippets": ["Arbitrary background excerpt."]}]
    assert writer._template_summary(papers, False, 1, set()) == ""
    assert writer._template_comparison(papers, False) == ""
    assert writer._limitation_pool(papers) == []
    text = " ".join(writer._build_section_text(dict(section(), selected_papers=papers), False))
    assert "No safely attributed claims" in text
    assert "perfect simulator" not in text
    assert "Arbitrary background" not in text


def test_template_keeps_complete_safe_claim_with_own_citation():
    text = writer._template_comparison(section()["selected_papers"], False)
    assert "Alpha learns policy and value predictions for tree search [paper:alpha]." in text
    assert "contributes" not in text


def test_quote_guard_requires_clause_length_and_coverage():
    copied = "The learned environment model enables the agent to predict future observations and rewards across long interactive trajectories"
    assert quote_diagnostic(copied, [copied])["quote_like"]
    technical = "partially observable Markov decision process with continuous state and action spaces"
    assert not quote_diagnostic(technical, [technical])["quote_like"]
    assert not quote_diagnostic("An agent plans ahead using predictions learned from experience", [copied])["quote_like"]


@pytest.mark.parametrize("label,status", [("neutral", "weak"), ("contradiction", "unsupported")])
def test_copied_source_is_not_upgraded_to_supported(label, status):
    text = "The learned environment model enables the agent to predict future observations and rewards across long interactive trajectories"
    ev = Evidence(evidence_id="e", paper_id="paper:alpha", text=text)
    class NLI:
        def best_match(self, claim, evidences):
            return NLIResult(label, "indirect" if label == "neutral" else "contradictory", 0.9)
    result = map_claims("t", text + " [paper:alpha].", EvidenceStore(task_id="t", evidence=[ev]),
                        ParsedPapers(task_id="t", papers=[]), NLI())
    assert result.entries[0].status == status
    assert result.entries[0].quote_diagnostic["quote_like"]


def test_role_violation_is_separate_from_entailment():
    src = source(role="background")
    ev = Evidence(evidence_id=src["evidence_id"], paper_id=src["paper_id"], text=src["source_quote"], supports_claims=[src])
    class NLI:
        def best_match(self, claim, evidences):
            return NLIResult("entailment", "direct", 0.8)
    row = dict(text="Alpha needs a simulator", cites=[src["paper_id"]], scope="method", source_bindings=[src])
    result = map_claims("t", row["text"] + " [paper:alpha].", EvidenceStore(task_id="t", evidence=[ev]),
                        ParsedPapers(task_id="t", papers=[]), NLI(), structured_claims=[row])
    assert result.entries[0].nli_status == "supported"
    assert result.entries[0].status == "unsupported"
    assert result.entries[0].source_role_violation == "source_role_scope_mismatch"


def test_writer_selection_rejects_unknown_roles_and_unbound_rows():
    src = source()
    ev = dict(evidence_id=src["evidence_id"], paper_id=src["paper_id"], text=src["source_quote"], source_type="abstract", supports_claims=[src])
    card = dict(paper_id=src["paper_id"], title="Alpha", method="untrusted field")
    entry = writer._selected_paper_entry(card, {src["paper_id"]: [ev]})
    assert entry["method"] == src["claim_text"]
    for mutation in ({"source_role": "unknown"}, {"evidence_ids": ["fabricated"]}, {"paper_id": "other"}, {"support_type": "indirect"}):
        altered = copy.deepcopy(ev)
        altered["supports_claims"][0].update(mutation)
        assert writer._selected_paper_entry(card, {src["paper_id"]: [altered]})["sources"] == []


def test_final_composition_no_goal_leaks_or_duplicate_assets():
    sec = section()
    sec["artifact_slots"] = [dict(artifact_id="evaluation_protocol_matrix", placement="after_topic_paragraph", caption="Evaluation")]
    sec["selected_papers"][0]["sources"][0]["source_role"] = "own_result"
    cards = [dict(paper_id="paper:alpha", title="Alpha", year=2024)]
    collected = []
    text = writer._render_survey("World Models", "en", [sec], [], cards, {}, structured_out=collected)
    assert "Rule-based taxonomy fallback" not in text
    assert text.count("](evaluation_protocol_matrix)") == 1
    assert text.count("## Future Directions") == 1
    assert "**Future Directions**" not in text
    conclusion = text.split("## Conclusion")[1].split("## References")[0]
    assert "Alpha learns policy and value" in conclusion
    assert "The field is moving" not in conclusion
    assert collected and all(row["source_bindings"] for row in collected)


def test_future_proposal_separate_from_observed_premise_and_bounded():
    bits = [dict(paper_id="paper:alpha", title="Alpha", text="The evaluation excludes unseen games")]
    text = writer._future_directions_text(bits, [], False, set())
    units = sentence_units(text, {"paper:alpha"})
    assert units[0][1] == ["paper:alpha"]
    assert not units[1][1] and "proposes" in units[1][0]
    assert len(units) == 2


def test_rewritten_sentence_cannot_escape_source_contract_by_staling_manifest():
    src = source()
    row = dict(text="Old wording", cites=[src["paper_id"]], scope="method", source_bindings=[src])
    ev = Evidence(evidence_id=src["evidence_id"], paper_id=src["paper_id"], text=src["source_quote"], supports_claims=[src])
    class NLI:
        def best_match(self, claim, evidences):
            return NLIResult("entailment", "direct", .9)
    result = map_claims("t", "Rewritten sentence without a current binding [paper:alpha].",
                        EvidenceStore(task_id="t", evidence=[ev]), ParsedPapers(task_id="t", papers=[]),
                        NLI(), structured_claims=[row])
    assert result.entries[0].source_role_violation == "missing_source_binding"
    assert result.entries[0].status == "unsupported"


def test_validated_claim_units_reports_rejection_reasons():
    """Wave7 diagnostics: a full failure raises with the per-claim reasons, so
    a template fallback is attributable instead of silent."""
    from tools.write_survey import _validated_claim_units
    import pytest

    raw = [{"text": "A claim with an alias that does not exist", "scope": "method",
            "cite": ["P9"], "sources": ["P1S1"]}]
    sources = {"P1S1": {"paper_id": "paper:1", "claim_text": "x", "source_role": "own_method",
                        "source_quote": "we do", "evidence_ids": ["e1"], "support_type": "entailment"}}
    with pytest.raises(ValueError, match="unknown_cite_alias"):
        _validated_claim_units(raw, {"P1": "paper:1"}, {"paper:1"}, False, None, sources)


def test_alias_normalization_accepts_bracketed_and_lowercase_tags():
    """Wave8 ①: the prompt advertises bracketed tags, validation wants bare
    keys, so [p1]/p1/[P1S1] in the model's arrays must survive."""
    llm = ScriptedLLM([json.dumps([{**claim(), "cite": ["[p1]"], "sources": ["[p1s1]"]}])])
    collected = []
    paragraphs = writer._llm_section_body(section(), llm.chat, False, set(), collected,
                                         {"paper:alpha": "abstract_only", "paper:beta": "abstract_only"})
    assert paragraphs
    assert collected[0]["cites"] == ["paper:alpha"]
    assert collected[0]["source_bindings"][0]["evidence_id"] == "paper:alpha_abs_p0_0"


def test_alias_normalization_accepts_lowercase_embedded_tag():
    raw = {**claim(), "text": "Alpha [p1] learns policy and value predictions for tree search", "cite": ["p1"]}
    units = writer._validated_claim_units(
        [raw], {"P1": "paper:alpha"}, {"paper:alpha"}, False, None,
        {"P1S1": source()})
    assert units and "[paper:alpha]" in units[0]["sentence"]


def test_alias_failure_retries_once_and_persists_raw_reply(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "_WRITER_REJECTS_PATH", tmp_path / "rejects.jsonl")
    bad = json.dumps([{**claim(), "cite": ["P9"], "text": "Alpha was rated at grandmaster level"}])
    llm = ScriptedLLM([bad, json.dumps([claim()])])
    paragraphs = writer._llm_section_body(section(), llm.chat, False, set(), None,
                                         {"paper:alpha": "abstract_only", "paper:beta": "abstract_only"})
    assert paragraphs
    assert len(llm.prompts) == 2
    assert "bare tags" in llm.prompts[1]
    lines = (tmp_path / "rejects.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["attempt"] == 1 and "P9" in record["reply"] and record["rejects"]


def test_alias_double_failure_persists_both_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "_WRITER_REJECTS_PATH", tmp_path / "rejects.jsonl")
    bad = json.dumps([{**claim(), "cite": ["P9"]}])
    llm = ScriptedLLM([bad, bad])
    with pytest.raises(ValueError, match="unknown_cite_alias"):
        writer._llm_section_body(section(), llm.chat, False, set(), None,
                                 {"paper:alpha": "abstract_only", "paper:beta": "abstract_only"})
    lines = (tmp_path / "rejects.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["attempt"] for line in lines] == [1, 2]


def test_non_alias_failure_does_not_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "_WRITER_REJECTS_PATH", tmp_path / "rejects.jsonl")
    llm = ScriptedLLM([json.dumps([{**claim(), "scope": "background"}])])
    with pytest.raises(ValueError, match="source_role_out_of_scope"):
        writer._llm_section_body(section(), llm.chat, False, set(), None,
                                 {"paper:alpha": "abstract_only", "paper:beta": "abstract_only"})
    assert len(llm.prompts) == 1
    # Wave8b: every total-validation failure persists its raw reply, so a
    # role-scope storm is attributable offline (alias-only blindness bit wave7).
    lines = (tmp_path / "rejects.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1 and "source_role_out_of_scope" in lines[0]


def test_safe_section_goal_swaps_skeleton_placeholder():
    """Wave8 ②: the P2 seed-skeleton placeholder must never reach prose."""
    goal = writer._safe_section_goal({"section_title": "Internal Representation",
                                      "section_goal": "Merged from a seed survey taxonomy skeleton."})
    assert "merged from a seed survey" not in goal.casefold()
    assert "Internal Representation" in goal
