import pytest
from pydantic import ValidationError
from tools.models.artifacts import PaperCard, Claim, Evidence
from tools.models.common import evidence_id


def test_paper_card_defaults():
    c = PaperCard(paper_id="paper:x", title="T")
    assert c.possible_claims == {}
    assert c.card_type == "deep"


def test_claim_with_empty_evidence_ok():
    cl = Claim(text="x", dimension="limitations", evidence_ids=[])
    assert cl.evidence_ids == []


def test_card_rejects_missing_required():
    with pytest.raises(ValidationError):
        PaperCard(title="no id")  # type: ignore


def test_evidence_id_roundtrip():
    eid = evidence_id("paper:x", 5, 2)
    e = Evidence(evidence_id=eid, paper_id="paper:x", source_page=5, source_paragraph_index=2, text="t")
    assert e.evidence_id == "paper:x_p5_2"
