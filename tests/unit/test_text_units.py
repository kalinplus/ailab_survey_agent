"""Cross-module consistency: claim_mapper and evaluate_survey must derive
their claims from the same sentence-unit kernel (tools/verify/text_units.py,
specs/claim切分归一统一内核.md acceptance 1).

The fixture packs every known divergence: heading lines, an image embed whose
caption holds a bracket, a quoted line, a table block, a math interval,
caption-like fake brackets, a trailing citation-only fragment, a
multi-citation sentence and a seed-style id.
"""

import tools.evaluate_survey as evaluate_survey
from tools.verify import claim_mapper
from tools.verify.text_units import sentence_units

FIXTURE_MD = """# Survey title heading

## Section heading

![Figure caption bracket [not a citation]](artifact_1)

> Quoted line citing [paper:q].

| Method | Year |
|---|---|

The reward lies in [0,1] across domains [paper:a]. Caption-like bracket [Future Matrix] follows. Models drop the tag. [paper:b].
Two refs in one sentence [paper:a] and [paper:b].
Seed style claim [seed_2018].
"""

KNOWN_IDS = {"paper:a", "paper:b", "seed_2018"}

EXPECTED_UNITS = [
    ("The reward lies in  across domains", ["paper:a"]),
    ("Caption-like bracket  follows", []),
    ("Models drop the tag", ["paper:b"]),          # trailing fragment merged back
    ("Two refs in one sentence  and", ["paper:a", "paper:b"]),
    ("Seed style claim", ["seed_2018"]),
]


def test_evaluator_units_come_from_shared_kernel():
    assert evaluate_survey._sentence_units(FIXTURE_MD, KNOWN_IDS) == EXPECTED_UNITS
    assert sentence_units(FIXTURE_MD, KNOWN_IDS) == EXPECTED_UNITS


def test_generation_claims_align_unit_by_unit():
    """claim_mapper keeps sentence granularity (first citation) but every
    claim must be the corresponding kernel unit with the same first id."""
    claims = claim_mapper.extract_claims_with_citations(FIXTURE_MD, KNOWN_IDS)
    assert claims == [(text, cites[0]) for text, cites in EXPECTED_UNITS if cites]


def test_evaluator_pairs_align_unit_by_unit():
    """evaluate_survey expands to (claim, citation) pairs — same units, same
    id order, every citation instead of only the first."""
    pairs = evaluate_survey.extract_claim_pairs(FIXTURE_MD, KNOWN_IDS)
    assert pairs == [(text, c) for text, cites in EXPECTED_UNITS for c in cites]


def test_sentence_counts_match_across_granularities():
    units = sentence_units(FIXTURE_MD, KNOWN_IDS)
    cited = [c for _, c in units if c]
    assert len(claim_mapper.extract_claims_with_citations(FIXTURE_MD, KNOWN_IDS)) == len(cited)
    assert len(evaluate_survey.extract_claim_pairs(FIXTURE_MD, KNOWN_IDS)) \
        == sum(len(c) for c in cited)


def test_fake_citations_bound_nowhere():
    """Math interval, caption brackets and the quote-line id never become a
    bound id on either side (paper:q is prefix-valid but its line is a quote,
    so structural exclusion drops it before validation)."""
    generation_ids = {pid for _, pid
                      in claim_mapper.extract_claims_with_citations(FIXTURE_MD, KNOWN_IDS)}
    pair_ids = {pid for _, pid in evaluate_survey.extract_claim_pairs(FIXTURE_MD, KNOWN_IDS)}
    assert generation_ids == pair_ids == {"paper:a", "paper:b", "seed_2018"}


def test_known_ids_none_agrees_on_both_sides():
    """known_ids=None means prefix-rule only: paper: ids bind everywhere and
    the seed-style id binds nowhere — identically on both sides."""
    units = sentence_units(FIXTURE_MD, None)
    expected_claims = [(text, cites[0]) for text, cites in units if cites]
    expected_pairs = [(text, c) for text, cites in units for c in cites]
    assert claim_mapper.extract_claims_with_citations(FIXTURE_MD, None) == expected_claims
    assert evaluate_survey.extract_claim_pairs(FIXTURE_MD, None) == expected_pairs
    assert ("Seed style claim", "seed_2018") not in expected_claims
