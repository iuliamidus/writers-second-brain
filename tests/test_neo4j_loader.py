"""Guards around the parts of the loader that build Cypher by string.

Labels and relationship types cannot be parameterised in Cypher, so they get
interpolated. These tests exist to make sure that stays safe once the LLM
enrichment stage starts inventing relationship names.
"""

import pytest

from wsb.graph.neo4j_loader import NODE_KEYS, _check_identifier, _label_and_key


@pytest.mark.parametrize("rel", ["MENTIONS", "PUBLISHED_ON", "SIMILAR_TO", "Continues", "R2"])
def test_accepts_plausible_relationship_types(rel):
    assert _check_identifier(rel) == rel


@pytest.mark.parametrize(
    "rel",
    [
        "]-() MATCH (n) DETACH DELETE n //",  # the attack this is here to stop
        "rel-type",                            # hyphen closes the pattern
        "has space",
        "2FAST",                               # cannot start with a digit
        "",
        "x" * 65,                              # unbounded length
    ],
)
def test_rejects_anything_that_could_break_out(rel):
    with pytest.raises(ValueError):
        _check_identifier(rel)


def test_every_node_type_maps_to_a_constrained_key():
    for node_type in NODE_KEYS:
        label, key = _label_and_key(node_type)
        assert label == node_type
        assert _check_identifier(key)


def test_unknown_node_type_fails_loudly():
    # Better to stop the push than to silently drop edges for a node type
    # someone added to the edge builders but forgot to declare here.
    with pytest.raises(ValueError, match="Unknown node type"):
        _label_and_key("Theme")
