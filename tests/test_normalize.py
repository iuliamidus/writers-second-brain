"""Tests for the parts most likely to silently corrupt the archive."""

from wsb.normalize import fold
from wsb.graph.edges import _series_key


def test_fold_collapses_romanian_spellings():
    # Comma-below (correct), cedilla (legacy) and bare ASCII must all match.
    assert fold("cărți") == fold("carti")
    assert fold("şi") == fold("și") == "si"
    assert fold("Ţară") == fold("țara") == "tara"


def test_fold_handles_other_diacritics():
    assert fold("Reflexii și reflecții") == "reflexii si reflectii"
    assert fold("Norvegia – o călătorie") == "norvegia – o calatorie"


def test_series_detection():
    assert _series_key("Norway – a Solitary Road Trip – 1") == ("norway – a solitary road trip", 1)
    assert _series_key("Norway – a Solitary Road Trip – 2") == ("norway – a solitary road trip", 2)
    assert _series_key("Comfortably Numb in Hotel California (18)")[1] == 18


def test_series_ignores_short_stems():
    # "5%" or "Life 2" are not series, they just end in a digit.
    assert _series_key("Life 2") is None
