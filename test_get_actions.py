import random
import pytest

from .helpers import (
    get_pairs,
    get_direction,
    get_directions,
    get_actions,
)


def test_get_pairs_ok():
    assert get_pairs([1, 2, 3, 4]) == [[1, 2], [2, 3], [3, 4]]


def test_get_pairs_too_short():
    with pytest.raises(ValueError):
        get_pairs([1])


@pytest.mark.parametrize(
    "coord1, coord2, expected",
    [
        ((0, 0), (1, 0), "right"),
        ((1, 0), (1, 1), "down"),
        ((1, 1), (0, 1), "left"),
        ((0, 1), (0, 0), "up"),
    ],
)
def test_get_direction_spatial(coord1, coord2, expected):
    assert get_direction(coord1, coord2) == expected


def test_get_direction_cardinal():
    assert get_direction((0, 0), (0, 1), use_cardinal=True) == "south"


def test_get_directions():
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert get_directions(coords) == ["right", "down", "left"]


def test_get_actions_deterministic(monkeypatch):
    # Force random.choice to pick the first verb for reproducibility
    monkeypatch.setattr(random, "choice", lambda verbs: verbs[0])
    directions = ["up", "left", "down"]
    assert get_actions(directions, verbs=["go", "move"]) == [
        "go up",
        "go left",
        "go down",
    ]
