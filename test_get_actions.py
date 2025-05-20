import random
import pytest

from .helpers import (
    get_pairs,
    get_direction,
    get_directions,
    get_actions,
    get_rotations_between_directions,
    get_total_actions,
    dx_dy_to_direction,
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
    assert get_directions(coords, include_rotations=False) == ["right", "down", "left"]


def test_get_directions_with_rotations():
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert get_directions(coords, include_rotations=True) == ["right", "CW", "down", "CW", "left"]


@pytest.mark.parametrize(
    "prev_dir, next_dir, expected",
    [
        ("up", "up", []),
        ("up", "right", ["CW"]),
        ("up", "down", ["CW", "CW"]),  # 180° rotation requires two 90° rotations
        ("up", "left", ["CCW"]),
        ("right", "down", ["CW"]),
        ("right", "left", ["CW", "CW"]),  # 180° rotation
        ("down", "up", ["CW", "CW"]),  # 180° rotation
    ],
)
def test_get_rotations_between_directions(prev_dir, next_dir, expected):
    assert get_rotations_between_directions(prev_dir, next_dir) == expected


def test_get_actions_deterministic(monkeypatch):
    # Force random.choice to pick the first verb for reproducibility
    monkeypatch.setattr(random, "choice", lambda verbs: verbs[0])
    directions = ["up", "left", "down"]
    assert get_actions(directions, movement_verbs=["go", "move"]) == [
        "go up",
        "go left",
        "go down",
    ]


def test_get_actions_with_rotations(monkeypatch):
    # Force random.choice to pick the first verb for reproducibility
    monkeypatch.setattr(random, "choice", lambda verbs: verbs[0])
    directions = ["up", "CW", "right", "CCW", "up"]
    assert get_actions(
        directions, 
        movement_verbs=["go", "move"], 
        rotation_verbs=["turn", "rotate"]
    ) == [
        "go up",
        "turn right",  # CW becomes "turn right"
        "go right",
        "turn left",   # CCW becomes "turn left"
        "go up",
    ]


def test_get_actions_with_180_rotation(monkeypatch):
    # Force random.choice to pick the first verb for reproducibility
    monkeypatch.setattr(random, "choice", lambda verbs: verbs[0])
    # Two CW rotations represent a 180-degree turn
    directions = ["up", "CW", "CW", "down"]
    assert get_actions(
        directions, 
        movement_verbs=["go", "move"], 
        rotation_verbs=["turn", "rotate"]
    ) == [
        "go up",
        "turn right",  # First CW becomes "turn right"
        "turn right",  # Second CW becomes "turn right" again
        "go down",
    ]


def test_get_total_actions():
    # Straight path with no turns
    path1 = [(0, 0), (1, 0), (2, 0)]  # right, right = 2 movements, 0 rotations
    assert get_total_actions(path1) == 2
    
    # Path with one 90-degree turn
    path2 = [(0, 0), (1, 0), (1, 1)]  # right, down = 2 movements, 1 rotation
    assert get_total_actions(path2) == 3
    
    # Path with two 90-degree turns
    path3 = [(0, 0), (1, 0), (1, 1), (0, 1)]  # right, down, left = 3 movements, 2 rotations
    assert get_total_actions(path3) == 5
    
    # Path with a 180-degree turn (which is 2 rotations)
    path4 = [(0, 0), (1, 0), (0, 0)]  # right, left = 2 movements, 2 rotations (180-degree turn)
    assert get_total_actions(path4) == 4
    
    # Single point (no actions)
    path5 = [(0, 0)]
    assert get_total_actions(path5) == 0
    
    # Empty path
    path6 = []
    assert get_total_actions(path6) == 0


def test_dx_dy_to_direction():
    assert dx_dy_to_direction([1, 0]) == "right"
    assert dx_dy_to_direction([0, 1]) == "down"
    assert dx_dy_to_direction([-1, 0]) == "left"
    assert dx_dy_to_direction([0, -1]) == "up"
    with pytest.raises(ValueError):
        dx_dy_to_direction([1, 1])


def test_get_directions_with_initial_rotation():
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    
    # Agent initially facing up, needs to turn right to face right
    assert get_directions(coords, initial_direction=[0, -1]) == ["CW", "right", "CW", "down", "CW", "left"]
    
    # Agent initially facing right, no initial rotation needed
    assert get_directions(coords, initial_direction=[1, 0]) == ["right", "CW", "down", "CW", "left"]
    
    # Agent initially facing down, needs to turn left to face right
    assert get_directions(coords, initial_direction=[0, 1]) == ["CCW", "right", "CW", "down", "CW", "left"]
    
    # Agent initially facing left, needs to turn 180 degrees (two CWs) to face right
    assert get_directions(coords, initial_direction=[-1, 0]) == ["CW", "CW", "right", "CW", "down", "CW", "left"]
