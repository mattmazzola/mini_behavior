import pytest

from .helpers import find_paths, get_total_actions

@pytest.mark.parametrize(
    "grid,start,end,expected_paths",
    [
        # 1. Two equally short paths in a 2×2 open grid
        (
            [[0, 0], [0, 0]],
            (0, 0),
            (1, 1),
            [
                [(0, 0), (1, 0), (1, 1)],
                [(0, 0), (0, 1), (1, 1)],
            ],
        ),
        # 2. Only one valid shortest path because of obstacles
        (
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            (0, 0),
            (2, 0),
            [
                [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)],
            ],
        ),
    ],
)
def test_find_paths_success(grid, start, end, expected_paths):
    """Paths found match the expected shortest paths (order-insensitive)."""
    paths = find_paths(grid, start, end, empty_cell_value=0)

    # normalise order for comparison
    sorted_returned = sorted(paths)
    sorted_expected = sorted(expected_paths)
    assert sorted_returned == sorted_expected


def test_find_paths_unreachable():
    """When no path exists an empty list is returned."""
    grid = [
        [0, 1],
        [1, 0],
    ]
    paths = find_paths(grid, (0, 0), (1, 1), empty_cell_value=0)
    assert paths == []


def test_all_returned_paths_are_shortest():
    """Every returned path must have the same minimal length."""
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    start, end = (0, 0), (2, 2)
    paths = find_paths(grid, start, end, empty_cell_value=0)
    assert paths, "At least one path should be found"

    lengths = {len(p) for p in paths}
    assert len(lengths) == 1, "All paths must share the same minimal length"


def test_large_grid_handles_memory():
    """A larger 8×8 grid with paths optimized for minimum actions."""
    grid = [
        [0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0],
    ]
    start, end = (0, 0), (7, 7)
    possible_paths = [
        [
            (0, 0), (0, 1),
            (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
            (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, 7),
        ],
        [
            (0, 0), (1, 0),
            (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
            (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, 7),
        ]
    ]

    paths = find_paths(grid, start, end, empty_cell_value=0)
    
    assert len(paths) >= 1, "At least one path should be found"
    
    for path in paths:
        assert path in possible_paths, f"Unexpected path {path}"
    
    # Verify all paths have the same number of actions
    action_counts = [get_total_actions(path) for path in paths]
    assert len(set(action_counts)) == 1, "All paths should have the same number of actions"


def test_agent_case_one():
    grid = [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 2, 2, 2, 0, 1],
        [1, 0, 0, 2, 2, 2, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]
    start, end = (2, 5), (3, 6)
    possible_paths = [
        [(2, 5), (3, 5), (3, 6)],
        [(2, 5), (2, 6), (3, 6)],
    ]

    paths = find_paths(grid, start, end, empty_cell_value=0)
    
    assert paths, "At least one path should be found"
    
    for path in paths:
        assert path in possible_paths, f"Unexpected path {path}"
    
    # Verify all paths have the same number of actions
    action_counts = [get_total_actions(path) for path in paths]
    assert len(set(action_counts)) == 1, "All paths should have the same number of actions"
