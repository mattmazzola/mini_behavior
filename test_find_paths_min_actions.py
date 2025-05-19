from .helpers import find_paths, get_total_actions


def test_find_paths_minimum_actions():
    """Test that find_paths returns paths with minimum number of actions."""
    # Grid with multiple paths, but one requires fewer rotations
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    start = (0, 0)
    end = (4, 4)

    # There are many paths from (0,0) to (4,4), but we expect only the ones with minimal actions
    paths = find_paths(grid, start, end, empty_cell_value=0)

    # Calculate the number of actions for each path
    action_counts = [get_total_actions(path) for path in paths]

    # All returned paths should have the same minimum number of actions
    min_actions = min(action_counts)
    for count in action_counts:
        assert count == min_actions

    # The straight paths (only right+down or down+right movements) should have fewest rotations
    # Two example optimal paths with minimum actions
    # right, right, right, right, down, down, down, down
    expected_path1 = [(0, 0), (1, 0), (2, 0), (3, 0),
                      (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
    # down, down, down, down, right, right, right, right
    expected_path2 = [(0, 0), (0, 1), (0, 2), (0, 3),
                      (0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]

    # For each of these optimal paths, there is at most one turn
    assert get_total_actions(expected_path1) == min_actions
    assert get_total_actions(expected_path2) == min_actions

    # The minimum actions should be 8 movements + 1 rotation = 9
    assert min_actions == 9


def test_find_paths_same_coords_different_rotations():
    """Test that paths with same coordinates but more rotations are excluded."""
    # Create a simple grid with two possible paths to the goal
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]

    start = (0, 0)
    end = (2, 2)

    # Expected paths:
    # Path 1 (minimal turns): (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2) = 4 moves + 1 turn = 5 actions
    # Path 2 (more turns): (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) = 4 moves + 1 turn = 5 actions
    # Path 3 (inefficient): (0,0) -> (1,0) -> (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) = many unnecessary moves and turns

    paths = find_paths(grid, start, end, empty_cell_value=0)

    # Both path 1 and path 2 should be included (they both have the same number of actions)
    assert len(paths) == 2

    # Verify that all paths have the minimum number of actions
    action_counts = [get_total_actions(path) for path in paths]
    assert all(count == action_counts[0] for count in action_counts)

    # Check that the paths have the expected number of actions (4 moves + 1 turn = 5)
    assert action_counts[0] == 5


def test_find_paths_optimizes_for_rotations():
    """Test that find_paths optimizes for paths with fewer rotations."""
    # Create a grid where there are multiple paths of the same length,
    # but some require more rotations than others
    grid = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    start = (0, 0)
    end = (3, 2)

    paths = find_paths(grid, start, end, empty_cell_value=0)

    # Calculate the number of actions for each path
    action_counts = [get_total_actions(path) for path in paths]

    # All returned paths should have the same minimum number of actions
    assert len(set(action_counts)) == 1

    # Verify that the optimal paths are returned
    # "Straight" paths should have the fewest rotations (just one turn):
    # (0,0) -> (1,0) -> (2,0) -> (3,0) -> (3,1) -> (3,2) = 5 moves + 1 turn = 6 actions
    # (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (3,2) = 5 moves + 1 turn = 6 actions
    # While zigzagging would have more turns.

    expected_optimal_actions = 6
    assert action_counts[0] == expected_optimal_actions


def test_find_paths_with_initial_rotation():
    """Test that find_paths considers initial rotation when selecting optimal paths."""
    grid = [
        [0, 0],
        [0, 0],
    ]

    start = (0, 0)
    end = (1, 1)

    paths_no_direction = find_paths(grid, start, end, empty_cell_value=0)
    assert len(
        paths_no_direction) == 2, "Two paths should be found without agent direction"

    paths_agent_facing_right = find_paths(
        grid, start, end, empty_cell_value=0, agent_dx_dy=[1, 0])
    assert len(
        paths_agent_facing_right) == 1, "Only one path should be found with agent facing right"
