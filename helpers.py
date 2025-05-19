from enum import IntEnum, auto

import numpy as np


import random
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
from PIL import Image

T = TypeVar('T')


class CellType(IntEnum):
    Empty = 0
    Wall = auto()
    Table = auto()
    Printer = auto()
    Agent = auto()


def get_flattened_grid_by_composition(
    width: int,
    height: int,
    agent_col_row: list[int],
    agent_dcol_drow: list[int],
    printer_col_row: list[int],
    table_col_row: list[int],
    table_width_height: list[int],
) -> np.ndarray:
    flat_grid = np.zeros((height, width), dtype=int)

    # Set all sides to be walls
    flat_grid[:, 0] = CellType.Wall
    flat_grid[:, -1] = CellType.Wall
    flat_grid[0, :] = CellType.Wall
    flat_grid[-1, :] = CellType.Wall

    # Set the table area
    table_col, table_row = table_col_row
    table_width, table_height = table_width_height
    flat_grid[table_row:table_row + table_height,
              table_col:table_col + table_width] = CellType.Table

    # Set the printer area
    printer_col, printer_row = printer_col_row
    flat_grid[printer_row, printer_col] = CellType.Printer

    # Set the agent position
    agent_col, agent_row = agent_col_row
    flat_grid[agent_row, agent_col] = CellType.Agent

    return flat_grid


def get_total_actions(path: List[Tuple[int, int]]) -> int:
    """
    Calculate the total number of actions (movements + rotations) needed for a path.

    Args:
        path: A list of coordinates representing a path

    Returns:
        int: The total number of actions (movements + rotations)
    """
    if len(path) <= 1:
        return 0

    # Get the basic directions without incorporating rotations yet
    coord_pairs = get_pairs(path)
    basic_directions = [get_direction(pair[0], pair[1])
                        for pair in coord_pairs]

    # Count movements (always equal to len(basic_directions))
    movement_count = len(basic_directions)

    # Count rotations
    rotation_count = 0
    current_dir = basic_directions[0]

    for next_dir in basic_directions[1:]:
        rotations = get_rotations_between_directions(current_dir, next_dir)
        rotation_count += len(rotations)
        current_dir = next_dir

    return movement_count + rotation_count


def find_paths(
    grid: list[list[int]],
    start: tuple[int, int],
    end: tuple[int, int],
    empty_cell_value: int,
    initial_direction: List[int] = None,
):
    """
    Find all **shortest** paths from start to end in a grid, avoiding cells with a specific value.
    The paths are optimized for the minimum number of actions (movements + rotations), including initial rotations.

    Args:
        grid (list[list[int]]): The grid to search, given as list of rows, where each row is a list of integers.
        start (tuple[int, int]): The starting coordinates (col, row).
        end (tuple[int, int]): The ending coordinates (col, row).
        empty_cell_value (int): The value of cells that can be traversed.
        initial_direction (List[int], optional): The initial direction vector [dx, dy]. If provided, 
                                          considers initial rotations in path optimization.

    Returns:
        list[list[tuple[int, int]]]: A list of paths with minimum number of actions, where each path is a list of coordinates.
    """

    def is_valid_move(row: int, col: int) -> bool:
        return 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] == empty_cell_value

    def dfs(row: int, col: int, path: list[tuple[int, int]]):
        nonlocal shortest_path_length
        # If path is longer than the shortest path, abandon this path
        if shortest_path_length is not None and len(path) > shortest_path_length:
            return

        # If reached the desired end point
        if (col, row) == (e_col, e_row):
            # If the path is shorter than the shortest paths, clear the current list
            if shortest_path_length is None or len(path) < shortest_path_length:
                shortest_path_length = len(path)
                row_col_paths.clear()

            # If the path is equal to the shortest paths, add it to the list
            if len(path) == shortest_path_length:
                row_col_paths.append(path.copy())

            return

        visited.add((row, col))

        # Explore all four possible directions
        for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n_row, n_col = row + d_row, col + d_col
            if is_valid_move(n_row, n_col) and (n_row, n_col) not in visited:
                dfs(n_row, n_col, path + [(n_row, n_col)])

        visited.remove((row, col))

    row_col_paths = []
    visited = set()
    shortest_path_length: int | None = None

    s_col, s_row = start
    e_col, e_row = end
    dfs(s_row, s_col, [(s_row, s_col)])

    # Switch (row, col) to (col, row)
    col_row_paths = [[(col, row) for row, col in path]
                     for path in row_col_paths]

    # Convert to int
    col_row_int_paths = [[(int(col), int(row))
                          for col, row in path] for path in col_row_paths]

    # Filter paths based on minimum number of actions (movements + rotations)
    if col_row_int_paths:
        # If initial direction is provided, include initial rotation cost in optimization
        if initial_direction is not None:
            try:
                # Calculate the total actions including initial rotations for each path
                action_counts = []
                # Track initial rotations separately for better optimization
                initial_rotation_counts = []

                for path in col_row_int_paths:
                    # Basic path actions without initial rotation
                    path_actions = get_total_actions(path)

                    # Count initial rotations if path has at least two points
                    initial_rotations_count = 0
                    if len(path) >= 2:
                        # Get the first direction in the path
                        first_direction = get_direction(path[0], path[1])
                        # Get starting direction
                        start_direction = dx_dy_to_direction(initial_direction)
                        # Get the rotations needed
                        initial_rotations = get_rotations_between_directions(
                            start_direction, first_direction)
                        initial_rotations_count = len(initial_rotations)
                        # Add rotation cost
                        path_actions += initial_rotations_count

                    action_counts.append(path_actions)
                    initial_rotation_counts.append(initial_rotations_count)

                # First priority: minimize total actions
                min_actions = min(action_counts)
                # Get all paths with minimum total actions
                min_action_indices = [i for i, count in enumerate(
                    action_counts) if count == min_actions]

                # Second priority: among paths with minimum actions, prefer those with fewer initial rotations
                min_initial_rotations = min(
                    initial_rotation_counts[i] for i in min_action_indices)
                min_rotation_indices = [
                    i for i in min_action_indices if initial_rotation_counts[i] == min_initial_rotations]

                # Return optimized paths
                optimized_paths = [col_row_int_paths[i]
                                   for i in min_rotation_indices]
                return optimized_paths

            except ValueError:
                # Fall back to regular path optimization if there's an issue with direction vectors
                action_counts = [get_total_actions(
                    path) for path in col_row_int_paths]
        else:
            # Regular path optimization without considering initial rotation
            action_counts = [get_total_actions(
                path) for path in col_row_int_paths]

        min_actions = min(action_counts)
        optimized_paths = [path for path, count in zip(
            col_row_int_paths, action_counts) if count == min_actions]
        return optimized_paths

    return col_row_int_paths


def get_direction(
    coord1: Tuple[int, int],
    coord2: Tuple[int, int],
    use_cardinal: bool = False,
) -> Optional[str]:
    col1, row1 = coord1
    col2, row2 = coord2

    # If any coordinate is negative, raise a ValueError
    if row1 < 0 or col1 < 0 or row2 < 0 or col2 < 0:
        raise ValueError(
            f"You provided negative values for coordinates: {coord1}, {coord2}. All coordinates must be positive values.")

    drow = row2 - row1
    dcol = col2 - col1

    if use_cardinal:
        up, down, left, right = 'north', 'south', 'west', 'east'
    else:
        up, down, left, right = 'up', 'down', 'left', 'right'

    match (drow, dcol):
        case (0, 0):
            return None
        case (-1, 0):
            return up
        case (1, 0):
            return down
        case (0, -1):
            return left
        case (0, 1):
            return right
        case _:
            raise ValueError(
                f"Invalid coordinates provided! {coord1} and {coord2} are not adjacent or are diagonal. Distances: {drow}, {dcol}")


def get_pairs(items: Sequence[T]) -> List[List[T]]:
    """
    Given a sequence of items, return pairs of sequential items.

    Args:
        items (Sequence[T]): A sequence of items of type T

    Returns:
        List[List[T]]: A list of pairs, where each pair consists of consecutive items from the input

    Raises:
        ValueError: If the input sequence has fewer than 2 items

    Example:
        >>> get_pairs([1, 2, 3, 4])
        [[1, 2], [2, 3], [3, 4]]
    """
    if len(items) < 2:
        raise ValueError("Input sequence must have at least 2 items")

    return [[items[i], items[i+1]] for i in range(len(items) - 1)]


def get_rotations_between_directions(prev_dir: str, next_dir: str) -> List[str]:
    """
    Determine the rotation(s) needed to go from one direction to another.

    Args:
        prev_dir: The previous direction ("up", "right", "down", "left")
        next_dir: The next direction ("up", "right", "down", "left")

    Returns:
        List of rotation directions ("CW", "CCW") or empty list if no rotation needed
    """
    dir_to_idx = {"up": 0, "right": 1, "down": 2, "left": 3}

    if prev_dir == next_dir:
        return []

    prev_idx = dir_to_idx[prev_dir]
    next_idx = dir_to_idx[next_dir]

    # Calculate the shortest rotation
    diff = (next_idx - prev_idx) % 4

    if diff == 1:  # Clockwise 90 degrees
        return ["CW"]
    elif diff == 3:  # Counter-clockwise 90 degrees
        return ["CCW"]
    elif diff == 2:  # 180 degrees - requires two 90-degree rotations
        # Use two CW rotations for 180-degree rotations
        return ["CW", "CW"]

    return []


def dx_dy_to_direction(dx_dy: List[int]) -> str:
    """
    Convert an agent's direction vector to a cardinal direction string.

    Args:
        dx_dy: The direction vector [dx, dy] where dx and dy are either 0, 1, or -1

    Returns:
        A direction string ('up', 'down', 'left', 'right')
    """
    dx, dy = dx_dy

    if dx == 1 and dy == 0:  # Facing right
        return 'right'
    elif dx == 0 and dy == 1:  # Facing down
        return 'down'
    elif dx == -1 and dy == 0:  # Facing left
        return 'left'
    elif dx == 0 and dy == -1:  # Facing up
        return 'up'
    else:
        raise ValueError(
            f"Invalid direction vector: {dx_dy}. Expected [dx, dy] where each is 0, 1, or -1.")


def get_directions(
    coordinates: List[Union[Tuple[int, int], List[int]]],
    use_cardinal: bool = False,
    include_rotations: bool = True,
    initial_direction: List[int] = None,
) -> List[str]:
    """
    Takes a list of coordinates and returns a list of directions between adjacent coordinates.
    If include_rotations is True, it will include CW or CCW rotation directions when needed.
    If initial_direction is provided, it will include initial rotation to align with the first direction.

    Args:
        coordinates: A list of coordinate tuples [(row1, col1), (row2, col2), ...]
        use_cardinal: If True, returns cardinal directions
        include_rotations: If True, includes rotation directions (CW, CCW)
        initial_direction: The initial direction vector [dx, dy], if available

    Returns:
        A list of direction strings
        Spatial ["up", "down", "left", "right"] and potentially ["CW", "CCW"]
        Cardinal ["north", "south", "east", "west"] and potentially ["CW", "CCW"]
    """
    # Get pairs of adjacent coordinates
    coord_pairs = get_pairs(coordinates)

    # Determine direction for each pair
    basic_directions = [get_direction(pair[0], pair[1], use_cardinal)
                        for pair in coord_pairs]

    if not include_rotations or len(basic_directions) <= 1:
        return basic_directions

    # Include rotations between directions
    final_directions = []

    # Check if we need an initial rotation based on starting direction
    if initial_direction is not None and len(basic_directions) > 0:
        try:
            initial_dir = dx_dy_to_direction(initial_direction)
            first_move_dir = basic_directions[0]

            # Add initial rotation(s) if needed
            initial_rotations = get_rotations_between_directions(
                initial_dir, first_move_dir)
            final_directions.extend(initial_rotations)
        except ValueError:
            # If there's an issue with the direction vector, just ignore it
            pass

    # Add the first direction
    current_dir = basic_directions[0]
    final_directions.append(current_dir)

    # Add remaining directions with rotations as needed
    for next_dir in basic_directions[1:]:
        rotations = get_rotations_between_directions(current_dir, next_dir)
        final_directions.extend(rotations)
        final_directions.append(next_dir)
        current_dir = next_dir

    return final_directions


def get_actions(
    directions: list[str],
    movement_verbs: list[str] = ["go", "move"],
    rotation_verbs: list[str] = ["turn", "rotate"],
) -> list[str]:
    """
    Generate action strings by prefixing each direction with an appropriate verb.
    For movement directions (up, down, left, right), use movement verbs.
    For rotation directions (CW, CCW), use rotation verbs and convert to right/left.

    Args:
        directions: List of direction strings (e.g., "up", "down", "left", "right", "CW", "CCW")
        movement_verbs: List of verb strings to use for movement directions (default: ["go", "move"])
        rotation_verbs: List of verb strings to use for rotation directions (default: ["turn", "rotate"])

    Returns:
        List of action strings with appropriate format based on direction type
    """
    actions = []
    for direction in directions:
        if direction in ["up", "down", "left", "right", "north", "south", "east", "west"]:
            # Regular movement direction
            verb = random.choice(movement_verbs)
            actions.append(f"{verb} {direction}")
        elif direction == "CW":
            # Clockwise rotation - use "right"
            verb = random.choice(rotation_verbs)
            actions.append(f"{verb} right")
        elif direction == "CCW":
            # Counter-clockwise rotation - use "left"
            verb = random.choice(rotation_verbs)
            actions.append(f"{verb} left")

    return actions


def get_actions_prompt(
    actions: list[str],
):

    # Construct prompts using a template
    prompt_template = """
Task: Maze Navigation Simulation

Determine the final destination coordinates (x,y) in pixels from the starting point (green circle) following the given action sequence.

Definitions of the actions are verb direction instructions, such as "Go up", or "Move right".
Possible directions are up/left/down/right
This means move one grid space in the absolute up/left/down/right direction.

The final destination should be provided as [x, y] pixels from the top left corner of the image.

Full Action Sequence: {actions}

Initial maze: <image>

Response:
""".strip()

    # Construct the prompt
    prompt = prompt_template.format(
        actions=", ".join(actions),
    )

    return prompt


def get_destination_prompt(coordinate: Tuple[int, int]) -> str:
    # Construct prompts using a template
    prompt_template = """
Task: Maze Navigation Simulation

You are provided a maze.
Wall are shown as black lines
The start point is shown as a green circle
The end point is shown as a red cross

You CANNOT move between cells which would cross walls (black lines).

Definitions of the actions are verb direction instructions, such as "Go up", or "Move right".
Possible directions are up/left/down/right
This means move one grid space in the absolute up/left/down/right direction.

The final state is given as [x, y] pixels from the top left corner of the image.

Example:
* Final state: [104, 75]

Describe the initial state of the maze from the image provided.
Respond with the sequence of actions that should taken to move from the starting point (green circle) do the final state (red cross).
Remember DO NOT cross walls (black lines).

Final State: {final_state}

Initial maze: <image>

Response:
""".strip()

    # Construct the prompt
    prompt = prompt_template.format(
        final_state=str(coordinate),
    )

    return prompt


def get_prompt(data_item: dict, prompt_type: Literal["actions", "final_state"]) -> str:
    """
    Generate a prompt based on the data item and prompt type.

    Args:
        data_item: A single item from the dataset
        prompt_type: The information provided in the prompt ("actions" or "final_state")

    Returns:
        str: Generated prompt

    Raises:
        ValueError: If the prompt_type is not recognized or if the data_item doesn't have the required fields
    """

    match prompt_type:
        case "actions":
            return get_actions_prompt(data_item["actions"])
        case "final_state":
            return get_destination_prompt(data_item["end_pos"]["x_y"])
        case _:
            raise ValueError(f"Unrecognized prompt type: {prompt_type}")
