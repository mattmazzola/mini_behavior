import numpy as np
import pytest

from .helpers import get_flattened_grid_by_composition


def test_get_flattened_grid_by_composition():
    width = 8
    height = 8
    agent_col_row = [2, 5]
    agent_dcol_drow = [0, 0]
    printer_col_row = [3, 6]
    table_col_row = [3, 3]
    table_width_height = [3, 2]
    expected_grid = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 2, 2, 2, 0, 1],
        [1, 0, 0, 2, 2, 2, 0, 1],
        [1, 0, 4, 0, 0, 0, 0, 1],
        [1, 0, 0, 3, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ])

    actual_grid = get_flattened_grid_by_composition(
        width,
        height,
        agent_col_row,
        agent_dcol_drow,
        printer_col_row,
        table_col_row,
        table_width_height,
    )
    np.testing.assert_array_equal(actual_grid, expected_grid)
