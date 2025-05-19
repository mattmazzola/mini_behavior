import argparse
from datetime import datetime
import json
from pathlib import Path
from gymnasium import Env
from minigrid.wrappers import *
from helpers import CellType, find_paths, get_actions, get_directions, get_flattened_grid_by_composition
from mini_behavior.window import Window
from mini_behavior.utils.save import save_demo
from mini_behavior.grid import GridDimension
import numpy as np
import cv2

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32
show_furniture = False
env: Env
window: Window
maze_index = 0


def redraw(img):
    if not args.agent_view:
        env.set_render_mode('rgb_array')
        img = env.render()

    window.no_closeup()
    window.set_inventory(env)
    window.show_img(img)


def render_furniture():
    global show_furniture
    show_furniture = not show_furniture

    if show_furniture:
        img = np.copy(env.furniture_view)

        # i, j = env.agent.cur_pos
        i, j = env.agent_pos
        ymin = j * TILE_PIXELS
        ymax = (j + 1) * TILE_PIXELS
        xmin = i * TILE_PIXELS
        xmax = (i + 1) * TILE_PIXELS

        img[ymin:ymax, xmin:xmax, :] = GridDimension.render_agent(
            img[ymin:ymax, xmin:xmax, :], env.agent_dir)
        img = env.render_furniture_states(img)

        window.show_img(img)
    else:
        obs = env.gen_obs()
        redraw(obs)


def show_states():
    imgs = env.render_states()
    window.show_closeup(imgs)


def reset():
    if args.seed != -1:
        env.seed(args.seed)

    global maze_index
    maze_index += 1

    obs = env.reset()

    if hasattr(env, 'mission'):
        window.set_caption(env.mission)

    image, _ = obs
    redraw(image)


def load():
    if args.seed != -1:
        env.seed(args.seed)

    env.reset()
    obs = env.load_state(args.load)

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)


def step(action):
    prev_obs = env.gen_obs()
    obs, reward, done, truncated, info = env.step(action)

    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if args.save:
        all_steps[env.step_count] = (prev_obs, action)

    if done:
        print('done!')
        if args.save:
            save_demo(all_steps, args.env, env.episode)
        reset()
    else:
        redraw(obs)


def switch_dim(dim):
    env.switch_dim(dim)
    print(f'switching to dim: {env.render_dim}')
    obs = env.gen_obs()
    redraw(obs)


def get_agent_position():
    install_a_printer_env = env.env.env
    agent_col_row = install_a_printer_env.agent_pos.tolist()
    agent_dx_dy = install_a_printer_env.dir_vec.tolist()
    agent_next_col_row = install_a_printer_env.front_pos.tolist()

    return agent_col_row, agent_dx_dy, agent_next_col_row


def get_printer_position():
    install_a_printer_env = env.env.env
    printer_col_row = np.array(
        install_a_printer_env.objs['printer'][0].cur_pos).tolist()

    return printer_col_row


def get_table_colrow_widthheight():
    install_a_printer_env = env.env.env
    table_obj = install_a_printer_env.objs['table'][0]
    table_col_row = np.array(table_obj.cur_pos).tolist()
    table_width_height = [table_obj.width, table_obj.height]

    return table_col_row, table_width_height


def save_image(
    save_grid: bool = True,
    save_window: bool = False,
):
    def get_grid_image_without_highlight():
        env.set_render_mode('rgb_array')

        install_a_printer_env = env.env.env
        original_highlight = install_a_printer_env.highlight
        install_a_printer_env.highlight = False
        img = env.render()
        install_a_printer_env.highlight = original_highlight

        return img

    # Get path to save the image
    images_path = dataset_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    # Save the grid image without agent highlight
    grid_image_path = images_path / f"maze_{maze_index}_grid.png"
    window_image_path = images_path / f"maze_{maze_index}_window.png"

    if save_grid:
        grid_image = get_grid_image_without_highlight()
        cv2.imwrite(str(grid_image_path), cv2.cvtColor(
            grid_image, cv2.COLOR_RGB2BGR))
        print(f"Grid image saved to {grid_image_path}")

    if save_window:
        window.save_img(window_image_path)
        print(f"Window image saved to {window_image_path}")

    return grid_image_path, window_image_path


def get_agent_to_printer_paths(
    grid: list[list[int]],
    agent_dx_dy: list[int] = None
):
    # Make copy of the grid to avoid modifying the original
    grid_copy = np.copy(grid)

    # Get agent col and row from grid
    for agent_row_col in np.argwhere(grid_copy == CellType.Agent.value):
        agent_col_row = [agent_row_col[1], agent_row_col[0]]
        break

    # Get printer col and row from grid
    for printer_row_col in np.argwhere(grid_copy == CellType.Printer.value):
        printer_col_row = [printer_row_col[1], printer_row_col[0]]
        break

    # Set agent position to empty
    grid_copy[agent_col_row[1]][agent_col_row[0]] = CellType.Empty.value
    # Set printer position to empty
    grid_copy[printer_col_row[1]][printer_col_row[0]] = CellType.Empty.value

    shortest_paths = find_paths(
        grid=grid_copy,
        start=agent_col_row,
        end=printer_col_row,
        empty_cell_value=CellType.Empty.value,
        initial_direction=agent_dx_dy
    )

    return shortest_paths


def _get_flattened_grid_from_encoding():
    install_a_printer_env = env.env.env

    grid_encoded = install_a_printer_env.grid.encode()
    width = install_a_printer_env.grid.width
    height = install_a_printer_env.grid.height

    flat_grid = np.zeros((height, width), dtype=int)

    for row in range(height):
        for col in range(width):
            cell = grid_encoded[row][col]

    return flat_grid


def key_handler_cartesian(event):
    print('pressed', event.key)
    if event.key == 'escape':
        window.close()
        return
    if event.key == 'backspace':
        reset()
        return
    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return
    # Spacebar
    if event.key == ' ':
        render_furniture()
        return
    if event.key == 'pageup':
        step('choose')
        return
    if event.key == 'enter':
        env.save_state()
        return
    if event.key == 'pagedown':
        show_states()
        return
    if event.key == '0':
        switch_dim(None)
        return
    if event.key == '1':
        switch_dim(0)
        return
    if event.key == '2':
        switch_dim(1)
        return
    if event.key == '3':
        switch_dim(2)
        return


def key_handler_primitive(event):
    print('pressed', event.key)
    match event.key:
        case 'escape':
            window.close()
            return
        case 'left':
            step(env.actions.left)
            return
        case 'right':
            step(env.actions.right)
            return
        case 'up':
            step(env.actions.forward)
            return
        case '0':
            step(env.actions.pickup_0)
            return
        case '1':
            step(env.actions.pickup_1)
            return
        case '2':
            step(env.actions.pickup_2)
            return
        case '3':
            step(env.actions.drop_0)
            return
        case '4':
            step(env.actions.drop_1)
            return
        case '5':
            step(env.actions.drop_2)
            return
        # case 't':
        #     step(env.actions.toggle)
        #     return
        case 'o':
            step(env.actions.open)
            return
        case 'c':
            step(env.actions.close)
            return
        case 'k':
            step(env.actions.cook)
            return
        case 's':
            step(env.actions.slice)
            return
        case 'i':
            step(env.actions.drop_in)
            return
        case 'pagedown':
            show_states()
            return
        case 'a':
            agent_col_row, agent_dx_dy, agent_next_col_row = get_agent_position()
            print(
                f"Agent [col,row]: {agent_col_row}\n"
                f"Agent [dx, dy]: {agent_dx_dy}\n"
                f"Agent next [col,row]: {agent_next_col_row}"
            )
            return
        case 'p':
            printer_col_row = get_printer_position()
            print(f"Printer position: {printer_col_row}")
            return
        case 't':
            [tc, tr], [tw, th] = get_table_colrow_widthheight()
            print(
                f"Table: col={tc}, row={tr}, width={tw}, height={th}"
            )
            return
        case 'd':
            data_item = create_data_item()
            dataset_json_file.write(json.dumps(data_item) + "\n")
            print(f"Data item saved to {dataset_json_path}")
            reset()
            return
        case 'r':
            reset()
            return
        case _:
            pass


def col_row_to_pixel(
    col_row: list[int],
    *,
    tile_pixels: int = TILE_PIXELS,
    center: bool = True,
) -> list[int]:
    """
    Convert grid coordinates [col,row] to pixel coordinates [x,y].

    A 1-pixel border separates tiles.  The top/left outer border is also
    1-pixel thick, so the first tile starts at (border_pixels, border_pixels).

    If `center` is True the function returns the centre of the tile,
    otherwise the top-left corner.
    """
    col, row = col_row
    x_base = col * tile_pixels
    y_base = row * tile_pixels

    if center:
        x_base += tile_pixels // 2
        y_base += tile_pixels // 2

    return [int(x_base), int(y_base)]


def create_data_item():
    print(f'Maze index: {maze_index}')
    grid_image_path, _ = save_image()
    agent_col_row, agent_dx_dy, agent_next_col_row = get_agent_position()
    printer_col_row = get_printer_position()
    agent_pixel_xy = col_row_to_pixel(agent_col_row)
    printer_pixel_xy = col_row_to_pixel(printer_col_row)
    table_col_row, table_width_height = get_table_colrow_widthheight()

    install_a_printer_env = env.env.env
    width = install_a_printer_env.grid.width
    height = install_a_printer_env.grid.height

    flattened_grid = get_flattened_grid_by_composition(
        width,
        height,
        agent_col_row,
        agent_dx_dy,
        printer_col_row,
        table_col_row,
        table_width_height
    )
    agent_to_printer_paths = get_agent_to_printer_paths(
        grid=flattened_grid,
        agent_dx_dy=agent_dx_dy
    )

    # Process each path to generate corresponding directions and actions
    paths_with_instructions = []
    for path in agent_to_printer_paths:
        directions = get_directions(path, initial_direction=agent_dx_dy)
        actions = get_actions(directions)
        paths_with_instructions.append({
            "coordinates": path,
            "directions": directions,
            "instructions": actions
        })

    relative_image_path = grid_image_path.relative_to(
        dataset_json_path.parent)

    data_item = {
        "image_path": str(relative_image_path),
        "agent": {
            "col_row": agent_col_row,
            "xy_pixel": agent_pixel_xy,
        },
        "printer": {
            "col_row": printer_col_row,
            "xy_pixel": printer_pixel_xy,
        },
        "table": {
            "col_row": table_col_row,
            "width_height": table_width_height,
        },
        "paths": paths_with_instructions,
    }

    return data_item


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        help="gym environment to load",
        default='MiniGrid-InstallingAPrinter-8x8-N2-v0'
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        help="size at which to render tiles",
        default=32
    )
    parser.add_argument(
        "--num_mazes",
        type=int,
        help="Number of InstallAPrinter mazes to generate in the dataset",
        default=1
    )
    parser.add_argument(
        '--agent_view',
        default=False,
        help="Draw what the agent sees (partially observable view)",
        action='store_true'
    )
    parser.add_argument(
        "--save",
        action='store_true',
        help="whether or not to save the demo_16"
    )
    parser.add_argument(
        "--load",
        default=None,
        help="path to load state from"
    )

    return parser


def main():
    global env, window, args, all_steps, dataset_path, dataset_json_path, dataset_json_file

    parser = create_parser()
    args = parser.parse_args()
    env = gym.make(args.env)

    all_steps = {}
    datasets_path = Path(__file__).parent / 'datasets'
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = datasets_path / f"installaprinter_{datetime_str}"
    dataset_path.mkdir(parents=True, exist_ok=True)

    dataset_json_path = dataset_path / "dataset.jsonl"
    dataset_json_file = dataset_json_path.open('w')

    window = Window('mini_behavior - ' + args.env)
    if env.mode == "cartesian":
        window.reg_key_handler(key_handler_cartesian)
    elif env.mode == "primitive":
        window.reg_key_handler(key_handler_primitive)

    reset()

    block = True if args.num_mazes < 1 else False
    window.show(block=block)

    print("Exited event loop")

    for _ in range(args.num_mazes):
        data_item = create_data_item()
        dataset_json_file.write(json.dumps(data_item) + "\n")
        print(f"Data item saved to {dataset_json_path}")
        reset()

    dataset_json_file.close()
    # print("Waiting for 10 seconds before starting the demo...")

    # agent_pos = get_agent_pos(env)
    # sleep(10)


if __name__ == "__main__":
    main()
