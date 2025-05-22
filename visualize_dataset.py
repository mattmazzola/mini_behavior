from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

import streamlit as st
from PIL import Image


def _load_jsonl(path: str | Path) -> List[dict]:
    """Read a `.jsonl` dataset file and return a list with one dict per line."""
    with Path(path).open("r") as fp:
        return [json.loads(line) for line in fp.readlines()]


def _absolute_image_path(dataset_path: Path, rel_img: str | Path) -> Path:
    """Resolve image path relative to the dataset file."""
    rel_img = Path(rel_img)
    # If already absolute – just return
    if rel_img.is_absolute():
        return rel_img
    return (dataset_path.parent / rel_img).resolve()


def main():
    # Use full-width page layout
    st.set_page_config(
        page_title="MiniBehavior - Install A Printer Dataset Viewer",
        layout="wide",
    )

    # Initialize session state
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    # Use first command line argument as dataset path if provided
    default_dataset_path = "datasets/installaprinter_20250516_222606/dataset.jsonl"
    if len(sys.argv) > 1:
        default_dataset_path = sys.argv[1]

    dataset_path_str = st.text_input(
        "Path to dataset (.jsonl)",
        value=default_dataset_path,
    )

    if not dataset_path_str or not os.path.exists(dataset_path_str):
        st.error("File does not exist. Please check the path.")
    elif not dataset_path_str.endswith(".jsonl"):
        st.error("File is not a .jsonl file.")

    if "dataset_path" not in st.session_state or st.session_state.dataset_path != dataset_path_str:
        try:
            dataset = _load_jsonl(dataset_path_str)
            st.success(f"Dataset loaded successfully! Found {len(dataset)} entries.")
        except json.JSONDecodeError:
            st.error("Error parsing the JSONL file. Make sure it contains valid JSON lines.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            st.stop()

        st.session_state.dataset_items = dataset
        st.session_state.dataset_path = dataset_path_str
        st.session_state.current_index = 0

    items = st.session_state.dataset_items

    # Abort if the file was empty / unreadable
    if not items:
        st.warning("The selected dataset file is empty or could not be parsed.")
        st.stop()

    num_items = len(items)

    # Functions to handle navigation
    def go_previous():
        st.session_state.current_index = max(0, st.session_state.current_index - 1)

    def go_next():
        st.session_state.current_index = min(num_items - 1, st.session_state.current_index + 1)

    def jump_to_line():
        # Convert from 1-based (UI) to 0-based (internal)
        st.session_state.current_index = st.session_state.jump_input - 1

    # Control buttons
    col_prev, col_next, col_counter, col_goto = st.columns(4)

    # Previous and Next buttons adjacent to each other
    with col_prev:
        prev_disabled = st.session_state.current_index <= 0
        st.button("⬅️ Previous", on_click=go_previous, disabled=prev_disabled, key="prev_button")

    with col_next:
        next_disabled = st.session_state.current_index >= num_items - 1
        st.button("Next ➡️", on_click=go_next, disabled=next_disabled, key="next_button")

    # Show entry number and total
    with col_counter:
        st.write(f"Line {st.session_state.current_index + 1} of {num_items}")

    with col_goto:
        col_label, col_input = st.columns([1, 3])
        with col_label:
            st.write("Go to line:")
        with col_input:
            st.number_input(
                "Go to line:",
                min_value=1,
                max_value=len(items),
                value=st.session_state.current_index + 1,
                step=1,
                label_visibility="collapsed",
                key="jump_input",
                on_change=jump_to_line,
            )

    item = items[st.session_state.current_index]

    left, right = st.columns(2)

    with left:
        img_path = _absolute_image_path(Path(st.session_state.dataset_path), item["image_path"])
        if img_path.exists():
            st.image(Image.open(img_path), caption=str(img_path))
        else:
            st.error(f"Image not found: {img_path}")

        st.subheader("Prompt:")
        st.text("Placeholder prompt")

    with right:
        st.subheader("Data:")
        st.json(item)


if __name__ == "__main__":
    main()
