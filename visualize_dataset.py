from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

import streamlit as st
from PIL import Image


def _load_dataset(path: Path) -> List[dict]:
    """Read a `.jsonl` dataset file and return a list with one dict per line."""
    with path.open("r") as fp:
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
    st.set_page_config(layout="wide")
    st.title("MiniBehavior - Install A Printer Dataset Viewer")

    # Use first command line argument as dataset path if provided
    default_dataset_path = "datasets/installaprinter_20250516_222606/dataset.jsonl"
    if len(sys.argv) > 1:
        default_dataset_path = sys.argv[1]
    
    dataset_path_str = st.text_input(  # keep as *str* for reliable comparison
        "Path to dataset (.jsonl)",
        value=default_dataset_path,
    )

    if not dataset_path_str or not os.path.exists(dataset_path_str):
        st.error("File does not exist. Please check the path.")
    elif not dataset_path_str.endswith('.jsonl'):
        st.error("File is not a .jsonl file.")

    if (
        "dataset_path" not in st.session_state
        or st.session_state.dataset_path != dataset_path_str
    ):
        dataset_path = Path(dataset_path_str)
        if not dataset_path.is_file():
            st.warning("Please provide a valid *.jsonl dataset file.")
            st.stop()

        st.session_state.dataset_items = _load_dataset(dataset_path)
        st.session_state.dataset_path = dataset_path_str
        st.session_state.index = 0

    items = st.session_state.dataset_items

    # Abort if the file was empty / unreadable
    if not items:
        st.warning("The selected dataset file is empty or could not be parsed.")
        st.stop()

    num_items = len(items)

    # Control buttons
    col_prev, col_next, col_counter, col_goto = st.columns(4)

    with col_prev:
        if st.button("⬅ Previous", disabled=st.session_state.index <= 0):
            st.session_state.index = max(0, st.session_state.index - 1)

    with col_next:
        if st.button("Next ➡", disabled=st.session_state.index >= num_items - 1):
            st.session_state.index = min(num_items - 1, st.session_state.index + 1)

    with col_counter:
        st.write(f"Item: {st.session_state.index + 1} / {num_items}")

    with col_goto:
        # “Go to” control – updates index first, counter is shown afterwards
        goto_line = st.number_input(
            "Go to line (1-based)",
            min_value=1,
            max_value=num_items,
            step=1,
            value=st.session_state.index + 1,
            label_visibility="collapsed",
        )
        if goto_line - 1 != st.session_state.index:
            st.session_state.index = goto_line - 1

    # Display current item
    item = items[st.session_state.index]

    left, right = st.columns(2)

    with left:
        img_path = _absolute_image_path(
            Path(st.session_state.dataset_path), item["image_path"])
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
