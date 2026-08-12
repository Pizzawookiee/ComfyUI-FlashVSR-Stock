from __future__ import annotations

import os

import folder_paths


MODEL_CATEGORY = "flashvsr"
MODEL_DIRECTORY = os.path.join(folder_paths.models_dir, MODEL_CATEGORY)

folder_paths.add_model_folder_path(MODEL_CATEGORY, MODEL_DIRECTORY, is_default=True)


def filenames() -> list[str]:
    return [
        name
        for name in folder_paths.get_filename_list(MODEL_CATEGORY)
        if name.lower().endswith(".safetensors")
    ]


def component_filenames(kind: str) -> list[str]:
    names = filenames()
    tests = {
        "dit": lambda n: "flashvsr1_1" in n.lower() or ("flashvsr" in n.lower() and "lq" not in n.lower() and "decoder" not in n.lower()),
        "lq": lambda n: "lq_proj" in n.lower(),
        "prompt": lambda n: "prompt" in n.lower(),
        "decoder": lambda n: "tcdecoder" in n.lower(),
    }
    if kind not in tests:
        raise KeyError(kind)
    matched = [name for name in names if tests[kind](name)]
    return matched or [f"No {kind} safetensors found in models/{MODEL_CATEGORY}"]


def full_path(filename: str) -> str:
    if filename.startswith("No "):
        raise FileNotFoundError(filename)
    return folder_paths.get_full_path_or_raise(MODEL_CATEGORY, filename)
