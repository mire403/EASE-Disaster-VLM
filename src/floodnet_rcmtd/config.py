from pathlib import Path
from typing import Any

import yaml

PATH_KEYS = {"data_root", "output_root", "checkpoint", "manifest"}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for key in PATH_KEYS & data.keys():
        value = Path(data[key])
        data[key] = str((config_path.parent / value).resolve() if not value.is_absolute() else value)
    data["_config_path"] = str(config_path)
    return data
