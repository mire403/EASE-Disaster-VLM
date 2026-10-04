from pathlib import Path

from floodnet_rcmtd.config import load_config


def test_load_config_resolves_paths_and_preserves_seed(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("seed: 17\ndata_root: ./data\n", encoding="utf-8")

    config = load_config(config_file)

    assert config["seed"] == 17
    assert config["data_root"] == str((tmp_path / "data").resolve())


def test_default_data_config_sets_majority_answer_warning_threshold():
    config = load_config(
        Path(__file__).parents[1] / "configs" / "data" / "floodnet_v1.yaml"
    )

    assert config["max_majority_answer_fraction"] == 0.95
