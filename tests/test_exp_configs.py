"""Smoke test for all 5 ablation experiment configurations."""

from pathlib import Path

import pytest

from scripts.train import load_config, train

CONFIG_FILES = [
    "configs/exp/e0_baseline_val.yaml",
    "configs/exp/e1_cosface.yaml",
    "configs/exp/e2_cosface_hardid.yaml",
    "configs/exp/e3_cosface_hardid_absneg.yaml",
    "configs/exp/e1b_arcface.yaml",
    "configs/exp/e1c_cosface_small.yaml",
    "configs/exp/e4_ce_hardid.yaml",
    "configs/exp/e5_ce_hardid_absneg.yaml",
]


@pytest.mark.parametrize("cfg_path", CONFIG_FILES)
def test_config_structure(cfg_path: str) -> None:
    """Verify that each config file exists, parses correctly, and has matching exp_name."""
    config = load_config(cfg_path)
    expected_exp_name = Path(cfg_path).stem
    assert config.get("exp_name") == expected_exp_name
    assert "data" in config
    assert "training" in config
    assert "loss" in config
    assert config["data"].get("val_identities") == 100


def test_specific_config_settings() -> None:
    """Verify hyperparameter settings across the ablations."""
    c0 = load_config("configs/exp/e0_baseline_val.yaml")
    assert c0["loss"].get("cls", "ce") == "ce"
    assert c0["data"].get("sampler", "pk") == "pk"

    c1 = load_config("configs/exp/e1_cosface.yaml")
    assert c1["loss"]["cls"] == "cosface"
    assert c1["loss"]["margin_m"] == 0.35
    assert c1["loss"]["scale_s"] == 30.0

    c2 = load_config("configs/exp/e2_cosface_hardid.yaml")
    assert c2["loss"]["cls"] == "cosface"
    assert c2["data"]["sampler"] == "hard_identity"
    assert c2["data"]["hard_sampler"]["group_size"] == 4
    assert c2["data"]["hard_sampler"]["m_neighbors"] == 12

    c3 = load_config("configs/exp/e3_cosface_hardid_absneg.yaml")
    assert c3["loss"]["cls"] == "cosface"
    assert c3["data"]["sampler"] == "hard_identity"
    assert c3["loss"]["neg_pair"]["enabled"] is True
    assert c3["loss"]["neg_pair"]["tau_neg"] == 0.5
    assert c3["loss"]["neg_pair"]["weight"] == 0.5

    c1b = load_config("configs/exp/e1b_arcface.yaml")
    assert c1b["loss"]["cls"] == "arcface"
    assert c1b["loss"]["margin_m"] == 0.5
    assert c1b["loss"]["scale_s"] == 30.0

    c1c = load_config("configs/exp/e1c_cosface_small.yaml")
    assert c1c["loss"]["cls"] == "cosface"
    assert c1c["loss"]["margin_m"] == 0.20
    assert c1c["loss"]["scale_s"] == 30.0
    assert c1c["loss"]["margin_warmup_epochs"] == 10
    assert c1c["data"].get("sampler", "pk") == "pk"

    c4 = load_config("configs/exp/e4_ce_hardid.yaml")
    assert c4["loss"].get("cls", "ce") == "ce"
    assert c4["loss"]["label_smoothing"] == 0.1
    assert c4["data"]["sampler"] == "hard_identity"
    assert c4["data"]["hard_sampler"]["group_size"] == 4
    assert c4["data"]["hard_sampler"]["m_neighbors"] == 12

    c5 = load_config("configs/exp/e5_ce_hardid_absneg.yaml")
    assert c5["loss"].get("cls", "ce") == "ce"
    assert c5["loss"]["label_smoothing"] == 0.1
    assert c5["data"]["sampler"] == "hard_identity"
    assert c5["loss"]["neg_pair"]["enabled"] is True
    assert c5["loss"]["neg_pair"]["tau_neg"] == 0.5
    assert c5["loss"]["neg_pair"]["weight"] == 0.5


@pytest.mark.parametrize("cfg_path", CONFIG_FILES)
def test_smoke_run_cpu(cfg_path: str, tmp_path: Path) -> None:
    """Run a fast 1-epoch smoke test on CPU with 50 identities."""
    config = load_config(cfg_path)
    config["output_root"] = str(tmp_path / "outputs")
    config["smoke_test"] = True
    config["smoke_test_identities"] = 50
    config["smoke_test_epochs"] = 1
    config["training"]["device"] = "cpu"
    config["training"]["use_amp"] = False
    config["training"]["val_every"] = 1
    config["data"]["num_workers"] = 0
    config["data"]["p"] = 4
    config["data"]["k"] = 4
    if "hard_sampler" in config.get("data", {}):
        config["data"]["hard_sampler"]["group_size"] = 2
        config["data"]["hard_sampler"]["m_neighbors"] = 4
        config["data"]["hard_sampler"]["mining_start_epoch"] = 1
        config["data"]["hard_sampler"]["refresh_every"] = 1

    # Check if dataset is available on machine; if not, skip smoke execution
    data_dir = Path(config.get("data_root", "data/Market-1501-v15.09.15"))
    if not (data_dir / "bounding_box_train").is_dir():
        pytest.skip(f"Market-1501 dataset not found at {data_dir}")

    ckpt_path = train(config)
    assert ckpt_path.is_file()
    val_csv = tmp_path / "outputs" / f"{Path(cfg_path).stem}_smoke" / "val_metrics.csv"
    assert val_csv.is_file()
