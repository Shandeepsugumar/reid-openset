"""Unit tests for check_repo_files.py hygiene validator."""

from pathlib import Path
from scripts.check_repo_files import check_file, MAX_FILE_SIZE_BYTES


def test_normal_files_allowed() -> None:
    valid_cases = [
        "src/reid/models/backbone.py",
        "src/reid/data/market1501.py",
        "configs/default.yaml",
        "README.md",
        "pyproject.toml",
        "tests/test_sampler.py",
        "results/baseline_resnet18/eval_results.json",
        "results/baseline_resnet18/roc_curve.png",
    ]
    for path in valid_cases:
        is_valid, reason = check_file(path, file_size_bytes=1024)
        assert is_valid, f"Expected {path} to be valid, but got error: {reason}"


def test_large_file_blocked() -> None:
    # Exactly 5MB should pass; 5MB + 1 byte must fail
    is_valid_at_limit, _ = check_file("big_code.py", file_size_bytes=MAX_FILE_SIZE_BYTES)
    assert is_valid_at_limit

    is_valid_over_limit, reason = check_file("huge_archive.iso", file_size_bytes=MAX_FILE_SIZE_BYTES + 1)
    assert not is_valid_over_limit
    assert "exceeds" in reason


def test_forbidden_extensions_blocked() -> None:
    forbidden_files = [
        "weights/model.pth",
        "weights/checkpoint.pt",
        "model.ckpt",
        "model.onnx",
        "engine.engine",
        "data.zip",
        "backup.tar.gz",
        "embeddings.npy",
        "data.npz",
        "scaler.pkl",
    ]
    for path in forbidden_files:
        is_valid, reason = check_file(path, file_size_bytes=500)
        assert not is_valid, f"Expected {path} to be blocked"
        assert "Forbidden" in reason or "extension" in reason


def test_forbidden_directories_blocked() -> None:
    forbidden_dir_files = [
        "data/Market-1501-v15.09.15/bounding_box_train/0002_c1s1_000451_03.jpg",
        "datasets/market1501/image.jpg",
        "checkpoints/latest_checkpoint.pth",
        "artifacts/baseline_resnet18/gallery_embeddings.npy",
        "runs/experiment_1/events.out",
        "outputs/eval_results.json",
    ]
    for path in forbidden_dir_files:
        is_valid, reason = check_file(path, file_size_bytes=500)
        assert not is_valid, f"Expected {path} to be blocked"
        assert "forbidden directory" in reason


def test_secret_files_blocked() -> None:
    secret_files = [
        ".env",
        ".env.local",
        ".env.production",
        "server.key",
        "cert.pem",
        "github_token.txt",
        "secret_keys.json",
    ]
    for path in secret_files:
        is_valid, reason = check_file(path, file_size_bytes=50)
        assert not is_valid, f"Expected {path} to be blocked as sensitive/secret"
