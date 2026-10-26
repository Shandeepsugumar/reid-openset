"""Pre-commit guard to prevent committing large files, datasets, weights, or secrets."""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Maximum allowed file size for repository tracking (5 MB)
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024

# Extensions that should never be tracked in the repository
FORBIDDEN_EXTENSIONS = {
    ".zip",
    ".7z",
    ".tar",
    ".gz",
    ".tgz",
    ".pth",
    ".pt",
    ".ckpt",
    ".onnx",
    ".engine",
    ".npy",
    ".npz",
    ".pkl",
    ".bin",
}

# Directories whose contents should never be tracked
FORBIDDEN_DIRECTORIES = {
    "data",
    "datasets",
    "checkpoints",
    "runs",
    "artifacts",
    "outputs",
    "wandb",
}

# Extensions and filename patterns that indicate secrets or private credentials
SECRET_EXTENSIONS = {".key", ".pem"}


def check_file(
    file_path: Union[str, Path],
    file_size_bytes: Optional[int] = None,
    max_size: int = MAX_FILE_SIZE_BYTES,
) -> Tuple[bool, Optional[str]]:
    """Validate whether a file is safe to commit.

    Pure function that does not depend on git state, making it directly testable.

    Args:
        file_path: Path or filename string.
        file_size_bytes: Optional explicit file size in bytes. If None and the file
                         exists on disk, os.path.getsize is used.
        max_size: Maximum allowed file size in bytes (default 5 MB).

    Returns:
        (is_valid, failure_reason)
    """
    path_obj = Path(file_path)
    norm_path = path_obj.as_posix()
    parts = norm_path.split("/")
    filename = path_obj.name.lower()
    suffix = path_obj.suffix.lower()

    # 1. Directory restriction check
    for part in parts[:-1]:
        part_clean = part.lower().strip()
        if part_clean in FORBIDDEN_DIRECTORIES or part_clean.startswith("market-1501"):
            return False, f"File sits inside forbidden directory '{part}'"

    # 2. Forbidden binary/model/dataset extensions
    # Handle compound extensions like .tar.gz
    double_suffix = "".join(path_obj.suffixes[-2:]).lower() if len(path_obj.suffixes) >= 2 else ""
    if suffix in FORBIDDEN_EXTENSIONS or double_suffix in FORBIDDEN_EXTENSIONS:
        ext = double_suffix if double_suffix in FORBIDDEN_EXTENSIONS else suffix
        return False, f"Forbidden large/binary file extension '{ext}'"

    # 3. Secret and credential patterns
    if suffix in SECRET_EXTENSIONS:
        return False, f"Secret credential extension '{suffix}' detected"

    if filename == ".env" or filename.startswith(".env."):
        return False, "Environment/secret file (.env) detected"

    if "token" in filename or filename.startswith("secret"):
        return False, f"Sensitive credential pattern in filename: '{path_obj.name}'"

    # 4. File size check
    if file_size_bytes is None and path_obj.is_file():
        try:
            file_size_bytes = path_obj.stat().st_size
        except OSError:
            file_size_bytes = None

    if file_size_bytes is not None and file_size_bytes > max_size:
        size_mb = file_size_bytes / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        return False, f"File size exceeds {max_mb:.1f} MB limit ({size_mb:.2f} MB)"

    return True, None


def get_staged_files() -> List[str]:
    """Retrieve list of files currently staged in git index."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return lines
    except (subprocess.SubprocessError, FileNotFoundError):
        return []


def main() -> int:
    staged_files = get_staged_files()
    if not staged_files:
        print("[Hygiene Check] No staged files detected. Check passed.")
        return 0

    violations = []
    for filepath in staged_files:
        is_valid, reason = check_file(filepath)
        if not is_valid:
            violations.append((filepath, reason))

    if violations:
        print("\n" + "=" * 70)
        print("ERROR: Staged files violate repository hygiene rules:")
        for path, reason in violations:
            print(f"  - {path}: {reason}")
        print("=" * 70)
        print("Action required: Unstage these files before committing:")
        print("  git restore --staged <file>   # or: git rm --cached <file>")
        print("=" * 70 + "\n")
        return 1

    print(f"[Hygiene Check] All {len(staged_files)} staged file(s) passed checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
