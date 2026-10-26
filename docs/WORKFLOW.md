# Repository Development Workflow & Git Rules

All development, experimentation, and maintenance tasks within this repository must adhere to the following workflow rules.

---

## 1. Branching Strategy & Branch Protection

- **`main` is protected by convention**:
  - Never commit directly to `main`.
  - All contributions must land on `main` via reviewed Pull Requests (PRs).

- **Task Inception Routine**:
  Before starting any new work or experiment, synchronize with remote `main` and branch off:
  ```bash
  git checkout main
  git pull --rebase origin main
  git checkout -b feature/<topic>     # for new features, porting, or tooling
  # or:
  git checkout -b exp/<experiment>    # for scientific experiments or model variations
  # or:
  git checkout -b fix/<bugfix>        # for defect corrections
  ```

---

## 2. Commit Standards

- **Commit frequently in small, logical increments**.
- Follow conventional commit prefixes:
  - `feat:` New features, modules, or functional capabilities.
  - `fix:` Bug fixes, calculation adjustments, or error corrections.
  - `test:` Unit tests, regression tests, or test harness updates.
  - `docs:` Documentation, benchmark notes, or workflow updates.
  - `chore:` Maintenance, refactoring, dependencies, or configuration changes.
  - `exp:` Standalone experiment runs or configuration variants.

---

## 3. Pull Request and Merging Routine

1. **Push branch to remote**:
   ```bash
   git push -u origin <branch-name>
   ```

2. **Open a Pull Request**:
   - Create a Pull Request into `main` using GitHub CLI (`gh pr create`) or via the GitHub web interface.
   - The user reviews and merges the PR on GitHub.
   - Do **NOT** merge the PR autonomously unless explicitly instructed.

3. **Post-Merge Local Cleanup**:
   After the PR is merged on GitHub:
   ```bash
   git checkout main
   git pull --rebase origin main
   git branch -d <branch-name>
   ```

---

## 4. Strict Repository Hygiene & Data Safety Rules

- **Never force-push (`git push -f`)** to any branch.
- **Never commit or push**:
  - Raw or extracted datasets (`data/`, `datasets/`, `Market-1501*/`).
  - Model checkpoints, weights, or engine files (`*.pth`, `*.pt`, `*.ckpt`, `*.onnx`, `*.engine`).
  - Serialized embeddings or arrays (`*.npy`, `*.npz`, `*.pkl`, `artifacts/`).
  - Private environment files, secret tokens, or API keys (`.env*`, `*.pem`, `*.key`, `*token*`).
- **Hygiene Verification**:
  - The repository has an active pre-commit hook in `.githooks/pre-commit` configured via `core.hooksPath`.
  - Staged files exceeding 5 MB or carrying forbidden extensions are automatically blocked.
