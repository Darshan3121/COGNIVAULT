# Contributing to Cognivault

Thank you for your interest in contributing to Cognivault! We welcome bug fixes, documentation improvements, feature enhancements, and test coverage.

Please take a few moments to review these guidelines before submitting a contribution.

---

## Code of Principles

- **Document Grounding First**: Features must respect document boundaries and grounding. Changes should never weaken document isolation or fabricate unsupported facts.
- **Focused Changes**: Keep pull requests focused on a single bug fix or feature. Avoid bundling unrelated refactorings or cosmetic changes with functional updates.
- **Local Privacy**: Cognivault prioritizes local LLM inference and privacy. Do not introduce requirements for mandatory external proprietary cloud APIs without configurable fallbacks.

---

## Prerequisites

- **Python**: Version 3.10 or higher (`python --version`)
- **Node.js**: Version 20.9.0 or higher (`node --version`)
- **Ollama**: Installed with `mistral:latest` pulled (`ollama pull mistral:latest`)

---

## Development Setup

> [!NOTE]
> All setup commands below assume your terminal starts at the repository root (`COGNIVAULT`).

1. **Fork and Clone the Repository**:
   Fork the repository on GitHub and clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/COGNIVAULT.git
   cd COGNIVAULT
   ```

2. **Set Up the Backend**:
   Navigate to the backend directory, create a virtual environment, activate it, and install dependencies:

   **Windows (PowerShell):**
   ```powershell
   cd backend
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   **macOS / Linux:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set Up the Frontend**:
   From the repository root:
   ```bash
   cd frontend
   npm install
   ```

4. **Verify Ollama**:
   - Check if Ollama is running (`ollama list` or check the Windows System Tray).
   - If not running, launch the Ollama desktop app (or run `ollama serve` in a dedicated terminal if on headless Linux).
   - Ensure the required model is pulled:
     ```bash
     ollama pull mistral:latest
     ```

---

## Branching & Workflow

1. Create a descriptive branch from `main`:
   ```bash
   git checkout -b fix/mcq-array-validation
   # or
   git checkout -b feat/export-mcq-json
   ```

2. **Commit Conventions**:
   Use clear, conventional commit messages:
   - `fix: correct regex pattern for ragged 2D array parsing`
   - `feat: add markdown export for generated MCQs`
   - `docs: update backend installation notes for macOS`
   - `test: add regression test for String immutability explanation`

---

## Running Verification & Tests

Before submitting a pull request, run the verification checks:

1. **Backend Python Compilation**:
   From the repository root, ensure all Python files compile cleanly without syntax errors:
   ```bash
   python -m compileall backend/app
   ```

2. **Backend Regression Audit**:
   Cognivault includes an end-to-end regression audit suite (`backend/test_regression_audit.py`) that queries the live backend API.

   **Prerequisites:**
   - Ollama must be running with `mistral:latest` available (`ollama list`).
   - The FastAPI backend must be actively running in a separate terminal (e.g. from `backend/`: `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`).

   **Execution (from the `backend/` directory with venv activated):**

   *On Windows (PowerShell):*
   ```powershell
   cd backend
   .\venv\Scripts\Activate.ps1
   python -u test_regression_audit.py
   ```

   *On macOS / Linux (bash):*
   ```bash
   cd backend
   source venv/bin/activate
   python -u test_regression_audit.py
   ```

3. **Frontend TypeScript Check**:
   From the repository root:
   ```bash
   cd frontend
   npx tsc --noEmit
   ```

---

## Submitting a Pull Request

1. Push your branch to your GitHub fork:
   ```bash
   git push -u origin your-branch-name
   ```
2. Open a Pull Request against the `main` branch of `Darshan3121/COGNIVAULT`.
3. Provide a clear PR description explaining:
   - What changed and why.
   - Any manual or automated tests executed.
   - Any breaking changes or new configuration options.

---

## Reporting Issues

- **Bug Reports**: Please open an issue using the Bug Report template. Include your operating system, Python/Node versions, steps to reproduce, and any relevant error logs.
- **Feature Requests**: Submit a Feature Request detailing the user story, proposed solution, and alternative approaches considered.

---

## Licensing Notice

This repository currently does not have a published open-source license, and all rights are reserved by the maintainer ([@Darshan3121](https://github.com/Darshan3121)).

If you are interested in contributing, please contact the maintainer to discuss and clarify licensing expectations before submitting substantial code contributions.
