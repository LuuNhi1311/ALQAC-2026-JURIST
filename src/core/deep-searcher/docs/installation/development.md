# 🛠️ Development Mode Installation

This guide is for contributors who want to modify DeepSearcher's code or develop new features.

## 📋 Prerequisites

- Python 3.10 or higher
- git
- [uv](https://github.com/astral-sh/uv) package manager (recommended for faster installation)

## 🔄 Installation Steps

### Step 1: Install uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a faster alternative to pip for Python package management.

=== "Using pip"
    ```bash
    pip install uv
    ```

=== "Using curl (Unix/macOS)"
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Using PowerShell (Windows)"
    ```powershell
    irm https://astral.sh/uv/install.ps1 | iex
    ```

For more options, see the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Step 2: Clone the repository

```bash
git clone https://github.com/zilliztech/deep-searcher.git
cd deep-searcher
```

### Step 3: Set up the development environment

=== "Using uv (Recommended)"
    ```bash
    uv sync
    source .venv/bin/activate
    ```

=== "Using pip"
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -e ".[dev,all]"
    ```

## 🧪 Running Tests

```bash
pytest tests/
```

## 📚 Additional Resources

For more detailed development setup instructions, including contribution guidelines, code style, and testing procedures, please refer to the [CONTRIBUTING.md](https://github.com/zilliztech/deep-searcher/blob/main/CONTRIBUTING.md) file in the repository. 