# 📦 Installation via pip

This method is recommended for most users who want to use DeepSearcher without modifying its source code.

## 📋 Prerequisites

- Python 3.10 or higher
- pip package manager (included with Python)
- Virtual environment tool (recommended)

## 🔄 Step-by-Step Installation

### Step 1: Create a virtual environment

```bash
python -m venv .venv
```

### Step 2: Activate the virtual environment

=== "Linux/macOS"
    ```bash
    source .venv/bin/activate
    ```

=== "Windows"
    ```bash
    .venv\Scripts\activate
    ```

### Step 3: Install DeepSearcher

```bash
pip install deepsearcher
```

## 🧩 Optional Dependencies

DeepSearcher supports various integrations through optional dependencies.

| Integration | Command | Description |
|-------------|---------|-------------|
| Ollama | `pip install "deepsearcher[ollama]"` | For local LLM deployment |
| All extras | `pip install "deepsearcher[all]"` | Installs all optional dependencies |

## ✅ Verify Installation

```python
# Simple verification
from deepsearcher import __version__
print(f"DeepSearcher version: {__version__}")
``` 