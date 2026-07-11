# 🔧 Installation

DeepSearcher offers multiple installation methods to suit different user needs.

## 📋 Installation Options

| Method | Best For | Description |
|--------|----------|-------------|
| [📦 Installation via pip](pip.md) | Most users | Quick and easy installation using pip package manager |
| [🛠️ Development mode](development.md) | Contributors | Setup for those who want to modify the code or contribute |

## 🚀 Quick Start

Once installed, you can verify your installation:

```python
from deepsearcher.configuration import Configuration
from deepsearcher.online_query import query

# Initialize with default configuration
config = Configuration()
print("DeepSearcher installed successfully!")
```

## 💻 System Requirements

- Python 3.10 or higher
- 4GB RAM minimum (8GB+ recommended)
- Internet connection for downloading models and dependencies 