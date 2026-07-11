# Configuration Overview

DeepSearcher provides flexible configuration options for all its components. You can customize the following aspects of the system:

## 📋 Components

| Component | Purpose | Documentation |
|-----------|---------|---------------|
| **LLM** | Large Language Models for query processing | [LLM Configuration](llm.md) |
| **Embedding Models** | Text embedding for vector retrieval | [Embedding Models](embedding.md) |
| **Vector Database** | Storage and retrieval of vector embeddings | [Vector Database](vector_db.md) |
| **File Loader** | Loading and processing various file formats | [File Loader](file_loader.md) |
| **Web Crawler** | Gathering information from web sources | [Web Crawler](web_crawler.md) |

## 🔄 Configuration Method

DeepSearcher uses a consistent configuration approach for all components:

```python
from deepsearcher.configuration import Configuration, init_config

# Create configuration
config = Configuration()

# Set provider configurations
config.set_provider_config("[component]", "[provider]", {"option": "value"})

# Initialize with configuration
init_config(config=config)
```

For detailed configuration options for each component, please visit the corresponding documentation pages linked in the table above.

