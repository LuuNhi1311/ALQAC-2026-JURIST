# 🚀 Quick Start

## Prerequisites

✅ Before you begin, prepare your `OPENAI_API_KEY` in your environment variables. If you change the LLM in the configuration, make sure to prepare the corresponding API key.

## Basic Usage

```python
# Import configuration modules
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.online_query import query

# Initialize configuration
config = Configuration()

# Customize your config here
# (See the Configuration Details section below for more options)
config.set_provider_config("llm", "OpenAI", {"model": "o1-mini"})
config.set_provider_config("embedding", "OpenAIEmbedding", {"model": "text-embedding-ada-002"})
init_config(config=config)

# Load data from local files
from deepsearcher.offline_loading import load_from_local_files
load_from_local_files(paths_or_directory=your_local_path)

# (Optional) Load data from websites
# Requires FIRECRAWL_API_KEY environment variable
from deepsearcher.offline_loading import load_from_website
load_from_website(urls=website_url)

# Query your data
result = query("Write a report about xxx.")  # Replace with your question
print(result)
```

## Next Steps

After completing this quick start, you might want to explore:

- [Command Line Interface](cli.md) for non-programmatic usage
- [Deployment](deployment.md) for setting up a web service 