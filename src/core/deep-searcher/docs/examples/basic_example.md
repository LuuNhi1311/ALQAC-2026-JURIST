# Basic Example

This example demonstrates the core functionality of DeepSearcher - loading documents and performing semantic search.

## Overview

The script performs these steps:

1. Configures DeepSearcher with default settings
2. Loads a PDF document about Milvus
3. Asks a question about Milvus and vector databases
4. Displays token usage information

## Code Example

```python
import logging
import os

from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.online_query import query
from deepsearcher.configuration import Configuration, init_config

httpx_logger = logging.getLogger("httpx")  # disable openai's logger output
httpx_logger.setLevel(logging.WARNING)

current_dir = os.path.dirname(os.path.abspath(__file__))

config = Configuration()  # Customize your config here
init_config(config=config)


# You should clone the milvus docs repo to your local machine first, execute:
# git clone https://github.com/milvus-io/milvus-docs.git
# Then replace the path below with the path to the milvus-docs repo on your local machine
# import glob
# all_md_files = glob.glob('xxx/milvus-docs/site/en/**/*.md', recursive=True)
# load_from_local_files(paths_or_directory=all_md_files, collection_name="milvus_docs", collection_description="All Milvus Documents")

# Hint: You can also load a single file, please execute it in the root directory of the deep searcher project
load_from_local_files(
    paths_or_directory=os.path.join(current_dir, "data/WhatisMilvus.pdf"),
    collection_name="milvus_docs",
    collection_description="All Milvus Documents",
    # force_new_collection=True, # If you want to drop origin collection and create a new collection every time, set force_new_collection to True
)

question = "Write a report comparing Milvus with other vector databases."

_, _, consumed_token = query(question, max_iter=1)
print(f"Consumed tokens: {consumed_token}")
```

## Running the Example

1. Make sure you have installed DeepSearcher: `pip install deepsearcher`
2. Create a data directory and add a PDF about Milvus (or use your own data)
3. Run the script: `python basic_example.py`

## Key Concepts

- **Configuration**: Using the default configuration
- **Document Loading**: Loading a single PDF file
- **Querying**: Asking a complex question requiring synthesis of information
- **Token Tracking**: Monitoring token usage from the LLM 