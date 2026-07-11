# Unstructured Integration Example

This example demonstrates how to use the Unstructured library with DeepSearcher for advanced document parsing.

## Overview

Unstructured is a powerful document processing library that can extract content from various document formats. This example shows:

1. Setting up Unstructured with DeepSearcher
2. Configuring the Unstructured API keys (optional)
3. Loading documents with Unstructured's parser
4. Querying the extracted content

## Code Example

```python
import logging
import os
from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.online_query import query
from deepsearcher.configuration import Configuration, init_config

# Suppress unnecessary logging from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)

# (Optional) Set API keys (ensure these are set securely in real applications)
os.environ['UNSTRUCTURED_API_KEY'] = '***************'
os.environ['UNSTRUCTURED_API_URL'] = '***************'


def main():
    # Step 1: Initialize configuration
    config = Configuration()

    # Configure Vector Database (Milvus) and File Loader (UnstructuredLoader)
    config.set_provider_config("vector_db", "Milvus", {})
    config.set_provider_config("file_loader", "UnstructuredLoader", {})

    # Apply the configuration
    init_config(config)

    # Step 2: Load data from a local file or directory into Milvus
    input_file = "your_local_file_or_directory"  # Replace with your actual file path
    collection_name = "Unstructured"
    collection_description = "All Milvus Documents"

    load_from_local_files(paths_or_directory=input_file, collection_name=collection_name, collection_description=collection_description)

    # Step 3: Query the loaded data
    question = "What is Milvus?"  # Replace with your actual question
    result = query(question)


if __name__ == "__main__":
    main()
```

## Running the Example

1. Install DeepSearcher with Unstructured support: `pip install deepsearcher "unstructured[all-docs]"`
2. (Optional) Sign up for the Unstructured API at [unstructured.io](https://unstructured.io) if you want to use their cloud service
3. Replace `your_local_file_or_directory` with your own document file path or directory
4. Run the script: `python load_local_file_using_unstructured.py`

## Unstructured Options

You can use Unstructured in two modes:

1. **API Mode**: Set the environment variables `UNSTRUCTURED_API_KEY` and `UNSTRUCTURED_API_URL` to use their cloud service
2. **Local Mode**: Don't set the environment variables, and Unstructured will process documents locally on your machine

## Key Concepts

- **Document Processing**: Advanced document parsing for various formats
- **API/Local Options**: Flexibility in deployment based on your needs
- **Integration**: Seamless integration with DeepSearcher's vector database and query capabilities 