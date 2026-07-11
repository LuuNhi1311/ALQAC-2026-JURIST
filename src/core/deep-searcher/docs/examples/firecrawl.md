# FireCrawl Integration Example

This example demonstrates how to use FireCrawl with DeepSearcher to crawl and extract content from websites.

## Overview

FireCrawl is a specialized web crawling service designed for AI applications. This example shows:

1. Setting up FireCrawl with DeepSearcher
2. Configuring API keys for the service
3. Crawling a website and extracting content
4. Querying the extracted content

## Code Example

```python
import logging
import os
from deepsearcher.offline_loading import load_from_website
from deepsearcher.online_query import query
from deepsearcher.configuration import Configuration, init_config

# Suppress unnecessary logging from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)

# Set API keys (ensure these are set securely in real applications)
os.environ['OPENAI_API_KEY'] = 'sk-***************'
os.environ['FIRECRAWL_API_KEY'] = 'fc-***************'


def main():
    # Step 1: Initialize configuration
    config = Configuration()

    # Set up Vector Database (Milvus) and Web Crawler (FireCrawlCrawler)
    config.set_provider_config("vector_db", "Milvus", {})
    config.set_provider_config("web_crawler", "FireCrawlCrawler", {})

    # Apply the configuration
    init_config(config)

    # Step 2: Load data from a website into Milvus
    website_url = "https://example.com"  # Replace with your target website
    collection_name = "FireCrawl"
    collection_description = "All Milvus Documents"

    # crawl a single webpage
    load_from_website(urls=website_url, collection_name=collection_name, collection_description=collection_description)
    # only applicable if using Firecrawl: deepsearcher can crawl multiple webpages, by setting max_depth, limit, allow_backward_links
    # load_from_website(urls=website_url, max_depth=2, limit=20, allow_backward_links=True, collection_name=collection_name, collection_description=collection_description)

    # Step 3: Query the loaded data
    question = "What is Milvus?"  # Replace with your actual question
    result = query(question)


if __name__ == "__main__":
    main()
```

## Running the Example

1. Install DeepSearcher: `pip install deepsearcher`
2. Sign up for a FireCrawl API key at [firecrawl.dev](https://docs.firecrawl.dev/introduction)
3. Replace the placeholder API keys with your actual keys
4. Change the `website_url` to the website you want to crawl
5. Run the script: `python load_website_using_firecrawl.py`

## Advanced Crawling Options

FireCrawl provides several advanced options for crawling:

- `max_depth`: Control how many links deep the crawler should go
- `limit`: Set a maximum number of pages to crawl
- `allow_backward_links`: Allow the crawler to navigate to parent/sibling pages

## Key Concepts

- **Web Crawling**: Extracting content from websites
- **Depth Control**: Managing how deep the crawler navigates
- **URL Processing**: Handling multiple pages from a single starting point
- **Vector Storage**: Storing the crawled content in a vector database for search 