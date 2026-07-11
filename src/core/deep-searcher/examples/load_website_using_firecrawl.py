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
    print(result)


if __name__ == "__main__":
    main()
