import logging
import os
import time

from deepsearcher.configuration import Configuration, init_config
from deepsearcher.online_query import query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



logger.info("Initializing DeepSearcher configuration")
config = Configuration()
config.set_provider_config("llm", "AzureOpenAI", {
    "model": "gpt-4.1",
    "api_key": "<yourkey>",
    "base_url": "https://<youraifoundry>.openai.azure.com/openai/",
    "api_version": "2024-12-01-preview"
})
config.set_provider_config("embedding", "OpenAIEmbedding", {
    "model": "text-embedding-ada-002",
    "api_key": "<yourkey>",
    "azure_endpoint": "https://<youraifoundry>.openai.azure.com/",
    "api_version": "2023-05-15"
    # Remove api_version and other Azure-specific parameters
})
config.set_provider_config("vector_db", "AzureSearch", {
    "endpoint": "https://<yourazureaisearch>.search.windows.net",
    "index_name": "<yourindex>",
    "api_key": "<yourkey>",
    "vector_field": "content_vector"
})

logger.info("Configuration initialized successfully")

try:
    logger.info("Applying global configuration")
    init_config(config)
    logger.info("Configuration applied globally")

    # Example question
    question = "Create a detailed report about what Python is all about"
    logger.info(f"Processing query: '{question}'")

    start_time = time.time()
    result = query(question)
    query_time = time.time() - start_time
    logger.info(f"Query processed in {query_time:.2f} seconds")

    logger.info("Retrieved result successfully")
    print(result[0])  # Print the first element of the tuple

    # Check if there's a second element in the tuple that contains source documents
    if len(result) > 1 and hasattr(result[1], "__len__"):
        logger.info(f"Found {len(result[1])} source documents")
        for i, doc in enumerate(result[1]):
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                logger.info(f"Source {i+1}: {doc.metadata['source']}")
except Exception as e:
    logger.error(f"Error executing query: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())