"""
Example usage of WatsonX embedding and LLM in DeepSearcher.

This example demonstrates how to configure and use IBM WatsonX
embedding models and language models with DeepSearcher.
"""

import os
from deepsearcher.configuration import Configuration

def main():
    """Example of using WatsonX with DeepSearcher."""

    # Initialize configuration
    config = Configuration()

    # Set up environment variables (alternatively, set these in your shell)
    # os.environ["WATSONX_APIKEY"] = "your-watsonx-api-key"
    # os.environ["WATSONX_URL"] = "https://your-watsonx-instance.com"
    # os.environ["WATSONX_PROJECT_ID"] = "your-project-id"

    # Example 1: Configure WatsonX Embedding
    print("=== WatsonX Embedding Configuration ===")

    # Basic configuration with default model
    config.set_provider_config("embedding", "WatsonXEmbedding", {})

    # Configuration with custom model
    config.set_provider_config("embedding", "WatsonXEmbedding", {
        "model": "ibm/slate-125m-english-rtrvr-v2"
    })

    # Configuration with explicit credentials
    # config.set_provider_config("embedding", "WatsonXEmbedding", {
    #     "model": "sentence-transformers/all-minilm-l6-v2",
    #     "api_key": "your-api-key",
    #     "url": "https://your-watsonx-instance.com",
    #     "project_id": "your-project-id"
    # })

    print("WatsonX Embedding configured successfully!")

    # Example 2: Configure WatsonX LLM
    print("\n=== WatsonX LLM Configuration ===")

    # Basic configuration with default model
    config.set_provider_config("llm", "WatsonX", {})

    # Configuration with custom model and parameters
    config.set_provider_config("llm", "WatsonX", {
        "model": "ibm/granite-3-3-8b-instruct",
        "max_new_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50
    })

    # Configuration with IBM Granite model
    config.set_provider_config("llm", "WatsonX", {
        "model": "ibm/granite-3-3-8b-instruct",
        "max_new_tokens": 512,
        "temperature": 0.1
    })

    print("WatsonX LLM configured successfully!")

    # Example 3: Test embedding functionality
    print("\n=== Testing WatsonX Embedding ===")
    try:
        from deepsearcher.embedding.watsonx_embedding import WatsonXEmbedding

        # Check if environment variables are set
        if all(os.getenv(var) for var in ["WATSONX_APIKEY", "WATSONX_URL", "WATSONX_PROJECT_ID"]):
            embedding = WatsonXEmbedding()

            # Test single query embedding
            query = "What is artificial intelligence?"
            query_embedding = embedding.embed_query(query)
            print(f"Query embedding dimension: {len(query_embedding)}")

            # Test document embeddings
            documents = [
                "Artificial intelligence is a branch of computer science.",
                "Machine learning is a subset of AI.",
                "Deep learning uses neural networks."
            ]
            doc_embeddings = embedding.embed_documents(documents)
            print(f"Document embeddings: {len(doc_embeddings)} vectors of dimension {len(doc_embeddings[0])}")

        else:
            print("Environment variables not set. Skipping embedding test.")

    except ImportError:
        print("WatsonX dependencies not installed. Run: pip install ibm-watsonx-ai")
    except Exception as e:
        print(f"Error testing embedding: {e}")

    # Example 4: Test LLM functionality
    print("\n=== Testing WatsonX LLM ===")
    try:
        from deepsearcher.llm.watsonx import WatsonX

        # Check if environment variables are set
        if all(os.getenv(var) for var in ["WATSONX_APIKEY", "WATSONX_URL", "WATSONX_PROJECT_ID"]):
            llm = WatsonX()

            # Test chat functionality
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Explain what artificial intelligence is in one sentence."}
            ]

            response = llm.chat(messages)
            print(f"LLM Response: {response.content}")
            print(f"Tokens used: {response.total_tokens}")

        else:
            print("Environment variables not set. Skipping LLM test.")

    except ImportError:
        print("WatsonX dependencies not installed. Run: pip install ibm-watsonx-ai")
    except Exception as e:
        print(f"Error testing LLM: {e}")

if __name__ == "__main__":
    main()
