from typing import List, Tuple

# from deepsearcher.configuration import vector_db, embedding_model, llm
from deepsearcher import configuration
from deepsearcher.vector_db.base import RetrievalResult


def query(original_query: str, max_iter: int = 3) -> Tuple[str, List[RetrievalResult], int]:
    """
    Query the knowledge base with a question and get an answer.

    This function uses the default searcher to query the knowledge base and generate
    an answer based on the retrieved information.

    Args:
        original_query: The question or query to search for.
        max_iter: Maximum number of iterations for the search process.

    Returns:
        A tuple containing:
            - The generated answer as a string
            - A list of retrieval results that were used to generate the answer
            - The number of tokens consumed during the process
    """
    default_searcher = configuration.default_searcher
    return default_searcher.query(original_query, max_iter=max_iter)


def retrieve(
    original_query: str, max_iter: int = 3
) -> Tuple[List[RetrievalResult], List[str], int]:
    """
    Retrieve relevant information from the knowledge base without generating an answer.

    This function uses the default searcher to retrieve information from the knowledge base
    that is relevant to the query.

    Args:
        original_query: The question or query to search for.
        max_iter: Maximum number of iterations for the search process.

    Returns:
        A tuple containing:
            - A list of retrieval results
            - An empty list (placeholder for future use)
            - The number of tokens consumed during the process
    """
    default_searcher = configuration.default_searcher
    retrieved_results, consume_tokens, metadata = default_searcher.retrieve(
        original_query, max_iter=max_iter
    )
    return retrieved_results, [], consume_tokens


def naive_retrieve(query: str, collection: str = None, top_k=10) -> List[RetrievalResult]:
    """
    Perform a simple retrieval from the knowledge base using the naive RAG approach.

    This function uses the naive RAG agent to retrieve information from the knowledge base
    without any advanced techniques like iterative refinement.

    Args:
        query: The question or query to search for.
        collection: The name of the collection to search in. If None, searches in all collections.
        top_k: The maximum number of results to return.

    Returns:
        A list of retrieval results.
    """
    naive_rag = configuration.naive_rag
    all_retrieved_results, consume_tokens, _ = naive_rag.retrieve(query)
    return all_retrieved_results


def naive_rag_query(
    query: str, collection: str = None, top_k=10
) -> Tuple[str, List[RetrievalResult]]:
    """
    Query the knowledge base using the naive RAG approach and get an answer.

    This function uses the naive RAG agent to query the knowledge base and generate
    an answer based on the retrieved information, without any advanced techniques.

    Args:
        query: The question or query to search for.
        collection: The name of the collection to search in. If None, searches in all collections.
        top_k: The maximum number of results to consider.

    Returns:
        A tuple containing:
            - The generated answer as a string
            - A list of retrieval results that were used to generate the answer
    """
    naive_rag = configuration.naive_rag
    answer, retrieved_results, consume_tokens = naive_rag.query(query)
    return answer, retrieved_results
