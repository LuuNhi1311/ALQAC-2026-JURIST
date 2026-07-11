from abc import ABC
from typing import Any, List, Tuple

from deepsearcher.vector_db import RetrievalResult


def describe_class(description):
    """
    Decorator function to add a description to a class.

    This decorator adds a __description__ attribute to the decorated class,
    which can be used for documentation or introspection.

    Args:
        description: The description to add to the class.

    Returns:
        A decorator function that adds the description to the class.
    """

    def decorator(cls):
        cls.__description__ = description
        return cls

    return decorator


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the DeepSearcher system.

    This class defines the basic interface for agents, including initialization
    and invocation methods.
    """

    def __init__(self, **kwargs):
        """
        Initialize a BaseAgent object.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def invoke(self, query: str, **kwargs) -> Any:
        """
        Invoke the agent and return the result.

        Args:
            query: The query string.
            **kwargs: Additional keyword arguments.

        Returns:
            The result of invoking the agent.
        """


class RAGAgent(BaseAgent):
    """
    Abstract base class for Retrieval-Augmented Generation (RAG) agents.

    This class extends BaseAgent with methods specific to RAG, including
    retrieval and query methods.
    """

    def __init__(self, **kwargs):
        """
        Initialize a RAGAgent object.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def retrieve(self, query: str, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        """
        Retrieve document results from the knowledge base.

        Args:
            query: The query string.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple containing:
                - the retrieved results
                - the total number of token usages of the LLM
                - any additional metadata, which can be an empty dictionary
        """

    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """
        Query the agent and return the answer.

        Args:
            query: The query string.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple containing:
                - the result generated from LLM
                - the retrieved document results
                - the total number of token usages of the LLM
        """
