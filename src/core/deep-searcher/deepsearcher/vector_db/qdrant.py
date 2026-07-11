import uuid
from typing import List, Optional, Union

import numpy as np

from deepsearcher.loader.splitter import Chunk
from deepsearcher.utils import log
from deepsearcher.vector_db.base import BaseVectorDB, CollectionInfo, RetrievalResult

DEFAULT_COLLECTION_NAME = "deepsearcher"

TEXT_PAYLOAD_KEY = "text"
REFERENCE_PAYLOAD_KEY = "reference"
METADATA_PAYLOAD_KEY = "metadata"


class Qdrant(BaseVectorDB):
    """Vector DB implementation powered by [Qdrant](https://qdrant.tech/)"""

    def __init__(
        self,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[int] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        default_collection: str = DEFAULT_COLLECTION_NAME,
    ):
        """
        Initialize the Qdrant client with flexible connection options.

        Args:
            location (Optional[str], optional):
                - If ":memory:" - use in-memory Qdrant instance.
                - If str - use it as a URL parameter.
                - If None - use default values for host and port.
                Defaults to None.

            url (Optional[str], optional):
                URL for Qdrant service, can include scheme, host, port, and prefix.
                Allows flexible connection string specification.
                Defaults to None.

            port (Optional[int], optional):
                Port of the REST API interface.
                Defaults to 6333.

            grpc_port (int, optional):
                Port of the gRPC interface.
                Defaults to 6334.

            prefer_grpc (bool, optional):
                If True, use gRPC interface whenever possible in custom methods.
                Defaults to False.

            https (Optional[bool], optional):
                If True, use HTTPS (SSL) protocol.
                Defaults to None.

            api_key (Optional[str], optional):
                API key for authentication in Qdrant Cloud.
                Defaults to None.

            prefix (Optional[str], optional):
                If not None, add prefix to the REST URL path.
                Example: 'service/v1' results in 'http://localhost:6333/service/v1/{qdrant-endpoint}'
                Defaults to None.

            timeout (Optional[int], optional):
                Timeout for REST and gRPC API requests.
                Default is 5 seconds for REST and unlimited for gRPC.
                Defaults to None.

            host (Optional[str], optional):
                Host name of Qdrant service.
                If url and host are None, defaults to 'localhost'.
                Defaults to None.

            path (Optional[str], optional):
                Persistence path for QdrantLocal.
                Defaults to None.

            default_collection (str, optional):
                Default collection name to be used.
        """
        try:
            from qdrant_client import QdrantClient
        except ImportError as original_error:
            raise ImportError(
                "Qdrant client is not installed. Install it using: pip install qdrant-client\n"
            ) from original_error

        super().__init__(default_collection)
        self.client = QdrantClient(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
        )

    def init_collection(
        self,
        dim: int,
        collection: Optional[str] = None,
        description: Optional[str] = "",
        force_new_collection: bool = False,
        text_max_length: int = 65_535,
        reference_max_length: int = 2048,
        distance_metric: str = "Cosine",
        *args,
        **kwargs,
    ):
        """
        Initialize a collection in Qdrant.

        Args:
            dim (int): Dimension of the vector embeddings.
            collection (Optional[str], optional): Collection name.
            description (Optional[str], optional): Collection description. Defaults to "".
            force_new_collection (bool, optional): Whether to force create a new collection if it already exists. Defaults to False.
            text_max_length (int, optional): Maximum length for text field. Defaults to 65_535.
            reference_max_length (int, optional): Maximum length for reference field. Defaults to 2048.
            distance_metric (str, optional): Metric type for vector similarity search. Defaults to "Cosine".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        from qdrant_client import models

        collection = collection or self.default_collection

        try:
            collection_exists = self.client.collection_exists(collection_name=collection)

            if force_new_collection and collection_exists:
                self.client.delete_collection(collection_name=collection)
                collection_exists = False

            if not collection_exists:
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(size=dim, distance=distance_metric),
                    *args,
                    **kwargs,
                )

                log.color_print(f"Created collection [{collection}] successfully")
        except Exception as e:
            log.critical(f"Failed to init Qdrant collection, error info: {e}")

    def insert_data(
        self,
        collection: Optional[str],
        chunks: List[Chunk],
        batch_size: int = 256,
        *args,
        **kwargs,
    ):
        """
        Insert data into a Qdrant collection.

        Args:
            collection (Optional[str]): Collection name.
            chunks (List[Chunk]): List of Chunk objects to insert.
            batch_size (int, optional): Number of chunks to insert in each batch. Defaults to 256.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        from qdrant_client import models

        try:
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]

                points = [
                    models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector=chunk.embedding,
                        payload={
                            TEXT_PAYLOAD_KEY: chunk.text,
                            REFERENCE_PAYLOAD_KEY: chunk.reference,
                            METADATA_PAYLOAD_KEY: chunk.metadata,
                        },
                    )
                    for chunk in batch_chunks
                ]

                self.client.upsert(
                    collection_name=collection or self.default_collection, points=points
                )
        except Exception as e:
            log.critical(f"Failed to insert data, error info: {e}")

    def search_data(
        self,
        collection: Optional[str],
        vector: Union[np.array, List[float]],
        top_k: int = 5,
        *args,
        **kwargs,
    ) -> List[RetrievalResult]:
        """
        Search for similar vectors in a Qdrant collection.

        Args:
            collection (Optional[str]): Collection name..
            vector (Union[np.array, List[float]]): Query vector for similarity search.
            top_k (int, optional): Number of results to return. Defaults to 5.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[RetrievalResult]: List of retrieval results containing similar vectors.
        """
        try:
            results = self.client.query_points(
                collection_name=collection or self.default_collection,
                query=vector,
                limit=top_k,
                with_payload=True,
                with_vectors=True,
            ).points

            return [
                RetrievalResult(
                    embedding=result.vector,
                    text=result.payload.get(TEXT_PAYLOAD_KEY, ""),
                    reference=result.payload.get(REFERENCE_PAYLOAD_KEY, ""),
                    score=result.score,
                    metadata=result.payload.get(METADATA_PAYLOAD_KEY, {}),
                )
                for result in results
            ]
        except Exception as e:
            log.critical(f"Failed to search data, error info: {e}")
            return []

    def list_collections(self, *args, **kwargs) -> List[CollectionInfo]:
        """
        List all collections in the Qdrant database.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[CollectionInfo]: List of collection information objects.
        """
        collection_infos = []

        try:
            collections = self.client.get_collections().collections
            for collection in collections:
                collection_infos.append(
                    CollectionInfo(
                        collection_name=collection.name,
                        # Qdrant doesn't have a native description field
                        description=collection.name,
                    )
                )
        except Exception as e:
            log.critical(f"Failed to list collections, error info: {e}")

        return collection_infos

    def clear_db(self, collection: Optional[str] = None, *args, **kwargs):
        """
        Clear (drop) a collection from the Qdrant database.

        Args:
            collection (str, optional): Collection name to drop.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        try:
            self.client.delete_collection(collection_name=collection or self.default_collection)
        except Exception as e:
            log.warning(f"Failed to drop collection, error info: {e}")
