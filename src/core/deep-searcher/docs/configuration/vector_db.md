# Vector Database Configuration

DeepSearcher uses vector databases to store and retrieve document embeddings for efficient semantic search.

## 📝 Basic Configuration

```python
config.set_provider_config("vector_db", "(VectorDBName)", "(Arguments dict)")
```

Currently supported vector databases:
- Milvus (including Milvus Lite and Zilliz Cloud)

## 🔍 Milvus Configuration

```python
config.set_provider_config("vector_db", "Milvus", {"uri": "./milvus.db", "token": ""})
```

### Deployment Options

??? example "Local Storage with Milvus Lite"

    Setting the `uri` as a local file (e.g., `./milvus.db`) automatically utilizes [Milvus Lite](https://milvus.io/docs/milvus_lite.md) to store all data in this file. This is the most convenient method for development and smaller datasets.

    ```python
    config.set_provider_config("vector_db", "Milvus", {"uri": "./milvus.db", "token": ""})
    ```

??? example "Standalone Milvus Server"

    For larger datasets, you can set up a more performant Milvus server using [Docker or Kubernetes](https://milvus.io/docs/quickstart.md). In this setup, use the server URI as your `uri` parameter:

    ```python
    config.set_provider_config("vector_db", "Milvus", {"uri": "http://localhost:19530", "token": ""})
    ```

    Also, you could specify other connection parameters supported by Milvus such as `user`, `password`, `secure` or others.
    ```python
    config.set_provider_config("vector_db", "Milvus", {"uri": "http://localhost:19530", "user": "<username>", "password":"<password>", "secure": True, "token": ""})
    ```

??? example "Zilliz Cloud (Managed Service)"

    [Zilliz Cloud](https://zilliz.com/cloud) provides a fully managed cloud service for Milvus. To use Zilliz Cloud, adjust the `uri` and `token` according to the [Public Endpoint and API Key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#free-cluster-details):

    ```python
    config.set_provider_config("vector_db", "Milvus", {
        "uri": "https://your-instance-id.api.gcp-us-west1.zillizcloud.com", 
        "token": "your_api_key"
    })
    ``` 