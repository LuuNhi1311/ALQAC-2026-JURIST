# File Loader Configuration

DeepSearcher supports various file loaders to extract and process content from different file formats.

## 📝 Basic Configuration

```python
config.set_provider_config("file_loader", "(FileLoaderName)", "(Arguments dict)")
```

## 📋 Available File Loaders

| Loader | Description | Supported Formats |
|--------|-------------|-------------------|
| **UnstructuredLoader** | General purpose document loader with broad format support | PDF, DOCX, PPT, HTML, etc. |
| **DoclingLoader** | Document processing library with extraction capabilities | See [documentation](https://docling-project.github.io/docling/usage/supported_formats/) |

## 🔍 File Loader Options

### Unstructured

[Unstructured](https://unstructured.io/) is a powerful library for extracting content from various document formats.

```python
config.set_provider_config("file_loader", "UnstructuredLoader", {})
```

??? tip "Setup Instructions"

    You can use Unstructured in two ways:

    1. **With API** (recommended for production)
       - Set environment variables:
         - `UNSTRUCTURED_API_KEY`
         - `UNSTRUCTURED_API_URL`

    2. **Local Processing**
       - Simply don't set the API environment variables
       - Install required dependencies:
         ```bash
         # Install core dependencies
         pip install unstructured-ingest
         
         # For all document formats
         pip install "unstructured[all-docs]"
         
         # For specific formats (e.g., PDF only)
         pip install "unstructured[pdf]"
         ```

    For more information:
    - [Unstructured Documentation](https://docs.unstructured.io/ingestion/overview)
    - [Installation Guide](https://docs.unstructured.io/open-source/installation/full-installation)

### Docling

[Docling](https://docling-project.github.io/docling/) provides document processing capabilities with support for multiple formats.

```python
config.set_provider_config("file_loader", "DoclingLoader", {})
```

??? tip "Setup Instructions"

    1. Install Docling:
       ```bash
       pip install docling
       ```

    2. For information on supported formats, see the [Docling documentation](https://docling-project.github.io/docling/usage/supported_formats/#supported-output-formats). 