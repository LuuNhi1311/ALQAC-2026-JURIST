# 💻 Command Line Interface

DeepSearcher provides a convenient command line interface for loading data and querying.

## 📥 Loading Data

Load data from files or URLs:

```shell
deepsearcher load "your_local_path_or_url"
```

Load into a specific collection:

```shell
deepsearcher load "your_local_path_or_url" --collection_name "your_collection_name" --collection_desc "your_collection_description"
```

### Examples

#### Loading from local files:

```shell
# Load a single file
deepsearcher load "/path/to/your/local/file.pdf"

# Load multiple files at once
deepsearcher load "/path/to/your/local/file1.pdf" "/path/to/your/local/file2.md"
```

#### Loading from URL:

> **Note:** Set `FIRECRAWL_API_KEY` in your environment variables. See [FireCrawl documentation](https://docs.firecrawl.dev/introduction) for more details.

```shell
deepsearcher load "https://www.wikiwand.com/en/articles/DeepSeek"
```

## 🔍 Querying Data

Query your loaded data:

```shell
deepsearcher query "Write a report about xxx."
```

## ❓ Help Commands

Get general help information:

```shell
deepsearcher --help
```

Get help for specific subcommands:

```shell
# Help for load command
deepsearcher load --help

# Help for query command
deepsearcher query --help
``` 