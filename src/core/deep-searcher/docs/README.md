# DeepSearcher Documentation

This directory contains the documentation for DeepSearcher, powered by MkDocs.

## Setup

1. Install MkDocs and required plugins:

```bash
pip install mkdocs mkdocs-material mkdocs-jupyter pymdown-extensions
```

2. Clone the repository:

```bash
git clone https://github.com/zilliztech/deep-searcher.git
cd deep-searcher
```

## Development

To serve the documentation locally:

```bash
mkdocs serve
```

This will start a local server at http://127.0.0.1:8000/ where you can preview the documentation.

## Building

To build the static site:

```bash
mkdocs build
```

This will generate the static site in the `site` directory.

## Deployment

The documentation is automatically deployed when changes are pushed to the main branch using GitHub Actions. 