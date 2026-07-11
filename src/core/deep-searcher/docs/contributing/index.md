# Contributing to DeepSearcher

We welcome contributions from everyone. This document provides guidelines to make the contribution process straightforward.


## Pull Request Process

1. Fork the repository and create your branch from `master`.
2. Make your changes.
3. Run tests and linting to ensure your code meets the project's standards.
4. Update documentation if necessary.
5. Submit a pull request.


## Linting and Formatting

Keeping a consistent style for code, code comments, commit messages, and PR descriptions will greatly accelerate your PR review process.
We require you to run code linter and formatter before submitting your pull requests:

To check the coding styles:

```shell
make lint
```

To fix the coding styles:

```shell
make format
```
Our CI pipeline also runs these checks automatically on all pull requests to ensure code quality and consistency.


## Development Environment Setup with uv

DeepSearcher uses [uv](https://github.com/astral-sh/uv) as the recommended package manager. uv is a fast, reliable Python package manager and installer. The project's `pyproject.toml` is configured to work with uv, which will provide faster dependency resolution and package installation compared to traditional tools.

### Install Project in Development Mode(aka Editable Installation)

1. Install uv if you haven't already:
   Follow the [offical installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

2. Clone the repository and navigate to the project directory:
   ```shell
   git clone https://github.com/zilliztech/deep-searcher.git && cd deep-searcher
   ```
3. Synchronize and install dependencies:
   ```shell
   uv sync
   source .venv/bin/activate
   ```
   `uv sync` will install all dependencies specified in `uv.lock` file. And the `source .venv/bin/activate` command will activate the virtual environment.

   - (Optional) To install all optional dependencies:
      ```shell
      uv sync --all-extras --dev
      ```

   - (Optional) To install specific optional dependencies:
      ```shell
      # Take optional `ollama` dependency for example
      uv sync --extra ollama
      ```
   For more optional dependencies, refer to the `[project.optional-dependencies]` part of `pyproject.toml` file.



### Adding Dependencies

When you need to add new dependencies to the `pyproject.toml` file, you can use the following commands:

```shell
uv add <package_name>
```
DeepSearcher uses optional dependencies to keep the default installation lightweight. Optional features can be installed using the syntax `deepsearcher[<extra>]`. To add a dependency to an optional extra, use the following command:

```shell
uv add <package_name> --optional <extra>
```
For more details, refer to the [offical Managing dependencies documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/).

### Dependencies Locking

For development, we use lockfiles to ensure consistent dependencies. You can use 
```shell
uv lock --check
```
to verify if your lockfile is up-to-date with your project dependencies.

When you modify or add dependencies in the project, the lockfile will be automatically updated the next time you run a uv command. You can also explicitly update the lockfile using:
```shell
uv lock
```

While the environment is synced automatically, it may also be explicitly synced using uv sync:
```shell
uv sync
```
Syncing the environment manually is especially useful for ensuring your editor has the correct versions of dependencies.


For more detailed information about dependency locking and syncing, refer to the [offical Locking and syncing documentation](https://docs.astral.sh/uv/concepts/projects/sync/).


## Running Tests

Before submitting your pull request, make sure to run the test suite to ensure your changes haven't introduced any regressions.

### Installing Test Dependencies

First, ensure you have pytest installed. If you haven't installed the development dependencies yet, you can do so with:

```shell
uv sync --all-extras --dev
```

This will install all development dependencies and optional dependencies including pytest and other testing tools.

### Running the Tests

To run all tests in the `tests` directory:

```shell
uv run pytest tests
```

For more verbose output that shows individual test results:

```shell
uv run pytest tests -v
```

You can also run tests for specific directories or files. For example:

```shell
# Run tests in a specific directory
uv run pytest tests/embedding

# Run tests in a specific file
uv run pytest tests/embedding/test_bedrock_embedding.py

# Run a specific test class
uv run pytest tests/embedding/test_bedrock_embedding.py::TestBedrockEmbedding

# Run a specific test method
uv run pytest tests/embedding/test_bedrock_embedding.py::TestBedrockEmbedding::test_init_default
```

The `-v` flag (verbose mode) provides more detailed output, showing each test case and its result individually. This is particularly useful when you want to see which specific tests are passing or failing.


## Developer Certificate of Origin (DCO)

All contributions require a sign-off, acknowledging the [Developer Certificate of Origin](https://developercertificate.org/). 
Add a `Signed-off-by` line to your commit message:

```text
Signed-off-by: Your Name <your.email@example.com>
``` 