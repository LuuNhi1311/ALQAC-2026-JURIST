# 🌐 Deployment

This guide explains how to deploy DeepSearcher as a web service.

## ⚙️ Configure Modules

You can configure all arguments by modifying the configuration file:

```yaml
# config.yaml - https://github.com/zilliztech/deep-searcher/blob/main/config.yaml
llm:
  provider: "OpenAI"
  api_key: "your_openai_api_key_here"
  # Additional configuration options...
```

> **Important:** Set your `OPENAI_API_KEY` in the `llm` section of the YAML file.

## 🚀 Start Service

The main script will run a FastAPI service with default address `localhost:8000`:

```shell
$ python main.py
```

Once started, you should see output indicating the service is running successfully.

## 🔍 Access via Browser

You can access the web service through your browser:

1. Open your browser and navigate to [http://localhost:8000/docs](http://localhost:8000/docs)
2. The Swagger UI will display all available API endpoints
3. Click the "Try it out" button on any endpoint to interact with it
4. Fill in the required parameters and execute the request

This interactive documentation makes it easy to test and use all DeepSearcher API functionality.

## 🐳 Docker Deployment

You can also deploy DeepSearcher using Docker for easier environment setup and management.

### Build Docker Image

To build the Docker image, run the following command from the project root directory:

```shell
docker build -t deepsearcher:latest .
```

This command builds a Docker image using the Dockerfile in the current directory and tags it as `deepsearcher:latest`.

### Run Docker Container

Once the image is built, you can run it as a container:

```shell
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_openai_api_key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/deepsearcher/config.yaml:/app/deepsearcher/config.yaml \
  deepsearcher:latest
```

This command:
- Maps port 8000 from the container to port 8000 on your host
- Sets the `OPENAI_API_KEY` environment variable
- Mounts the local `data`, `logs`, and configuration file to the container
- Runs the previously built `deepsearcher:latest` image

> **Note:** Replace `your_openai_api_key` with your actual OpenAI API key, or set any other environment variables required for your configuration. 