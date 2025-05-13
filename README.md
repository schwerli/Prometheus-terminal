![Code Coverage](https://github.com/Pantheon-temple/Prometheus/raw/coverage-badge/coverage.svg)

# Prometheus

Prometheus is a FastAPI-based backend service designed to perform intelligent codebase-level operations, including
answering questions, resolving issues, and reviewing pull requests. At its core, it implements a multi-agent approach
governed by a state machine to ensure code quality through automated reviews, build verification, and test execution.

## Features

- **Codebase Analysis**: Answer questions about your codebase and provide insights.
- **Issue Resolution**: Automatically resolve issues in your repository.
- **Pull Request Reviews**: Perform intelligent reviews of pull requests to ensure code quality.
- **Multi-Agent System**: Uses a state machine to coordinate multiple agents for efficient task execution.
- **Integration with External Services**: Seamlessly connects with other services in the `Pantheon-temple` organization.

## Integrations

Prometheus can be connected to other services for extended functionality:

- **[Argus-GitHub](https://github.com/Pantheon-temple/Argus-GitHub)**: Automatically pull the latest changes from your
  GitHub repository, answer/fix issues, and review pull requests.
- **[Hermes](https://github.com/Pantheon-temple/Hermes)**: Enable a chat interface to interact with your codebase (
  GitHub or local).
- **Custom Services**: Connect your own services by sending requests to the FastAPI endpoints.

## Quick Start

### Prerequisites

- Docker
- Docker Compose
- API keys for external services (e.g., OpenAI, Anthropic, etc.)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Pantheon-temple/Prometheus.git
   cd Prometheus
   ```

2. Copy the `example.env` file to `.env` and update it with your API keys and other required configurations:
   ```bash
   mv example.env .env
   ```

3. Start the services using Docker Compose:
    - For Linux:
      ```bash
      docker-compose up --build
      ```
    - For Windows or macOS:
      ```bash
      docker-compose -f docker-compose.win_mac.yml up --build
      ```

4. Access Prometheus:
    - Service: [http://localhost:9001](http://localhost:9001)
    - OpenAPI Docs: [http://localhost:9001/docs](http://localhost:9001/docs)

### Database Setup

#### PostgreSQL

Run the following command to start a PostgreSQL container:

```bash
docker run -d \
  -p 5432:5432 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=postgres \
  postgres
```

#### Neo4j

Run the following command to start a Neo4j container:

```bash
docker run -d \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["apoc"]' \
  -e NEO4J_dbms_memory_heap_initial__size=4G \
  -e NEO4J_dbms_memory_heap_max__size=8G \
  -e NEO4J_dbms_memory_pagecache_size=4G \
  neo4j
```

Verify Neo4J at: [http://localhost:7474](http://localhost:7474)

## Configuration

Prometheus requires the following environment variables to be set in the `.env` file:

- **Neo4j**:
    - `PROMETHEUS_NEO4J_URI`
    - `PROMETHEUS_NEO4J_USERNAME`
    - `PROMETHEUS_NEO4J_PASSWORD`
- **LLM Models**:
    - `PROMETHEUS_ADVANCED_MODEL`
    - `PROMETHEUS_BASE_MODEL`
    - API keys: `PROMETHEUS_OPENAI_API_KEY`, `PROMETHEUS_ANTHROPIC_API_KEY`, `PROMETHEUS_GEMINI_API_KEY`,
      `PROMETHEUS_OPENROUTER_API_KEY`
- **Other Settings**:
    - `PROMETHEUS_WORKING_DIRECTORY`
    - `PROMETHEUS_GITHUB_ACCESS_TOKEN`
    - `PROMETHEUS_KNOWLEDGE_GRAPH_MAX_AST_DEPTH`
    - `PROMETHEUS_NEO4J_BATCH_SIZE`

## Development

To contribute to Prometheus, follow these steps:

#### Prerequisites
* Python >= 3.11


1. Install dependencies:
   ```bash
   pip install hatchling
   pip install .
   ```

2. Run tests:
   ```bash
   pytest
   ```

3. Start the development server:
   ```bash
   uvicorn prometheus.app.main:app --host 0.0.0.0 --port 9001
   ```

## License

Prometheus is licensed under the [MIT License](LICENSE).

## Contact

For questions or support, please open an issue in
the [GitHub repository](https://github.com/Pantheon-temple/Prometheus/issues).
