![code coverage](https://github.com/Pantheon-temple/Prometheus/raw/coverage-badge/coverage.svg)

Prometheus is a FastAPI backend service that performs intelligent codebase-level operations including answering questions, resolving issues, and reviewing pull requests. At its core, it implements a multi-agent approach governed by a state machine that ensures code quality through automated reviews, build verification, and test execution.

Prometheus can be connected to other services provided in the `Pantheon-temple` orgnization, for example:
* Connect it to `Pantheon-temple/Argus-GitHub` to automatically pull the latest changes from your GitHub repository, answer/fix issues, and review pull requests.
* Connect it to `Pantheon-temple/Hermes` to have chat interface with your codebase (GitHub or local).
* Connect it your own service, by sending requests to the FastAPI endpoints.

# Quick start

In this project we use `docker-compose.yml`(for Linux), or `docker-compose.win_mac.yml` (for Windows or MacOS). You should update the `example_settings.toml` with the API keys, and then rename the file to `settings.toml`.

Now, simply run `docker compose up`, and you can access Promtheus at `http://localhost:9001` and the OpenAPI docs at `http://localhost:9001/docs`


```
docker run -d \
  -p 5432:5432 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=postgres \
  postgres
```

```
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