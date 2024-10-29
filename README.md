![code coverage](https://github.com/Pantheon-temple/Prometheus/raw/coverage-badge/coverage.svg)

# How to run the project

First you need to start the Neo4j container for storing the knowledge graph.

```bash
docker run \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["apoc"]' \
  -v ./data_neo4j:/data \
  neo4j:5.20.0
```

```bash
docker run \
  -p 5432:5432 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -v ./data_postgres:/var/lib/postgresql/data \
  postgres
```