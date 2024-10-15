![code coverage](https://github.com/Pantheon-temple/Prometheus/raw/coverage-badge/coverage.svg)

# How to run the project

First you need to start the Neo4j container for storing the knowledge graph.

```bash
docker run \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/neo4j \
  -e NEO4JLABS_PLUGINS='["apoc"]' \
  neo4j:5.20.0
```