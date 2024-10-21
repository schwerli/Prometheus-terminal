from testcontainers.neo4j import Neo4jContainer


def clean_neo4j_container(neo4j_container: Neo4jContainer):
  with neo4j_container.get_driver() as driver:
    with driver.session() as session:
      session.run("MATCH (n) DETACH DELETE n")
