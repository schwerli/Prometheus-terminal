"""Service for managing Neo4j database driver."""

from neo4j import GraphDatabase


class Neo4jService:
    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str):
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password),
            connection_timeout=300,
            max_transaction_retry_time=300,
            keep_alive=True,
        )

    def close(self):
        self.neo4j_driver.close()
