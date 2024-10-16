import neo4j


def format_neo4j_result(result: neo4j.Result) -> str:
  """Format a Neo4j result into a string.

  Args:
    result: The result from a Neo4j query.

  Returns:
    A string representation of the result.
  """
  data = result.data()
  output = ""
  for index, row_result in enumerate(data):
    output += f"Result {index+1}:\n"
    for key in sorted(row_result.keys()):
      output += f"{key}: {row_result[key]}\n"
    output += "\n\n"
  return output.strip()


def run_neo4j_query(query: str, driver: neo4j.GraphDatabase.driver) -> str:
  """Run a read-only Neo4j query and format the result into a string.

  Args:
    query: The query to run.
    driver: The Neo4j driver to use.

  Returns:
    A string representation of the result.
  """

  def query_transaction(tx):
    result = tx.run(query)
    return format_neo4j_result(result)

  with driver.session() as session:
    return session.execute_read(query_transaction)
