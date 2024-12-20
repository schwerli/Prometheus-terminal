import neo4j

from prometheus.utils.str_util import truncate_text


def format_neo4j_result(result: neo4j.Result, max_token_per_result: int) -> str:
  """Format a Neo4j result into a string.

  Args:
    result: The result from a Neo4j query.
    max_token_per_result: Maximum number of tokens per result.

  Returns:
    A string representation of the result.
  """
  data = result.data()
  output = ""
  for index, row_result in enumerate(data):
    output += f"Result {index+1}:\n"
    for key in sorted(row_result.keys()):
      output += f"{key}: {str(row_result[key])}\n"
    output += "\n\n"
  return truncate_text(output.strip(), max_token_per_result)


def run_neo4j_query(
  query: str, driver: neo4j.GraphDatabase.driver, max_token_per_result: int
) -> str:
  """Run a read-only Neo4j query and format the result into a string.

  Args:
    query: The query to run.
    driver: The Neo4j driver to use.
    max_token_per_result: Maximum number of tokens per result.

  Returns:
    A string representation of the result.
  """

  def query_transaction(tx):
    result = tx.run(query)
    return format_neo4j_result(result, max_token_per_result)

  with driver.session() as session:
    return session.execute_read(query_transaction)
