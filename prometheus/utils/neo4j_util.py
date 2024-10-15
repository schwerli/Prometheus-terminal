import neo4j

def format_neo4j_result(result: neo4j.Result) -> str:
  data = result.data()
  output = ""
  for index, row_result in enumerate(data):
    output += f"Result {index+1}:\n"
    for key in sorted(row_result.keys()):
      output += f"{key}: {row_result[key]}\n"
    output += "\n\n"
  return output.strip()