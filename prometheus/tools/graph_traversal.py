from langchain.pydantic_v1 import BaseModel, Field

from neo4j import GraphDatabase

from prometheus.utils import neo4j_util

MAX_RESULT = 100

class FindFileNodeWithBasenameInput(BaseModel):
  basename: str = Field("The basename of FileNode to search for")

def find_file_node_with_basename(basename: str, driver: GraphDatabase.driver) -> str:
  def _find_file_node_with_basename(tx):
    result = tx.run(
      f"""\
        MATCH (f:FileNode {{ basename: '{basename}' }})
        RETURN f AS FileNode
        ORDER BY f.node_id
        LIMIT {MAX_RESULT}
      """
    )

    return neo4j_util.format_neo4j_result(result)
  
  with driver.session() as session:
    return session.execute_read(_find_file_node_with_basename)
  
class FindASTNodeWithText(BaseModel):
  text: str = Field("Search ASTNode that exactly contains this text.")

def find_ast_node_with_text(text: str, driver: GraphDatabase.driver) -> str:
  def _find_ast_node_with_text(tx):
    result = tx.run(
      f"""\
        MATCH (f:FileNode) -[:HAS_AST]-> (:ASTNode) -[:PARENT_OF*]-> (a:ASTNode)
        WHERE a.text CONTAINS '{text}'
        RETURN f as FileNode, a AS ASTNode
        ORDER BY SIZE(a.text)
        LIMIT {MAX_RESULT}
      """
    )

    return neo4j_util.format_neo4j_result(result)
  
  with driver.session() as session:
    return session.execute_read(_find_ast_node_with_text)
  
class FindASTNodeWithType(BaseModel):
  type: str = Field("Search ASTNode that has this tree-sitter node type.")

def find_ast_node_with_type(type: str, driver: GraphDatabase.driver) -> str:
  def _find_ast_node_with_type(tx):
    result = tx.run(
      f"""\
        MATCH (f:FileNode) -[:HAS_AST]-> (:ASTNode) -[:PARENT_OF*]-> (a:ASTNode {{ type: '{type}' }})
        RETURN f as FileNode, a AS ASTNode
        ORDER a.node_id
        LIMIT {MAX_RESULT}
      """
    )

    return neo4j_util.format_neo4j_result(result)
  
  with driver.session() as session:
    return session.execute_read(_find_ast_node_with_type)
  
class FindTextNodeWithText(BaseModel):
  text: str = Field("Search TextNode that exactly contains this text.")

def find_text_node_with_text(text: str, driver: GraphDatabase.driver) -> str:
  def _find_text_node_with_text(tx):
    result = tx.run(
      f"""\
        MATCH (f:FileNode) -[:HAS_TEXT]-> (t:TextNode)
        WHERE t.text CONTAINS '{text}'
        RETURN f as FileNode, t AS TextNode
        ORDER t.node_id
        LIMIT {MAX_RESULT}
      """
    )

    return neo4j_util.format_neo4j_result(result)
  
  with driver.session() as session:
    return session.execute_read(_find_text_node_with_text)