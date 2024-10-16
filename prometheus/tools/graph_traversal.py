from pathlib import Path
from pydantic import BaseModel, Field

from neo4j import GraphDatabase

from prometheus.parser import tree_sitter_parser
from prometheus.utils import neo4j_util

MAX_RESULT = 20

###############################################################################
#                          FileNode retrieval                                 #
###############################################################################


class FindFileNodeWithBasenameInput(BaseModel):
  basename: str = Field("The basename of FileNode to search for")


def find_file_node_with_basename(basename: str, driver: GraphDatabase.driver) -> str:
  query = f"""
      MATCH (f:FileNode {{ basename: '{basename}' }})
      RETURN f AS FileNode
      ORDER BY f.node_id
      LIMIT {MAX_RESULT}
  """
  return neo4j_util.run_neo4j_query(query, driver)


class FindFileNodeWithRelativePathInput(BaseModel):
  relative_path: str = Field("The relative_path of FileNode to search for")


def find_file_node_with_relative_path(
  relative_path: str, driver: GraphDatabase.driver
) -> str:
  query = f"""\
    MATCH (f:FileNode {{ relative_path: '{relative_path}' }})
    RETURN f AS FileNode
    ORDER BY f.node_id
    LIMIT {MAX_RESULT}
  """
  return neo4j_util.run_neo4j_query(query, driver)


###############################################################################
#                          ASTNode retrieval                                  #
###############################################################################


class FindASTNodeWithTextInput(BaseModel):
  text: str = Field("Search ASTNode that exactly contains this text.")


def find_ast_node_with_text(text: str, driver: GraphDatabase.driver) -> str:
  query = f"""\
    MATCH (f:FileNode) -[:HAS_AST]-> (:ASTNode) -[:PARENT_OF*]-> (a:ASTNode)
    WHERE a.text CONTAINS '{text}'
    RETURN f as FileNode, a AS ASTNode
    ORDER BY SIZE(a.text)
    LIMIT {MAX_RESULT}
  """
  return neo4j_util.run_neo4j_query(query, driver)


class FindASTNodeWithTypeInput(BaseModel):
  type: str = Field("Search ASTNode that has this tree-sitter node type.")


def find_ast_node_with_type(type: str, driver: GraphDatabase.driver) -> str:
  query = f"""\
    MATCH (f:FileNode) -[:HAS_AST]-> (:ASTNode) -[:PARENT_OF*]-> (a:ASTNode {{ type: '{type}' }})
    RETURN f as FileNode, a AS ASTNode
    ORDER BY a.node_id
    LIMIT {MAX_RESULT}
  """
  return neo4j_util.run_neo4j_query(query, driver)


class FindASTNodeWithTextInFileInput(BaseModel):
  text: str = Field("Search ASTNode that exactly contains this text.")
  basename: str = Field("The basename of FileNode to search ASTNode.")


def find_ast_node_with_text_in_file(
  text: str, basename: str, driver: GraphDatabase.driver
) -> str:
  query = f"""\
    MATCH (f:FileNode) -[:HAS_AST]-> (:ASTNode) -[:PARENT_OF*]-> (a:ASTNode)
    WHERE f.basename = '{basename}' AND a.text CONTAINS '{text}'
    RETURN f as FileNode, a AS ASTNode
    ORDER BY SIZE(a.text)
    LIMIT {MAX_RESULT}
  """
  return neo4j_util.run_neo4j_query(query, driver)


class FindASTNodeWithTypeInFileInput(BaseModel):
  type: str = Field("Search ASTNode with this tree-sitter node type.")
  basename: str = Field("The basename of FileNode to search ASTNode.")


def find_ast_node_with_type_in_file(
  type: str, basename: str, driver: GraphDatabase.driver
) -> str:
  query = f"""\
    MATCH (f:FileNode) -[:HAS_AST]-> (:ASTNode) -[:PARENT_OF*]-> (a:ASTNode)
    WHERE f.basename = '{basename}' AND a.type = '{type}'
    RETURN f as FileNode, a AS ASTNode
    ORDER BY SIZE(a.text)
    LIMIT {MAX_RESULT}
  """
  return neo4j_util.run_neo4j_query(query, driver)


class FindASTNodeWithTypeAndTextInput(BaseModel):
  type: str = Field("Search ASTNode with this tree-sitter node type.")
  text: str = Field("Search ASTNode that exactly contains this text.")


def find_ast_node_with_type_and_text(
  type: str, text: str, driver: GraphDatabase.driver
) -> str:
  query = f"""\
    MATCH (f:FileNode) -[:HAS_AST]-> (:ASTNode) -[:PARENT_OF*]-> (a:ASTNode)
    WHERE a.type = '{type}' AND a.text CONTAINS '{text}'
    RETURN f as FileNode, a AS ASTNode
    ORDER BY SIZE(a.text)
    LIMIT {MAX_RESULT}
  """
  return neo4j_util.run_neo4j_query(query, driver)


###############################################################################
#                          TextNode retrieval                                 #
###############################################################################


class FindTextNodeWithTextInput(BaseModel):
  text: str = Field("Search TextNode that exactly contains this text.")


def find_text_node_with_text(text: str, driver: GraphDatabase.driver) -> str:
  query = f"""\
    MATCH (f:FileNode) -[:HAS_TEXT]-> (t:TextNode)
    WHERE t.text CONTAINS '{text}'
    RETURN f as FileNode, t AS TextNode
    ORDER BY t.node_id
    LIMIT {MAX_RESULT}
  """
  return neo4j_util.run_neo4j_query(query, driver)


class FindTextNodeWithTextInFileInput(BaseModel):
  text: str = Field("Search TextNode that exactly contains this text.")
  basename: str = Field("The basename of FileNode to search TextNode.")


def find_text_node_with_text_in_file(
  text: str, basename: str, driver: GraphDatabase.driver
) -> str:
  query = f"""\
    MATCH (f:FileNode) -[:HAS_TEXT]-> (t:TextNode)
    WHERE f.basename = '{basename}' AND t.text CONTAINS '{text}'
    RETURN f as FileNode, t AS TextNode
    ORDER BY t.node_id
    LIMIT {MAX_RESULT}
  """
  return neo4j_util.run_neo4j_query(query, driver)


class GetNextTextNodeWithNodeIdInput(BaseModel):
  node_id: int = Field("Get the next TextNode of this given node_id.")


def get_next_text_node_with_node_id(node_id: str, driver: GraphDatabase.driver) -> str:
  query = f"""\
    MATCH (a:TextNode {{ node_id: {node_id} }}) -[:NEXT_CHUNK]-> (b:TextNode)
    RETURN b as TextNode
  """
  return neo4j_util.run_neo4j_query(query, driver)


###############################################################################
#                                 Other                                       #
###############################################################################


class PreviewFileContentWithBasenameInput(BaseModel):
  basename: str = Field("The basename of FileNode to preview.")


def preview_file_content_with_basename(
  basename: str, driver: GraphDatabase.driver
) -> str:
  source_code_query = f"""\
    MATCH (f:FileNode {{ basename: '{basename}' }}) -[:HAS_AST]-> (a:ASTNode)
    WITH f, apoc.text.split(a.text, '\\R') AS lines
    RETURN f as FileNode, apoc.text.join(lines[0..300], '\\n') AS preview
    ORDER BY f.node_id
  """

  text_query = f"""\
    MATCH (f:FileNode {{ basename: '{basename}' }}) -[:HAS_TEXT]-> (t:TextNode)
    WHERE NOT EXISTS((:TextNode) -[:NEXT_CHUNK]-> (t))
    RETURN f as FileNode, t.text AS preview
    ORDER BY f.node_id
  """

  if tree_sitter_parser.supports_file(Path(basename)):
    return neo4j_util.run_neo4j_query(source_code_query, driver)
  return neo4j_util.run_neo4j_query(text_query, driver)


class GetParentNodeInput(BaseModel):
  node_id: str = Field(description="Get parent node of node with this node_id")


def get_parent_node(node_id: int, driver: GraphDatabase.driver) -> str:
  query = f"""\
    MATCH (p) -[r]-> (c {{ node_id: {node_id} }})
    WHERE type(r) IN ['HAS_FILE', 'HAS_TEXT', 'HAS_AST', 'PARENT_OF']
    RETURN p as ParentNode, head(labels(p)) as ParentNodeType
    ORDER BY p.node_id 
  """
  return neo4j_util.run_neo4j_query(query, driver)


class GetChildrenNodeInput(BaseModel):
  node_id: str = Field(description="Get children nodes of node with this node_id")


def get_children_node(node_id: int, driver: GraphDatabase.driver) -> str:
  query = f"""\
    MATCH (p {{ node_id: {node_id} }}) -[r]-> (c)
    WHERE type(r) IN ['HAS_FILE', 'HAS_TEXT', 'HAS_AST', 'PARENT_OF']
    RETURN c as ChildNode, head(labels(p)) as ChildNodeType
    ORDER BY c.node_id 
  """
  return neo4j_util.run_neo4j_query(query, driver)
