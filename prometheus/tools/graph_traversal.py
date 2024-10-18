from pathlib import Path

from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from prometheus.parser import tree_sitter_parser
from prometheus.utils import neo4j_util

MAX_RESULT = 20

###############################################################################
#                          FileNode retrieval                                 #
###############################################################################


class FindFileNodeWithBasenameInput(BaseModel):
  basename: str = Field("The basename of FileNode to search for")


FIND_FILE_NODE_WITH_BASENAME_DESCRIPTION = """\
Find all FileNode in the graph with this basename of a file/dir. The basename must
include the extension, like 'bar.py', 'baz.java' or 'foo'
(in this case foo is a direcctory or a file without extension).

You can use this tool to check if a file/dir with this basename exists or get all
attributes related to the file/dir."""


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


FIND_FILE_NODE_WITH_RELATIVE_PATH_DESCRIPTION = """\
Search FileNode in the graph with this relative_path of a file/dir. The relative_path is
the relative path from the root path of codebase. The relative_path must include the extension,
like 'foo/bar/baz.java'.

You can use this tool to check if a file/dir with this relative_path exists or get all
attributes related to the file/dir."""


def find_file_node_with_relative_path(relative_path: str, driver: GraphDatabase.driver) -> str:
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


FIND_AST_NODE_WITH_TEXT_DESCRIPTION = """\
Find all ASTNode in the graph that exactly contains this text. The contains is
same as python's check `'foo' in text`, ie. it is case sensitive and is
looking for exact matches. Therefore the search text should be exact as well.

You can use this tool to find all source code in the code that contains this text."""


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


FIND_AST_NODE_WITH_TYPE_DESCRIPTION = """\
Find all ASTNode in the graph that has this tree-sitter node type.

You can use this tool to find all source code with a certain type, like method_declaration
or class_declaration."""


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


FIND_AST_NODE_WITH_TEXT_IN_FILE_DESCRIPTION = """\
Find all ASTNode in the graph that exactly contains this text in a file with this basename.
The contains is same as python's check `'foo' in text`, ie. it is case sensitive and is
looking for exact matches. Therefore the search text should be exact as well. The basename
must include the extension, like 'bar.py', 'baz.java' or 'foo'
(in this case foo is a direcctory or a file without extension).

You can use this tool to find all source code with a certain text in a specific file."""


def find_ast_node_with_text_in_file(text: str, basename: str, driver: GraphDatabase.driver) -> str:
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


FIND_AST_NODE_WITH_TYPE_IN_FILE_DESCRIPTION = """\
Find all ASTNode in the graph that has this tree-sitter node type in a file with this basename.
The basename must include the extension, like 'bar.py', 'baz.java' or 'foo'
(in this case foo is a direcctory or a file without extension).

You can use this tool to find all source code with a certain type in a specific file,
like method_declaration or class_declaration."""


def find_ast_node_with_type_in_file(type: str, basename: str, driver: GraphDatabase.driver) -> str:
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


FIND_AST_NODE_WITH_TYPE_AND_TEXT_DESCRIPTION = """\
Find all ASTNode in the graph that has this tree-sitter node type and exactly
contains this text. The contains is same as python's check `'foo' in text`,
ie. it is case sensitive and is looking for exact matches. Therefore the search
text should be exact as well.

You can use this tool to find all source code with a certain type and text.
For example to find all method_declartion that contains 'x = 1'."""


def find_ast_node_with_type_and_text(type: str, text: str, driver: GraphDatabase.driver) -> str:
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


FIND_TEXT_NODE_WITH_TEXT_DESCRIPTION = """\
Find all TextNode in the graph that exactly contains this text. The contains is
same as python's check `'foo' in text`, ie. it is case sensitive and is
looking for exact matches. Therefore the search text should be exact as well.

You can use this tool to find all text/documentation in codebase that contains this text."""


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


FIND_TEXT_NODE_WITH_TEXT_IN_FILE_DESCRIPTION = """\
Find all TextNode in the graph that exactly contains this text in a file with this basename.
The contains is same as python's check `'foo' in text`, ie. it is case sensitive and is
looking for exact matches. Therefore the search text should be exact as well.
The basename must include the extension, like 'bar.py', 'baz.java' or 'foo'
(in this case foo is a direcctory or a file without extension).

You can use this tool to find text/documentation in a specific file that contains this text."""


def find_text_node_with_text_in_file(text: str, basename: str, driver: GraphDatabase.driver) -> str:
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


GET_NEXT_TEXT_NODE_WITH_NODE_ID_DESCRIPTION = """\
Get the next TextNode of this given node_id.

You can use this tool to read the next section of text that you are interested in."""


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


PREVIEW_FILE_CONTENT_WITH_BASENAME_DESCRIPTION = """\
Preview the content of a file with this basename. The basename must include
the extension, like 'bar.py', 'baz.java' or 'foo' (in this case foo is a
direcctory or a file without extension).

You can use this tool to preview the content of a specific file to see what it contains
in the first 300 lines or the first section."""


def preview_file_content_with_basename(basename: str, driver: GraphDatabase.driver) -> str:
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


GET_PARENT_NODE_DESCRIPTION = """\
Get the parent node in graph of this given node_id.

This tool can be used to traverse the graph. For example to find the parent directory
of a file, or the parent ASTNode of a ASTnode (eg. 'method_declaration' to 'class_declaration')."""


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


GET_CHILDREN_NODE_DESCRIPTION = """\
Get the children nodes in graph of this given node_id.

This tool can be used to traverse the graph. For example to find the children files/directories
of a directory, or the children ASTNode of a ASTnode (eg. all source code components under 'class_declaration')."""


def get_children_node(node_id: int, driver: GraphDatabase.driver) -> str:
  query = f"""\
    MATCH (p {{ node_id: {node_id} }}) -[r]-> (c)
    WHERE type(r) IN ['HAS_FILE', 'HAS_TEXT', 'HAS_AST', 'PARENT_OF']
    RETURN c as ChildNode, head(labels(p)) as ChildNodeType
    ORDER BY c.node_id 
  """
  return neo4j_util.run_neo4j_query(query, driver)
