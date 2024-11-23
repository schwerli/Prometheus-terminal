import logging
from typing import Literal, Optional, Union

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.graphs.issue_state import IssueState
from prometheus.lang_graph.subgraphs.context_provider_subgraph import ContextProviderSubgraph
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import (
  IssueAnswerAndFixState,
  IssueType,
)
from prometheus.lang_graph.subgraphs.issue_classification_state import IssueClassificationState
from prometheus.utils.issue_util import format_issue_comments


class IssueToContextNode:
  def __init__(
    self,
    type_of_context: Union[IssueType, Literal["classification"]],
    model: BaseChatModel,
    kg: KnowledgeGraph,
    neo4j_driver: neo4j.Driver,
    max_token_per_neo4j_result: int,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.issue_to_context_node")
    self.type_of_context = type_of_context
    self.context_provider_subgraph = ContextProviderSubgraph(
      model, kg, neo4j_driver, max_token_per_neo4j_result, thread_id, checkpointer
    )

  def format_issue(self, state: Union[IssueAnswerAndFixState, IssueClassificationState]) -> str:
    return f"""\
A user has reported the following issue to the codebase:
Title:
{state["issue_title"]}

Issue description: 
{state["issue_body"]}

Issue comments:
{format_issue_comments(state["issue_comments"])}"""

  def format_classification_query(self, state: Union[IssueState, IssueClassificationState]):
    issue_description = self.format_issue(state)
    query = f"""\
{issue_description}

CONTEXT SEARCH OBJECTIVE:
Find all relevant code context that can help accurately classify this issue. Focus on:
- Implementation patterns that match issue characteristics
- Similar features or components mentioned
- Documentation describing system behavior
- Test cases demonstrating related functionality
- Configuration files defining relevant behaviors
- Related issue handling patterns

Search across:
1. Core implementation files
2. Interface definitions
3. Configuration schemas
4. Documentation files
5. Test suites
6. Example code

Prioritize finding:
- Code patterns matching the issue description
- Component interfaces relevant to the problem
- Configuration options affecting the behavior
- Documentation of related functionality
- Test cases covering similar scenarios
"""
    return query

  def format_bug_query(self, state: Union[IssueState, IssueClassificationState]):
    issue_description = self.format_issue(state)
    query = f"""\
{issue_description}

CONTEXT SEARCH OBJECTIVE:
Find all relevant code and configuration that could explain or be related to this bug. Specifically search for:
- Source code matching the described behavior or error
- Error handling and logging code
- Configuration settings that might affect this behavior
- Related function calls and data flows

Please exclude test files from the search as they are not relevant for fixing the issue.

Focus on source files, configuration files, and logging implementations that could help diagnose and fix this bug.
"""
    return query

  def format_feature_query(self, state: Union[IssueState, IssueClassificationState]):
    issue_description = self.format_issue(state)
    query = f"""\
{issue_description}

CONTEXT SEARCH OBJECTIVE:
Find all relevant code context to guide this feature implementation. Focus on:
- Similar existing features
- Extension points in the codebase
- Related interfaces and patterns
- Configuration frameworks
- Integration patterns
- Testing approaches

Search across:
1. Similar feature implementations
2. Extension interfaces
3. Configuration systems
4. Integration patterns
5. Test frameworks
6. Documentation guidelines

Prioritize finding:
- Similar feature implementations
- Extension mechanisms
- Configuration patterns
- Integration examples
- Test templates
- Documentation standards
- Development guidelines
"""
    return query

  def format_documentation_query(self, state: Union[IssueState, IssueClassificationState]):
    issue_description = self.format_issue(state)
    query = f"""\
{issue_description}

CONTEXT SEARCH OBJECTIVE:
Find all relevant code context to improve documentation. Focus on:
- Existing documentation patterns
- Code comments and docstrings
- Usage examples
- API documentation
- Configuration documentation
- Test case documentation

Search across:
1. Documentation files
2. Code comments
3. Example implementations
4. Test cases
5. Configuration files
6. API definitions

Prioritize finding:
- Related documentation
- Documentation templates
- Code documentation patterns
- Example usage
- Configuration documentation
- Test documentation
- API documentation standards
"""
    return query

  def format_question_query(self, state: Union[IssueState, IssueClassificationState]):
    issue_description = self.format_issue(state)
    query = f"""\
{issue_description}

CONTEXT SEARCH OBJECTIVE:
Find all relevant code context to answer this question comprehensively. Focus on:
- Direct implementations related to the question
- Documentation explaining the behavior
- Configuration options
- Usage examples
- Test cases demonstrating functionality
- Related features and patterns

Search across:
1. Implementation files
2. Documentation
3. Configuration files
4. Example code
5. Test cases
6. Related components

Prioritize finding:
- Relevant implementations
- Official documentation
- Configuration options
- Usage examples
- Test cases
- Related functionality
- Integration patterns
"""
    return query

  def __call__(self, state: Union[IssueState, IssueClassificationState]):
    self._logger.info(f"Finding context for {self.type_of_context}")
    if self.type_of_context == "classification":
      query = self.format_classification_query(state)
      state_context_key = "classification_context"
    elif self.type_of_context == IssueType.BUG:
      query = self.format_bug_query(state)
      state_context_key = "bug_context"
    elif self.type_of_context == IssueType.FEATURE:
      query = self.format_feature_query(state)
      state_context_key = "feature_context"
    elif self.type_of_context == IssueType.DOCUMENTATION:
      query = self.format_documentation_query(state)
      state_context_key = "documentation_context"
    elif self.type_of_context == IssueType.QUESTION:
      query = self.format_question_query(state)
      state_context_key = "question_context"
    else:
      raise ValueError(f"Unknown context type: {self.type_of_context}")

    context = self.context_provider_subgraph.invoke(query)
    self._logger.info(f"{state_context_key}: {context}")
    return {state_context_key: context}
