import logging
from typing import Literal, Optional, Union

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.graphs.issue_state import IssueState, IssueType
from prometheus.lang_graph.subgraphs.context_provider_subgraph import ContextProviderSubgraph
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

  def format_issue(self, state: IssueClassificationState) -> str:
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

{issue_description}

OBJECTIVE: Find ALL self-contained context needed to accurately classify this issue as a bug, feature request, documentation update, or question.

<reasoning>
1. Analyze issue characteristics:
   - Error patterns suggesting bugs
   - New functionality requests
   - Documentation gaps
   - Knowledge-seeking patterns

2. Search strategy:
   - Implementation files matching descriptions
   - Similar existing features
   - Documentation coverage
   - Related Q&A patterns

3. Required context categories:
   - Core implementations
   - Feature interfaces
   - Documentation files
   - Test coverage
   - Configuration settings
   - Issue history
</reasoning>

REQUIREMENTS:
- Context MUST be fully self-contained
- MUST include complete file paths
- MUST include full function/class implementations
- MUST preserve all code structure and formatting
- MUST include line numbers

<examples>
<example id="error-classification">
<issue>
Database queries timing out randomly
Error: Connection pool exhausted
</issue>

<search_targets>
1. Connection pool implementation
2. Database configuration
3. Error handling code
4. Related timeout settings
</search_targets>

<expected_context>
- Complete connection pool class
- Full database configuration
- Error handling implementations
- Timeout management code
</expected_context>
</example>
</examples>

Search priority:
1. Implementation patterns matching issue
2. Feature definitions and interfaces
3. Documentation coverage
4. Configuration schemas
5. Test implementations
6. Integration patterns
"""
    return query

  def format_bug_query(self, state: Union[IssueState, IssueClassificationState]):
    issue_description = self.format_issue(state)
    query = f"""\
{issue_description}

OBJECTIVE: Systematically identify and gather ALL code context required to understand and fix this bug.

<reasoning>
To fix any bug, we need to:
1. Trace the execution path:
   - Where is the error first manifested?
   - What functions/methods are in the direct call stack?
   - What class/object state is involved?

2. Identify dependencies:
   - What initializes the involved objects?
   - What methods modify the relevant state?
   - What helper functions are called?
   - What configuration affects behavior?

3. Analyze data flow:
   - How is data passed between components?
   - Where are values modified?
   - What validation occurs?
   - What type conversions happen?
</reasoning>

REQUIRED CONTEXT - Gather:
1. Class/Type Definitions:
   - Complete class implementation
   - All parent classes and interfaces
   - Member variable declarations
   - Constructor/initialization logic

2. Method Implementations:
   - Full method definitions
   - Helper functions called within
   - Validation methods
   - Error handling code

3. Configuration/State:
   - Default values
   - Configuration files
   - Environment variables
   - Global state

<examples>
<example id="calculation-error">
<bug>
total = ShoppingCart(items).calculate_total()
assert total == 150  # Fails, actual: 165
</bug>

<analysis>
Need to examine:
1. ShoppingCart initialization
   - How are items stored?
   - What happens during construction?
2. calculate_total implementation
   - Core calculation logic
   - Any price modifiers
3. Called methods
   - Price lookup
   - Discount calculation
   - Tax computation
</analysis>

<required_context>
# File: shopping/cart.py
class ShoppingCart:
    def __init__(self, items):
        # Full constructor
    
    def calculate_total(self):
        # Complete method

# File: shopping/pricing.py
def get_item_price(item):
    # Price lookup logic

def apply_discounts(subtotal):
    # Discount rules

def calculate_tax(amount):
    # Tax calculation
</required_context>
</example>
</examples>

CONSTRAINTS:
- Include complete relative file paths
- Include full class/function implementations
- Preserve all code formatting and comments
- Must include line numbers where relevant
- Show all configuration that affects behavior

Return all context that could influence the bug's behavior.
"""
    return query

  def format_feature_query(self, state: Union[IssueState, IssueClassificationState]):
    issue_description = self.format_issue(state)
    query = f"""\
{issue_description}

OBJECTIVE: Find ALL self-contained context needed to implement this new feature.

<reasoning>
1. Analyze feature requirements:
   - Similar features
   - Extension points
   - Integration needs
   - Configuration requirements

2. Search strategy:
   - Similar implementations
   - Extension interfaces
   - Integration patterns
   - Configuration systems

3. Required context:
   - Complete similar features
   - Full extension points
   - All integration patterns
   - Configuration examples
</reasoning>

REQUIREMENTS:
- Context MUST be fully self-contained
- MUST include complete file paths
- MUST include full function/class implementations
- MUST preserve all code structure and formatting
- MUST include line numbers

<examples>
<example id="new-auth-provider">
<issue>
Add support for OAuth2 provider
</issue>

<search_targets>
1. Existing auth providers
2. Auth interfaces
3. Provider configuration
4. Integration patterns
</search_targets>

<expected_context>
- Complete auth provider interface
- Existing provider implementations
- Full configuration system
- Integration examples
</expected_context>
</example>
</examples>

Search priority:
1. Similar implementations
2. Extension points
3. Integration patterns
4. Configuration systems
5. Test templates
6. Documentation standards
"""
    return query

  def format_documentation_query(self, state: Union[IssueState, IssueClassificationState]):
    issue_description = self.format_issue(state)
    query = f"""\
{issue_description}

OBJECTIVE: Find ALL self-contained context needed to improve or create documentation.

<reasoning>
1. Analyze documentation needs:
   - Implementation details
   - API patterns
   - Configuration options
   - Usage examples

2. Search strategy:
   - Existing documentation
   - Code implementation
   - Configuration files
   - Test examples

3. Required context:
   - Full implementations
   - Complete configurations
   - All related examples
   - Existing documentation
</reasoning>

REQUIREMENTS:
- Context MUST be fully self-contained
- MUST include complete file paths
- MUST include full function/class implementations
- MUST preserve all code structure and formatting
- MUST include line numbers

<examples>
<example id="api-docs">
<issue>
Missing API documentation for user service
</issue>

<search_targets>
1. User service implementation
2. API definitions
3. Usage examples
4. Existing docs
</search_targets>

<expected_context>
- Complete service implementation
- Full API interface
- All usage examples
- Related documentation
</expected_context>
</example>
</examples>

Search priority:
1. Code implementation
2. API definitions
3. Configuration options
4. Usage examples
5. Test cases
6. Existing documentation
"""
    return query

  def format_question_query(self, state: Union[IssueState, IssueClassificationState]):
    issue_description = self.format_issue(state)
    query = f"""\
{issue_description}

OBJECTIVE: Find ALL self-contained context needed to comprehensively answer this question.

<reasoning>
1. Analyze question focus:
   - Implementation details
   - Behavior patterns
   - Configuration options
   - Usage scenarios

2. Search strategy:
   - Direct implementations
   - Related components
   - Configuration settings
   - Usage examples

3. Required context:
   - Full implementations
   - Complete configurations
   - All related examples
   - Supporting documentation
</reasoning>

REQUIREMENTS:
- Context MUST be fully self-contained
- MUST include complete file paths
- MUST include full function/class implementations
- MUST preserve all code structure and formatting
- MUST include line numbers

<examples>
<example id="caching-question">
<issue>
How does the caching system handle race conditions?
</issue>

<search_targets>
1. Cache implementation
2. Lock mechanisms
3. Race condition handling
4. Usage patterns
</search_targets>

<expected_context>
- Complete cache implementation
- Full lock handling
- All race condition code
- Usage examples
</expected_context>
</example>
</examples>

Search priority:
1. Direct implementations
2. Related components
3. Configuration options
4. Usage examples
5. Test cases
6. Documentation
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
