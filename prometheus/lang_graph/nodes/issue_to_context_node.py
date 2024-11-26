'''
import logging
from typing import Literal, Optional, Union

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.graphs.issue_state import IssueState, IssueType
from prometheus.lang_graph.subgraphs.issue_classification_state import IssueClassificationState
from prometheus.utils.issue_util import format_issue_comments


  def format_issue(self, state: IssueClassificationState) -> str:
    return f"""\
A user has reported the following issue to the codebase:
Title:
{state["issue_title"]}

Issue description: 
{state["issue_body"]}

Issue comments:
{format_issue_comments(state["issue_comments"])}"""

  def format_bug_query(self, state: Union[IssueState, IssueClassificationState]):
    issue_description = self.format_issue(state)
    query = f"""\
{issue_description}

OBJECTIVE: Trace execution path and gather all code needed to understand and fix this bug.

<analysis>
To identify the root cause, we need to:

1. Trace execution path:
   - Find where execution starts
   - Follow the call chain
   - Identify state modifications

2. Find definition of involved code:
   - Find all relevant classes/methods
   - Find parent classes and interfaces
   - Find helper functions used

3. Analyze potential causes:
   - Check initialization logic
   - Verify calculations
   - Look for state corruption
   - Check inheritance behavior
</analysis>

<example>
A user has reported the following issue to the codebase:
Issue title: calculate_area returns wrong result

Issue description:
When calculating the area of a rectangle, it returns the wrong result. Below is a minimal reproducing example:

```python
rectangle = Rectangle(10, 5)
result = rectangle.calculate_area()
assert result == 50  # AssertionError: actual value is 75
```

Analysis Steps:
1. First, we need Rectangle class definition and its parent class
2. Then check Rectangle initialization and any methods called during it
3. Finally examine calculate_area implementation and its dependencies

Required Context:
# File: geometry/base.py
from abc import ABC, abstractmethod

class Figure(ABC):
    def __init__(self):
        self._cached_area = None
        self._scale_factor = 1.0
    
    @abstractmethod
    def calculate_area(self):
        pass
    
    def set_scale(self, factor):
        self._scale_factor = factor
        self._cached_area = None

# File: geometry/rectangle.py
from .base import Figure
from .utils import validate_dimensions
from .config import ENABLE_CACHING

class Rectangle(Figure):
    def __init__(self, length, width):
        super().__init__()
        validate_dimensions(length, width)
        self.length = length
        self.width = width
    
    def calculate_area(self):
        if ENABLE_CACHING and self._cached_area is not None:
            return self._cached_area
            
        area = self.length * self.width * self._scale_factor
        
        if ENABLE_CACHING:
            self._cached_area = area
            
        return area

# File: geometry/utils.py
def validate_dimensions(*args):
    for dim in args:
        if not isinstance(dim, (int, float)):
            raise TypeError(f"Dimension must be numeric, got {{type(dim)}}")
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {{dim}}")

# File: geometry/config.py
ENABLE_CACHING = True
DEFAULT_SCALE = 1.5  # This could be the issue - default scale factor is 1.5

Potential Issues:
1. Parent class Figure initializes _scale_factor = 1.0
2. config.py sets DEFAULT_SCALE = 1.5
3. Area caching might retain stale values
4. Scale factor affects all calculations

Additional Context Needed:
1. Where is DEFAULT_SCALE being applied?
2. Are there other places that modify _scale_factor?
3. Check for dimension validation side effects
4. Review caching behavior impact
</example>

CONSTRAINTS:
- Include complete file paths
- Show full implementations
- Preserve all comments
- Include relevant line numbers
- Show configuration context

Using this issue description, trace through the code to gather ALL relevant context needed to understand and fix the bug.
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

def format_question_query(state: Union[IssueState, IssueClassificationState]):
  issue_description = format_issue(state)
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
'''
