import logging
from typing import Dict

from langchain_core.messages import HumanMessage

from prometheus.utils.issue_util import format_issue_info


class IssueBugContextMessageNode:
  HUMAN_PROMPT = """\
{issue_info}

OBJECTIVE: Trace execution path and gather COMPLETE context needed to understand and fix this bug. The context
must be comprehensive so that a person with zero knowledge of the codebase can fully understand and fix this bug
without needing to look up any additional information.

<analysis>
To identify the root cause, we need to:
1. Trace execution path.
2. Find definition of involved code.
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

<steps>
<step1>
<thought>I need to find the Rectangle class to understand how it is initialized and understand calculate_area function</thought>
<action> find Rectangle class
# File: geometry/rectangle.py, line: 1-22
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
</action>
</step1>

<step2>
<thought>I need to find the Figure parent class because we call super().__init__()</thought>
<action> find __init__ method in Figure class
# File: geometry/base.py, line: 1-7
from abc import ABC, abstractmethod

class Figure(ABC):
    def __init__(self):
        self._cached_area = None
        self._scale_factor = 1.0
</action>
</step2>

<step3>
<thought>I need to find ENABLE_CACHING from config.py file because calculate_area function uses this variable that might affect the area calculation</thought>
<action> find ENABLE_CACHING in config file
# File: geometry/config.py, line 2-3
DEFAULT_SCALE = 1.5
</action>
</step3>

<step4>
<thought>Now we have complete context even for a person without any knowledge about the codebase to understand and fix the bug.</thought>
<action>Return all context we found
# File: geometry/rectangle.py, line: 1-22
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

# File: geometry/base.py, line: 1-7
from abc import ABC, abstractmethod

class Figure(ABC):
    def __init__(self):
        self._cached_area = None
        self._scale_factor = 1.0

# File: geometry/config.py, line 2-3
DEFAULT_SCALE = 1.5       
</action>
</step4>
<steps>

IMPORTANT REQUIREMENTS:
1. Do NOT stop until you have found ALL relevant code and context
2. Do NOT skip ANY potential execution paths
3. Do NOT assume ANYTHING about the codebase - find explicit evidence for everything
"""

  def __init__(self):
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.issue_bug_context_message_node")

  def __call__(self, state: Dict):
    human_message = HumanMessage(
      self.HUMAN_PROMPT.format(
        issue_info=format_issue_info(
          state["issue_title"], state["issue_body"], state["issue_comments"]
        ),
      )
    )
    self._logger.debug(f"Sending query to ContextProviderNode:\n{human_message}")
    return {"context_provider_messages": [human_message]}
