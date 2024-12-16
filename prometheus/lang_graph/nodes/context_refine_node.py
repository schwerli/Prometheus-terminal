import logging
from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.utils.issue_util import format_issue_info
from prometheus.utils.lang_graph_util import extract_ai_responses


class ContextRefineStructuredOutput(BaseModel):
  reasoning: str = Field(description="Your step by step reasoning.")
  refined_query: str = Field(
    "Additional query to ask the ContextRetriever if the context is not enough. Empty otherwise."
  )


class ContextRefineNode:
  SYS_PROMPT = """\
You are a software engineering assistant specialized in analyzing code context to determine if
additional source code or documentation from the codebase is absolutely necessary. You should be conservative 
in requesting additional context and only do so when the current context is clearly insufficient.

Step-by-Step Analysis Process:
1. Examine the current context and issue details carefully
2. Identify what source files are currently provided
3. Determine if the current context is sufficient by checking:
   - Can the issue be understood with current context?
   - Is the problem area clearly visible?
   - Are all directly referenced dependencies available?
   - Can a fix be implemented with this context?
4. Only request additional files if:
   - Critical referenced code is missing
   - Essential configuration is absent
   - Required interfaces are not visible
   - Documentation is needed for compliance/requirements

Your output must strictly follow this Pydantic model:
```python
class ContextRefineStructuredOutput(BaseModel):
    reasoning: str     # Your step by step reasoning about why current context is or isn't sufficient
    refined_query: str # Detailed query including issue context and specific files needed, empty if sufficient

Important Guidelines:
1. Default to working with current context unless clearly insufficient
2. Only request files when you can explain why they are CRITICAL for fixing the issue
3. Acceptable file types (request ONLY if essential):
   - Source code files (.java, .py, etc.)
   - Configuration files (.xml, .yaml, etc.)
   - Documentation files (.md, .txt, etc.)
4. Do NOT request:
   - Git history or commit information
   - Test execution logs
   - Runtime information
   - System logs
   - Database contents
   - Files that would be "nice to have" but aren't essential
5. Make specific, detailed queries that include:
   - Description of the issue being fixed
   - Why the files are needed
   - Specific file names or patterns to search for
   Examples:
   - "To fix the OAuth token validation issue where users are being rejected, I need the TokenValidator implementation to understand the validation logic that changed after the provider update. Please find TokenValidator.java and related OAuth configuration files."
   - "To resolve the security permission error in AdminController, I need to see the SecurityExpressions class that defines available security methods. The issue is causing access denied errors for admin users. Please find SecurityExpressions.java."
6. Return empty refined_query if:
   - Current context allows understanding the issue
   - The issue is primarily a logic/implementation error
   - Additional context wouldn't significantly help fix the issue
   - You're unsure if more context would help
""".replace("{", "{{").replace("}", "}}")

  REFINE_PROMPT = ("""\
Your task is to analyze if the current context is sufficient for a developer to understand and fix the issue.
Be conservative in requesting additional files - only request them if they are absolutely necessary to fix the issue.

{issue_info}

Current context provided:
{bug_context}

<examples>
<example>
Issue Title: "AuthenticationManager fails to validate OAuth tokens"
Issue Body: "When using OAuth authentication, the system is not properly validating tokens. Users are being rejected even with valid tokens."
Issue Comments:
- john: "This started happening after the OAuth2 provider update."
- sarah: "I noticed it's specifically failing for refresh tokens."

Context:
File: src/main/java/com/example/auth/AuthenticationManager.java
```java
public class AuthenticationManager {
    private final TokenValidator tokenValidator;
    
    @Autowired
    public AuthenticationManager(TokenValidator tokenValidator) {
        this.tokenValidator = tokenValidator;
    }
    
    public boolean validateToken(String token) {
        return tokenValidator.validate(token);
    }
}
```

Thought process:
1. Issue involves token validation failures
2. Current context shows only the delegation to TokenValidator
3. TokenValidator implementation is directly referenced and is critical
4. Cannot determine validation logic without TokenValidator
5. OAuth configuration is essential as it defines validation rules

Output:
{
  "reasoning": "The current context is insufficient because:\n1. We only see AuthenticationManager delegating to TokenValidator\n2. The TokenValidator implementation is directly referenced and contains the actual validation logic\n3. OAuth configuration is critical as it defines the validation rules that recently changed\nBoth files are essential to understand and fix the token validation issue.",
  "refined_query": "To fix the OAuth token validation issue where users are being rejected after a provider update, particularly for refresh tokens, I need to examine:\n1. The TokenValidator implementation to understand the validation logic\n2. OAuth configuration files that define the validation rules\nPlease find the TokenValidator class and OAuth configuration files referenced in the authentication system."
}
</example>

<example>
Issue Title: "Fix incorrect calculation in PricingService"
Issue Body: "The total price calculation is wrong when applying multiple discounts"
Issue Comments:
- mike: "The bug happens when there are both percentage and fixed discounts"
- jane: "Confirmed, order #1234 shows this issue"

Context:
File: src/main/java/com/example/pricing/PricingService.java
```java
public class PricingService {
    public double calculateTotal(Order order) {
        double total = order.getBasePrice();
        
        for (Discount discount : order.getDiscounts()) {
            if (discount.isPercentage()) {
                total *= (1 - discount.getValue() / 100.0);
            } else {
                total -= discount.getValue();
            }
        }
        
        return Math.max(0, total);
    }
}
```

Thought process:
1. Issue is about discount calculation logic
2. The bug involves the order of applying different discount types
3. Complete calculation implementation is visible
4. All discount handling logic is in this method
5. Referenced classes (Order, Discount) are used in simple ways

Output:
{
  "reasoning": "The current context is sufficient because:\n1. The complete discount calculation logic is visible\n2. We can see how different discount types are handled\n3. The issue is clearly in the mathematical logic of applying discounts sequentially\n4. The Order and Discount classes are used in straightforward ways (getBasePrice, getDiscounts, isPercentage, getValue)\n5. Additional context wouldn't help fix this logical error",
  "refined_query": ""
}
</example>
</examples>

Analyze if the current context is truly sufficient or if additional files are ABSOLUTELY NECESSARY to fix the issue. 
Be conservative - only request additional files if you can clearly explain why they are critical.
""".replace("{", "{{").replace("}", "}}")
   .replace("{{issue_info}}", "{issue_info}")
   .replace("{{bug_context}}", "{bug_context}")
)

  EDIT_AND_ERROR_PROMPT = ("""\
Your task is to analyze if a build/test failure occurred because critical source code or documentation files
were missing. Only request additional files if they are absolutely necessary to fix the error.

{issue_info}

Context provided to edit agent:
{bug_context}

Generated patch:
{edit_patch}

Error encountered:
{edit_error}

<examples>
<example>
Issue Title: "Fix user permission check in AdminController"
Issue Body: "Users are getting access denied even when they have admin role"
Issue Comments:
- alex: "This affects all admin panel endpoints"
- maria: "The roles are correctly set in the database"

Context:
File: src/main/java/com/example/admin/AdminController.java
```java
@RestController
@RequestMapping("/admin")
public class AdminController {
    @PreAuthorize("hasRole('ADMIN')")
    @GetMapping("/users")
    public List<UserDTO> getUsers() {
        return userService.getAllUsers();
    }
}
```

Patch:
```diff
diff --git a/src/main/java/com/example/admin/AdminController.java b/src/main/java/com/example/admin/AdminController.java
index 1234567..9876543 100000
--- a/src/main/java/com/example/admin/AdminController.java
+++ b/src/main/java/com/example/admin/AdminController.java
@@ -8,7 +8,7 @@ public class AdminController {
-    @PreAuthorize("hasRole('ADMIN')")
+    @PreAuthorize("hasAuthority('ADMIN')")
     @GetMapping("/users")
     public List<UserDTO> getUsers() {
         return userService.getAllUsers();
```

Error:
```
[ERROR] Failed to compile AdminController.java
[ERROR] cannot find symbol
[ERROR]   symbol:   method hasAuthority(String)
[ERROR]   location: class com.example.security.SecurityExpressions
[ERROR] at com.example.admin.AdminController.getUsers(AdminController.java:9)
```

Thought process:
1. Build error shows undefined hasAuthority method
2. SecurityExpressions class is directly referenced but missing
3. Error indicates a compilation issue, not logic
4. Cannot fix without SecurityExpressions definition

Output:
{
  "reasoning": "Additional context is necessary because:\n1. The build error explicitly shows SecurityExpressions.hasAuthority is undefined\n2. We need the SecurityExpressions class to see available security methods\n3. This is a compilation error, not a logic issue\n4. Cannot fix the error without knowing valid security expression methods",
  "refined_query": "To fix the access denied issue affecting admin users, where the build is failing due to an undefined hasAuthority method, I need to examine the SecurityExpressions class. The error occurs when trying to modify the permission check from hasRole to hasAuthority. Please find the SecurityExpressions class implementation to understand available security methods and fix the compilation error."
}
</example>

<example>
Issue Title: "Fix date parsing in LogProcessor"
Issue Body: "Some log entries are showing incorrect dates after processing"
Issue Comments:
- tom: "The format should be yyyy-MM-dd but some entries use dd-MM-yyyy"

Context:
File: src/main/java/com/example/logging/LogProcessor.java
```java
public class LogProcessor {
    private static final String DATE_FORMAT = "yyyy-MM-dd";
    
    public LogEntry processLine(String line) {
        try {
            String dateStr = line.substring(0, 10);
            Date date = new SimpleDateFormat(DATE_FORMAT).parse(dateStr);
            return new LogEntry(date, line.substring(11));
        } catch (ParseException e) {
            throw new LogProcessingException("Invalid date format", e);
        }
    }
}
```

Patch:
```diff
diff --git a/src/main/java/com/example/logging/LogProcessor.java b/src/main/java/com/example/logging/LogProcessor.java
index abcd123..efgh456 789
--- a/src/main/java/com/example/logging/LogProcessor.java
+++ b/src/main/java/com/example/logging/LogProcessor.java
@@ -4,7 +4,7 @@ public class LogProcessor {
     public LogEntry processLine(String line) {
         try {
             String dateStr = line.substring(0, 10);
-            Date date = new SimpleDateFormat(DATE_FORMAT).parse(dateStr);
+            Date date = new SimpleDateFormat(DATE_FORMAT).parse(dateStr.replace("-", "/"));
             return new LogEntry(date, line.substring(11));
         } catch (ParseException e) {
             throw new LogProcessingException("Invalid date format", e);
```

Error:
```
[ERROR] Tests run: 1, Failures: 1
[ERROR] Failed test: testProcessLine
[ERROR] Expected: date to be 2024-01-15
[ERROR] but was: 2024-13-01
[ERROR] at com.example.logging.LogProcessorTest.testProcessLine(LogProcessorTest.java:25)
```

Thought process:
1. Have full LogProcessor implementation
2. Test failure shows date parsing logic error
3. All relevant code is visible
4. Issue is with date string manipulation
5. Additional context wouldn't help

Output:
{
  "reasoning": "The current context is sufficient because:\n1. We can see the complete date parsing logic\n2. The test failure points to incorrect string manipulation\n3. The error is in the replace() call affecting date format\n4. All relevant code is in the LogProcessor class\n5. Additional files wouldn't help fix this logic error",
  "refined_query": ""
}
</example>
</examples>

Analyze if the build/test failure is due to missing critical source code or documentation files.
Be conservative - only request additional files if they are absolutely necessary to fix the error.
""".replace("{", "{{").replace("}", "}}")
   .replace("{{issue_info}}", "{issue_info}")
   .replace("{{bug_context}}", "{bug_context}")
   .replace("{{edit_patch}}", "{edit_patch}")
   .replace("{{edit_error}}", "{edit_error}")
)

  def __init__(self, model: BaseChatModel, has_edit_and_error: bool = False):
    self.has_edit_and_error = has_edit_and_error
    prompt = ChatPromptTemplate.from_messages(
      [("system", self.SYS_PROMPT), ("human", "{human_prompt}")]
    )
    structured_llm = model.with_structured_output(ContextRefineStructuredOutput)
    self.model = prompt | structured_llm
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.context_refine_message_node")

  def format_edit_and_error_message(self, state: Dict):
    edit_error = ""
    if "reproducing_test_fail_log" in state and state["reproducing_test_fail_log"]:
      edit_error = (
        f"Your failed to pass the bug exposing test cases:\n{state['reproducing_test_fail_log']}"
      )
    elif "build_fail_log" in state and state["build_fail_log"]:
      edit_error = f"Your failed to pass the build:\n{state['build_fail_log']}"
    elif "existing_test_fail_log" in state and state["existing_test_fail_log"]:
      edit_error = f"Your failed to existing test cases:\n{state['existing_test_fail_log']}"

    assert edit_error != ""

    bug_context = "\n".join(extract_ai_responses(state["context_provider_messages"]))

    return self.EDIT_AND_ERROR_PROMPT.format(
      issue_info=format_issue_info(
        state["issue_title"], state["issue_body"], state["issue_comments"]
      ),
      bug_context=bug_context,
      edit_patch=state["edit_patch"],
      edit_error=edit_error,
    )

  def format_refine_message(self, state: Dict):
    bug_context = "\n".join(extract_ai_responses(state["context_provider_messages"]))
    return self.REFINE_PROMPT.format(
      issue_info=format_issue_info(
        state["issue_title"], state["issue_body"], state["issue_comments"]
      ),
      bug_context=bug_context,
    )

  def __call__(self, state: Dict):
    if "max_refined_query_loop" in state and state["max_refined_query_loop"] == 0:
      self._logger.info("Reached max_refined_query_loop, not asking for more context")
      return {"refined_query": ""}

    if self.has_edit_and_error:
      human_prompt = self.format_edit_and_error_message(state)
    else:
      human_prompt = self.format_refine_message(state)
    response = self.model.invoke({"human_prompt": human_prompt})
    self._logger.debug(response)

    state_update = {"refined_query": response.refined_query}

    if "max_refined_query_loop" in state:
      state_update["max_refined_query_loop"] = state["max_refined_query_loop"]-1

    if response.refined_query:
      state_update["context_provider_messages"] = [HumanMessage(content=response.refined_query)]

    return state_update
