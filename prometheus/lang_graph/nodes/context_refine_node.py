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
additional source code or documentation from the codebase is needed. You can only request source code files,
configuration files, or documentation files that may exist in a codebase.

Step-by-Step Analysis Process:
1. Examine the current context and issue details
2. Identify what source files are involved
3. Check if you have all relevant code including:
   - Complete class/function implementations
   - Referenced classes and interfaces
   - Import statements and dependencies
   - Configuration files
   - Documentation files
4. Determine if accessing more files would help understand/resolve the issue
5. If needed, formulate specific queries for additional files

Your output must strictly follow this Pydantic model:
```python
class ContextRefineStructuredOutput(BaseModel):
    reasoning: str     # Your step by step reasoning about the context needs
    refined_query: str # Query for additional source/config/doc files, empty if context is sufficient

Important Guidelines:
1. Only request files that exist in the codebase
2. You can request:
   - Source code files (.java, .py, etc.)
   - Configuration files (.xml, .yaml, etc.)
   - Documentation files (.md, .txt, etc.)
3. Do NOT request:
   - Git history or commit information
   - Test execution logs
   - Runtime information
   - System logs
   - Database contents
4. Make specific file queries like:
   - "Find the implementation of BaseAuth.java"
   - "Get the configuration file referenced in Config.load()"
   - "Find the interface that CustomService implements"
5. Return empty refined_query if:
   - Current context is sufficient
   - The issue is a logic/implementation error
   - Needed information isn't in source/config files
""".replace("{", "{{").replace("}", "}}")

  REFINE_PROMPT = ("""\
Your task is to analyze if the current context contains enough source code and documentation
for a developer WITHOUT prior knowledge of the codebase to fix the issue. You can request
additional source files, configuration files, or documentation files from the codebase.

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
1. Issue involves OAuth token validation
2. Current context shows AuthenticationManager using TokenValidator
3. Missing TokenValidator implementation
4. Missing OAuth configuration
5. Need both to understand validation requirements

Output:
{
  "reasoning": "The current context only shows AuthenticationManager using TokenValidator but we're missing:\n1. TokenValidator interface/class implementation which contains the actual validation logic\n2. OAuth configuration files that specify the validation rules\nBoth are necessary to understand and fix the token validation issue.",
  "refined_query": "Find TokenValidator implementation and OAuth configuration files referenced in the authentication system"
}
</example>

<example>
Input:
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
2. Have complete implementation of calculation method
3. Logic error in handling multiple discounts
4. All relevant code is visible
5. No external dependencies needed

Output:
{
  "reasoning": "The context shows the complete discount calculation logic in PricingService. The issue is with the mathematical logic of applying multiple discounts sequentially. This is an implementation error in the code we can already see.",
  "refined_query": ""
}
</example>
</examples>

Analyze if more source code or documentation files are needed. Remember to only request files that exist in the codebase.
""".replace("{", "{{").replace("}", "}}")
   .replace("{{issue_info}}", "{issue_info}")
   .replace("{{bug_context}}", "{bug_context}")
)

  EDIT_AND_ERROR_PROMPT = ("""\
Your task is to analyze if a build/test failure occurred because the edit agent lacked
necessary source code or documentation files. You can request additional files from the codebase to help fix the error.

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
index 1234567..9876543 100644
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
1. Build error shows missing SecurityExpressions class
2. Current context only shows the controller
3. Need security configuration and expression handling
4. SecurityExpressions class defines available methods

Output:
{
  "reasoning": "The build error indicates the patch uses 'hasAuthority' method from SecurityExpressions, but we don't have:\n1. SecurityExpressions class that defines the available security methods\n2. Security configuration that sets up the expression handling\nBoth are needed to implement the correct permission check.",
  "refined_query": "Find SecurityExpressions class implementation and Spring security configuration files"
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
1. Have complete LogProcessor implementation
2. Test failure shows date parsing logic error
3. All date handling code is visible
4. This is a logic error in the code we can see

Output:
{
  "reasoning": "The test failure is due to incorrect date parsing logic in the existing code. We have the complete LogProcessor implementation and can see that the string replacement is causing incorrect parsing. This is a logic error, not a missing context issue.",
  "refined_query": ""
}
</example>
</examples>

Analyze if the build/test failure is due to missing source code or documentation files. Remember to only request files that exist in the codebase.
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
    if self.has_edit_and_error:
      human_prompt = self.format_edit_and_error_message(state)
    else:
      human_prompt = self.format_refine_message(state)
    response = self.model.invoke({"human_prompt": human_prompt})
    self._logger.debug(response)

    state_update = {"refined_query": response.refined_query}
    if response.refined_query:
      state_update["context_provider_messages"] = [HumanMessage(content=response.refined_query)]

    return state_update
