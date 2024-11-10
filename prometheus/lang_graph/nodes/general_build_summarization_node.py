"""Build execution analysis and summarization for software projects.

This module analyzes build execution attempts and provides structured summaries of
build systems, required commands, and failure information. It processes build
execution histories to determine build system presence, extract build steps, and
identify any failures.
"""

import logging
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class BuildClassification(BaseModel):
  """Structured output model for build analysis results.

  Attributes:
    exist_build: Boolean indicating presence of a build system.
    command_summary: Detailed description of build system and required commands.
    fail_log: Error logs from failed builds, empty string if successful.
  """

  exist_build: bool = Field(
    description="Indicates if there is any build system present in the project"
  )
  command_summary: str = Field(
    description="Summary of the build system and list of commands required to build the project"
  )
  fail_log: str = Field(
    description="Contains the error logs if build failed, empty string if successful"
  )


class GeneralBuildSummarizationNode:
  """Analyzes and summarizes build execution attempts for software projects.

  This class processes build execution histories to provide structured analysis
  of build systems, required build steps, and any failures encountered. It can
  identify and analyze various build systems including CMake, npm, Maven, and others.

  The analysis covers three main areas:
  1. Build system detection and classification
  2. Required build steps and commands
  3. Build failure analysis and error logging
  """

  SYS_PROMPT = """\
You are a build system expert analyzing build attempts for software projects. You'll review a history
of commands executed by an agent that attempted to build the project. Examine this build history to:

1. Determine if a build system exists (looking for Makefiles, CMakeLists.txt, package.json, etc.)
2. Analyze the build process and required commands
3. Identify any build failures and their causes

Provide three outputs:
1. exist_build: Boolean indicating if a build system is present
2. command_summary: Concise description of the build system and chronological list of commands needed for building, including:
   - Type of build system detected
   - Required dependencies or setup steps
   - Sequence of build commands to execute
3. fail_log: If build failed, provide the relevant error logs. Empty string if build succeeded

When analyzing commands:
- Focus on essential build steps
- Include dependency installation commands
- List commands in execution order
- Note any required environment setup

When capturing fail logs:
- Include complete error messages
- Focus on build-breaking errors
- Exclude warnings or non-critical messages
- Return empty string if build was successful

The input will contain messages showing the agent's attempts and their results.

Example 1: C++ Project with CMake
Input:
Assistant internal thought: Checking for CMake build system files
Assistant executed command: {"command": "find . -name CMakeLists.txt"}
Command output: ./CMakeLists.txt
./src/CMakeLists.txt
Assistant internal thought: Found CMake configuration. Checking build requirements
Assistant executed command: {"command": "cat CMakeLists.txt"}
Command output: cmake_minimum_required(VERSION 3.10)
project(MyProject)
find_package(Boost REQUIRED)
add_subdirectory(src)
Assistant internal thought: Project requires Boost. Setting up build directory and running CMake
Assistant executed command: {"command": "mkdir build && cd build && cmake .."}
Command output: -- The C compiler identification is GNU 9.4.0
-- Found Boost: /usr/lib/x86_64-linux-gnu/cmake/Boost-1.71.0
-- Configuring done
-- Generating done
-- Build files have been written to: /project/build
Assistant executed command: {"command": "cmake --build build"}
Command output: [ 20%] Building CXX object src/CMakeFiles/mylib.dir/utils.cpp.o
[ 40%] Building CXX object src/CMakeFiles/mylib.dir/parser.cpp.o
[ 60%] Linking CXX static library libmylib.a
[ 80%] Building CXX object src/CMakeFiles/myapp.dir/main.cpp.o
[100%] Linking CXX executable myapp

Output:
{
    "exist_build": true,
    "command_summary": "Project uses CMake build system with Boost dependency. Required steps:\n1. Create and enter build directory: mkdir build && cd build\n2. Generate build files: cmake ..\n3. Build project: cmake --build build\nMultiple CMakeLists.txt files indicate hierarchical project structure.",
    "fail_log": ""
}

Example 2: Project without Build System
Input:
Assistant internal thought: Looking for common build system files
Assistant executed command: {"command": "find . -maxdepth 1 -type f -name 'Makefile' -o -name 'CMakeLists.txt' -o -name 'package.json' -o -name 'build.gradle'"}
Command output: 
Assistant internal thought: No build files found. Checking source directory structure
Assistant executed command: {"command": "tree ."}
Command output: .
└── src
    └── script.py

Output:
{
    "exist_build": false,
    "command_summary": "No build system detected. Project contains only Python scripts without any build configuration.",
    "fail_log": ""
}

Example 3: Node.js Project with Failed Build
Input:
Assistant internal thought: Checking Node.js build configuration
Assistant executed command: {"command": "cat package.json"}
Command output: {
  "name": "web-app",
  "scripts": {
    "build": "webpack --mode production",
    "start": "node dist/server.js"
  },
  "dependencies": {
    "express": "^4.17.1"
  },
  "devDependencies": {
    "webpack": "^5.75.0",
    "webpack-cli": "^4.10.0"
  }
}
Assistant internal thought: Found webpack build configuration. Installing dependencies and building
Assistant executed command: {"command": "npm install"}
Command output: added 245 packages in 15s
Assistant executed command: {"command": "npm run build"}
Command output: ERROR in ./src/app.js
Module not found: Error: Can't resolve './config' in '/project/src'
 @ ./src/app.js 3:15-33

webpack 5.75.0 compiled with 1 error

Output:
{
    "exist_build": true,
    "command_summary": "Project uses webpack for building Node.js application. Required steps:\n1. Install dependencies: npm install\n2. Build project: npm run build",
    "fail_log": "ERROR in ./src/app.js\nModule not found: Error: Can't resolve './config' in '/project/src'\n @ ./src/app.js 3:15-33"
}

Example 4: Java Project with Maven
Input:
Assistant internal thought: Looking for Maven build configuration
Assistant executed command: {"command": "find . -name pom.xml"}
Command output: ./pom.xml
Assistant internal thought: Found Maven project. Checking configuration and building
Assistant executed command: {"command": "cat pom.xml"}
Command output: <?xml version="1.0" encoding="UTF-8"?>
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <version>1.0-SNAPSHOT</version>
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
            <version>2.7.0</version>
        </dependency>
    </dependencies>
</project>
Assistant executed command: {"command": "mvn clean install"}
Command output: [INFO] Scanning for projects...
[INFO] Building my-app 1.0-SNAPSHOT
[INFO] --- maven-clean-plugin:3.1.0:clean (default-clean) @ my-app ---
[INFO] --- maven-resources-plugin:3.2.0:resources (default-resources) @ my-app ---
[INFO] --- maven-compiler-plugin:3.8.1:compile (default-compile) @ my-app ---
[INFO] --- maven-jar-plugin:3.2.0:jar (default-jar) @ my-app ---
[INFO] --- maven-install-plugin:2.5.2:install (default-install) @ my-app ---
[INFO] BUILD SUCCESS

Output:
{
    "exist_build": true,
    "command_summary": "Project uses Maven build system with Spring Boot dependency. Required steps:\n1. Clean and build project: mvn clean install\nPOM file indicates it's a Spring Boot web application.",
    "fail_log": ""
}
""".replace("{", "{{").replace("}", "}}")

  def __init__(self, model: BaseChatModel):
    """Initializes the GeneralBuildSummarizationNode with an analysis model.

    Sets up the build summarizer with a prompt template and structured output
    model for analyzing build execution histories.

    Args:
      model: Language model instance that will be used for analyzing build
        histories and generating structured summaries. Must be a
        BaseChatModel implementation.
    """
    prompt = ChatPromptTemplate.from_messages(
      [("system", self.SYS_PROMPT), ("human", "{build_history}")]
    )
    structured_llm = model.with_structured_output(BuildClassification)
    self.model = prompt | structured_llm
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.general_build_summarization_node")

  def format_build_history(self, build_messages: Sequence[BaseMessage]):
    """Formats build execution messages into a structured history.

    Processes various message types (AI, Tool) into a chronological sequence
    of build execution steps and their outputs.

    Args:
      build_messages: Sequence of messages from build execution attempts.
        Can include AIMessage and ToolMessage types.

    Returns:
      List of formatted strings representing the build execution history,
      including internal thoughts, executed commands, and their outputs.
    """
    formatted_messages = []
    for message in build_messages:
      if isinstance(message, AIMessage):
        if message.content:
          formatted_messages.append(f"Assistant internal thought: {message.content}")
        if message.additional_kwargs and message.additional_kwargs["tool_calls"]:
          for tool_call in message.additional_kwargs["tool_calls"]:
            formatted_messages.append(f"Assistant executed command: {tool_call.function.arguments}")
      elif isinstance(message, ToolMessage):
        formatted_messages.append(f"Command output: {message.content}")
    return formatted_messages

  def __call__(self, state: IssueAnswerAndFixState):
    """Processes build state to generate structured build analysis.

    Analyzes the build execution history to determine build system presence,
    required commands, and any failures encountered.

    Args:
      state: Current state containing build execution messages and history.

    Returns:
      Dictionary that updates the statecontaining:
      - exist_build: Boolean indicating if a build system exists
      - build_command_summary: String describing build system and required commands
      - build_fail_log: String containing error logs (empty if successful)
    """
    build_history = "\n".join(self.format_build_history(state["build_messages"]))
    response = self.model.invoke({"build_history": build_history})
    self._logger.debug(f"GeneralBuildSummarizeNode response:\n{response}")

    return {
      "exist_build": response.exist_build,
      "build_command_summary": response.command_summary,
      "build_fail_log": response.fail_log,
    }
