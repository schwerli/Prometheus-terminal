import logging
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence

from prometheus.app.services.base_service import BaseService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.docker.general_container import GeneralContainer
from prometheus.docker.user_defined_container import UserDefinedContainer
from prometheus.git.git_repository import GitRepository
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.graphs.issue_graph import IssueGraph
from prometheus.lang_graph.graphs.issue_state import IssueType


class IssueService(BaseService):
    def __init__(
        self,
        neo4j_service: Neo4jService,
        repository_service,
        llm_service: LLMService,
        max_token_per_neo4j_result: int,
        working_directory: str,
    ):
        self.neo4j_service = neo4j_service
        self.repository_service = repository_service
        self.llm_service = llm_service
        self.max_token_per_neo4j_result = max_token_per_neo4j_result
        self.working_directory = working_directory
        self.answer_issue_log_dir = Path(self.working_directory) / "answer_issue_logs"
        self.answer_issue_log_dir.mkdir(parents=True, exist_ok=True)

    async def answer_issue(
        self,
        repository_id: int,
        repository: GitRepository,
        knowledge_graph: KnowledgeGraph,
        issue_number: int,
        issue_title: str,
        issue_body: str,
        issue_comments: Sequence[Mapping[str, str]],
        issue_type: IssueType,
        run_build: bool,
        run_existing_test: bool,
        run_reproduce_test: bool,
        number_of_candidate_patch: int,
        dockerfile_content: Optional[str] = None,
        image_name: Optional[str] = None,
        workdir: Optional[str] = None,
        build_commands: Optional[Sequence[str]] = None,
        test_commands: Optional[Sequence[str]] = None,
        push_to_remote: Optional[bool] = None,
    ):
        """
        Processes an issue, generates patches if needed, runs optional builds and tests, and returning the results.

        Args:
            repository_id: The ID of the repository to update.
            repository (GitRepository): The Git repository instance.
            knowledge_graph (KnowledgeGraph): The knowledge graph instance.
            issue_number (int): The number of the issue.
            issue_title (str): The title of the issue.
            issue_body (str): The body of the issue.
            issue_comments (Sequence[Mapping[str, str]]): Comments on the issue.
            issue_type (IssueType): The type of the issue (BUG or QUESTION).
            run_build (bool): Whether to run the build commands.
            run_existing_test (bool): Whether to run existing tests.
            run_reproduce_test (bool): Whether to run reproduce tests.
            number_of_candidate_patch (int): Number of candidate patches to generate.
            dockerfile_content (Optional[str]): Content of the Dockerfile for user-defined environments.
            image_name (Optional[str]): Name of the Docker image.
            workdir (Optional[str]): Working directory for the container.
            build_commands (Optional[Sequence[str]]): Commands to build the project.
            test_commands (Optional[Sequence[str]]): Commands to test the project.
            push_to_remote (Optional[bool]): Whether to push changes to a remote branch.
        Returns:
            Tuple containing:
                - edit_patch (str): The generated patch for the issue.
                - passed_reproducing_test (bool): Whether the reproducing test passed.
                - passed_build (bool): Whether the build passed.
                - passed_existing_test (bool): Whether the existing tests passed.
                - issue_response (str): Response generated for the issue.
        """
        logger = logging.getLogger("prometheus")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.answer_issue_log_dir / f"{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self.repository_service.update_repository_status(repository_id, is_working=True)
        try:
            # Construct the working directory
            if dockerfile_content or image_name:
                container = UserDefinedContainer(
                    repository.get_working_directory(),
                    workdir,
                    build_commands,
                    test_commands,
                    dockerfile_content,
                    image_name,
                )
            else:
                container = GeneralContainer(repository.get_working_directory())
            # Initialize the issue graph with the necessary services and parameters
            issue_graph = IssueGraph(
                advanced_model=self.llm_service.advanced_model,
                base_model=self.llm_service.base_model,
                kg=knowledge_graph,
                git_repo=repository,
                neo4j_driver=self.neo4j_service.neo4j_driver,
                max_token_per_neo4j_result=self.max_token_per_neo4j_result,
                container=container,
                build_commands=build_commands,
                test_commands=test_commands,
            )
            # Invoke the issue graph with the provided parameters
            output_state = issue_graph.invoke(
                issue_title,
                issue_body,
                issue_comments,
                issue_type,
                run_build,
                run_existing_test,
                run_reproduce_test,
                number_of_candidate_patch,
            )

            if output_state["issue_type"] == IssueType.BUG:
                # push to remote if requested
                remote_branch_name = None
                if output_state["edit_patch"] and push_to_remote:
                    remote_branch_name = f"prometheus_fix_{uuid.uuid4().hex[:10]}"
                    await repository.create_and_push_branch(
                        remote_branch_name, f"Fixes #{issue_number}", output_state["edit_patch"]
                    )

                return (
                    remote_branch_name,
                    output_state["edit_patch"],
                    output_state["passed_reproducing_test"],
                    output_state["passed_build"],
                    output_state["passed_existing_test"],
                    output_state["issue_response"],
                )
            elif output_state["issue_type"] == IssueType.QUESTION:
                return (
                    None,
                    None,
                    False,
                    False,
                    False,
                    output_state["issue_response"],
                )

            raise ValueError(
                f"Unknown issue type: {output_state['issue_type']}. Expected BUG or QUESTION."
            )
        except Exception as e:
            logger.error(f"Error in answer_issue: {str(e)}\n{traceback.format_exc()}")
            return None, None, False, False, False, None
        finally:
            self.repository_service.update_repository_status(repository_id, is_working=False)
            logger.removeHandler(file_handler)
            file_handler.close()
