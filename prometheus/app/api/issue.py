from fastapi import APIRouter, Request

from prometheus.app.decorators.require_login import requireLogin
from prometheus.app.models.requests.issue import IssueRequest
from prometheus.app.models.response.issue import IssueResponse
from prometheus.app.models.response.response import Response
from prometheus.exceptions.server_exception import ServerException

router = APIRouter()


@router.post(
    "/answer/",
    summary="Process and generate a response for an issue",
    description="Analyzes an issue, generates patches if needed, runs optional builds and tests, and can push changes "
    "to a remote branch.",
    response_description="Returns the patch, test results, and issue response",
    response_model=Response[IssueResponse],
)
@requireLogin
def answer_issue(issue: IssueRequest, request: Request) -> Response[IssueResponse]:
    if not request.app.state.service["knowledge_graph_service"].exists():
        raise ServerException(
            code=404,
            message="A repository is not uploaded, use /repository/ endpoint to upload one",
        )

    if issue.dockerfile_content or issue.image_name:
        if issue.workdir is None:
            raise ServerException(
                code=400,
                message="workdir must be provided for user defined environment",
            )

    (
        remote_branch_name,
        patch,
        passed_reproducing_test,
        passed_build,
        passed_existing_test,
        issue_response,
    ) = request.app.state.service["issue_service"].answer_issue(
        issue_number=issue.issue_number,
        issue_title=issue.issue_title,
        issue_body=issue.issue_body,
        issue_comments=issue.issue_comments if issue.issue_comments else [],
        issue_type=issue.issue_type,
        run_build=issue.run_build,
        run_existing_test=issue.run_existing_test,
        number_of_candidate_patch=issue.number_of_candidate_patch,
        dockerfile_content=issue.dockerfile_content,
        image_name=issue.image_name,
        workdir=issue.workdir,
        build_commands=issue.build_commands,
        test_commands=issue.test_commands,
        push_to_remote=issue.push_to_remote,
    )
    return Response(
        data=IssueResponse(
            patch=patch,
            passed_reproducing_test=passed_reproducing_test,
            passed_build=passed_build,
            passed_existing_test=passed_existing_test,
            issue_response=issue_response,
            remote_branch_name=remote_branch_name,
        )
    )
