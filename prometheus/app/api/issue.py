from fastapi import APIRouter, HTTPException, Request

from prometheus.app.models.requests.issue import IssueRequest

router = APIRouter()


@router.post(
    "/answer/",
    summary="Process and generate a response for an issue",
    description="Analyzes an issue, generates patches if needed, runs optional builds and tests, and can push changes to a remote branch.",
    response_description="Returns the patch, test results, and issue response",
)
def answer_issue(issue: IssueRequest, request: Request):
    if not request.app.state.service["knowledge_graph_service"].exists():
        raise HTTPException(
            status_code=404,
            detail="A repository is not uploaded, use /repository/ endpoint to upload one",
        )

    if issue.dockerfile_content or issue.image_name:
        if issue.workdir is None:
            raise HTTPException(
                status_code=400,
                detail="workdir must be provided for user defined environment",
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
    return {
        "patch": patch,
        "passed_reproducing_test": passed_reproducing_test,
        "passed_build": passed_build,
        "passed_existing_test": passed_existing_test,
        "issue_response": issue_response,
        "remote_branch_name": remote_branch_name,
    }
