import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from prometheus.docker.user_defined_container import UserDefinedContainer


@pytest.fixture
def temp_project_dir():
    # Create a temporary directory with some test files
    temp_dir = Path(tempfile.mkdtemp())
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def container(temp_project_dir):
    return UserDefinedContainer(
        temp_project_dir,
        "/app",
        ["pip install -r requirements.txt", "python setup.py build"],
        ["pytest tests/"],
        "FROM python:3.9\nWORKDIR /app\nCOPY . /app/",
        None,
    )


def test_initialization(container, temp_project_dir):
    """Test that the container is initialized correctly"""
    assert isinstance(container.tag_name, str)
    assert container.tag_name.startswith("prometheus_user_defined_container_")
    assert container.project_path != temp_project_dir
    assert (container.project_path / "test.txt").exists()


def test_get_dockerfile_content(container):
    dockerfile_content = container.get_dockerfile_content()

    assert dockerfile_content

    # Check for key elements in the Dockerfile
    assert "FROM python:3.9" in dockerfile_content
    assert "WORKDIR /app" in dockerfile_content
    assert "COPY . /app/" in dockerfile_content


def test_run_build(container):
    """Test that build commands are executed correctly"""
    container.execute_command = Mock()
    container.execute_command.side_effect = ["Output 1", "Output 2"]

    build_output = container.run_build()

    # Verify execute_command was called for each build command
    assert container.execute_command.call_count == 2
    container.execute_command.assert_any_call("pip install -r requirements.txt")
    container.execute_command.assert_any_call("python setup.py build")

    # Verify output format
    expected_output = (
        "$ pip install -r requirements.txt\nOutput 1\n$ python setup.py build\nOutput 2\n"
    )
    assert build_output == expected_output


def test_run_test(container):
    """Test that test commands are executed correctly"""
    container.execute_command = Mock()
    container.execute_command.return_value = "Test passed"

    test_output = container.run_test()

    # Verify execute_command was called for the test command
    container.execute_command.assert_called_once_with("pytest tests/")

    # Verify output format
    expected_output = "$ pytest tests/\nTest passed\n"
    assert test_output == expected_output
