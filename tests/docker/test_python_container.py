import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from prometheus.docker.python_container import PythonContainer, PythonProjectConfig


@pytest.fixture
def temp_project_dir():
  """Create a temporary project directory for testing."""
  temp_dir = Path(tempfile.mkdtemp())
  project_path = temp_dir / "test_project"
  project_path.mkdir(parents=True)
  yield project_path
  shutil.rmtree(temp_dir)


@pytest.fixture
def mock_docker():
  """Mock docker client and container."""
  with patch("docker.from_env") as mock_docker_client:
    # Set up mock container
    mock_container = MagicMock()
    mock_container.exec_run.return_value.output = b"Test output"

    # Set up mock client
    mock_client = MagicMock()
    mock_client.containers.run.return_value = mock_container
    mock_client.images = MagicMock()
    mock_docker_client.return_value = mock_client

    yield {"client": mock_client, "container": mock_container, "docker_client": mock_docker_client}


@pytest.fixture
def container(temp_project_dir, mock_docker):
  """Create a PythonContainer instance with mocked dependencies."""
  # Actually create the temporary directory structure
  container_temp_dir = Path(tempfile.mkdtemp())
  container_project_dir = container_temp_dir / "project"
  container_project_dir.mkdir(parents=True)

  def mock_copytree(src, dst, *args, **kwargs):
    dst.mkdir(parents=True, exist_ok=True)
    return dst

  with patch("shutil.copytree", side_effect=mock_copytree):
    container = PythonContainer(temp_project_dir)
    # Ensure the project directory exists
    Path(container.project_path).mkdir(parents=True, exist_ok=True)
    container.container = mock_docker["container"]
    return container


def test_init_creates_temp_directory(temp_project_dir):
  def mock_copytree(src, dst, *args, **kwargs):
    # Actually create the destination directory
    dst.mkdir(parents=True, exist_ok=True)

  with patch("shutil.copytree", side_effect=mock_copytree) as mock_copy:
    container = PythonContainer(temp_project_dir)
    mock_copy.assert_called_once()
    assert Path(container.project_path).exists()


@pytest.mark.parametrize(
  "files,content,expected",
  [
    (
      ["setup.py", "requirements.txt"],
      "import pytest",
      {
        "has_setup_py": True,
        "has_pyproject_toml": False,
        "has_requirements_txt": True,
        "has_pytest": True,
        "has_unittest": False,
      },
    ),
    (
      ["pyproject.toml"],
      "import unittest",
      {
        "has_setup_py": False,
        "has_pyproject_toml": True,
        "has_requirements_txt": False,
        "has_pytest": False,
        "has_unittest": True,
      },
    ),
  ],
)
def test_get_project_config(temp_project_dir, files, content, expected):
  def mock_copytree(src, dst, *args, **kwargs):
    dst.mkdir(parents=True, exist_ok=True)

  def create_path_mock(name):
    mock = MagicMock()
    mock.name = name
    mock.exists.return_value = name in files
    mock.absolute.return_value = mock
    mock.__truediv__ = lambda self, other: create_path_mock(other)
    mock.rglob = lambda pattern: [create_path_mock("test_file.py")]
    return mock

  # Setup file reading mock
  mocked_content = mock_open(read_data=content)

  with patch("shutil.copytree", side_effect=mock_copytree), patch("builtins.open", mocked_content):
    # Create a path class mock that returns our path mock
    path_class_mock = MagicMock()
    path_class_mock.side_effect = create_path_mock

    with patch("prometheus.docker.python_container.Path", path_class_mock):
      container = PythonContainer(temp_project_dir)
      config = container.project_config

      # Debug print
      print(f"\nTest case: {files}")
      print(f"Content: {content}")
      print(f"Expected: {expected}")
      print(f"Actual config: has_pytest={config.has_pytest}, has_unittest={config.has_unittest}")

      for key, value in expected.items():
        assert (
          getattr(config, key) == value
        ), f"Failed assertion for {key}, expected {value} but got {getattr(config, key)}"


def test_build_docker_image(container, mock_docker):
  # Create the project directory if it doesn't exist
  Path(container.project_path).mkdir(parents=True, exist_ok=True)

  container._build_docker_image()

  mock_docker["client"].images.build.assert_called_once()
  build_args = mock_docker["client"].images.build.call_args[1]
  assert build_args["tag"] == container.tag_name


@pytest.mark.parametrize(
  "test_framework,expected_command",
  [(True, "python -m pytest -v"), (False, "python -m unittest discover -v")],
)
def test_run_tests(container, mock_docker, test_framework, expected_command):
  # Patch the instance attribute instead of the class attribute
  container.project_config = PythonProjectConfig(
    has_setup_py=False,
    has_pyproject_toml=False,
    has_requirements_txt=False,
    has_pytest=test_framework,
    has_unittest=not test_framework,
  )

  result = container.run_tests()

  mock_docker["container"].exec_run.assert_called_with(
    expected_command, workdir="/app", environment={"PYTHONPATH": "/app"}
  )
  assert result == "Test output"


def test_cleanup(container, mock_docker):
  project_path = Path(container.project_path)

  with patch("shutil.rmtree") as mock_rmtree:
    container.cleanup()

    mock_docker["container"].stop.assert_called_once()
    mock_docker["container"].remove.assert_called_once()
    mock_docker["client"].images.remove.assert_called_once_with(container.tag_name)

    mock_rmtree.assert_called_once_with(project_path)
