import enum
from pathlib import Path


class FileType(enum.StrEnum):
    """Enum of all tree-sitter supported file types"""

    BASH = "bash"
    C = "c"
    CSHARP = "csharp"
    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    KOTLIN = "kotlin"
    PHP = "php"
    PYTHON = "python"
    SQL = "sql"
    YAML = "yaml"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_path(cls, path: Path):
        match path.suffix:
            case ".sh":
                return cls.BASH
            case ".c":
                return cls.C
            case ".cs":
                return cls.CSHARP
            case ".cpp" | ".cc" | ".cxx":
                return cls.CPP
            case ".go":
                return cls.GO
            case ".java":
                return cls.JAVA
            case ".js":
                return cls.JAVASCRIPT
            case ".kt":
                return cls.KOTLIN
            case ".php":
                return cls.PHP
            case ".py":
                return cls.PYTHON
            case ".sql":
                return cls.SQL
            case ".yaml" | ".yml":
                return cls.YAML
            case _:
                return cls.UNKNOWN
