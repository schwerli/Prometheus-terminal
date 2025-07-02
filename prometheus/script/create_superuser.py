import argparse

from prometheus.app.db import create_superuser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a superuser account.")
    parser.add_argument("--username", required=True, help="Superuser username")
    parser.add_argument("--email", required=True, help="Superuser email")
    parser.add_argument("--password", required=True, help="Superuser password")
    parser.add_argument("--github_token", required=False, help="Optional GitHub token")

    args = parser.parse_args()

    create_superuser(
        username=args.username,
        email=args.email,
        password=args.password,
        github_token=args.github_token,
    )
