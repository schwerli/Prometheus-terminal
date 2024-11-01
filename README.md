![code coverage](https://github.com/Pantheon-temple/Prometheus/raw/coverage-badge/coverage.svg)

# How to run the project

In this project we use `docker-compose.yml` to run everything. It expects you to set the following enviroment variables:

```
PROMETHEUS_LITELLM_MODEL="..."
PROMETHEUS_LITELLM_ANTHROPIC_API_KEY="..."
PROMETHEUS_GITHUB_ACCESS_TOKEN="..."
```

`PROMETHEUS_LITELLM_MODEL` and `PROMETHEUS_LITELLM_ANTHROPIC_API_KEY` are parameters to [ChatLiteLLM](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.litellm.ChatLiteLLM.html), and `PROMETHEUS_GITHUB_ACCESS_TOKEN` is the GitHub access token.

You can also create `.env` with these varaibles.

Now, simply run `docker compose up`.