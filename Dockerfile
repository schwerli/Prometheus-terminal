FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip \
    && pip install hatchling \
    && pip install .

EXPOSE 8000

CMD ["uvicorn", "prometheus.app.main:app", "--host", "0.0.0.0", "--port", "8000"]