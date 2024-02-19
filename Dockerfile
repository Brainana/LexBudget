# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

ARG PYTHON_VERSION=3.11.7
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

ENV POETRY_VIRTUALENVS_PATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/streamlit/streamlit-example.git

RUN pip install --upgrade pip
RUN pip install openai
RUN pip install streamlit
RUN pip install trubrics
RUN pip install streamlit_feedback
RUN pip install poetry
COPY pyproject.toml /.
RUN poetry config virtualenvs.in-project true
RUN poetry install --no-root --no-ansi -vvv

# Expose the port that the application listens on.
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["poetry", "run", "streamlit", "run", "BudgetChatbot_AssistAPI.py", "--server.port=8501", "--server.address=0.0.0.0"]
