FROM python:3.12.3-slim

WORKDIR /app


RUN  apt-get update
RUN  apt-get install build-essential -y
COPY pyproject.toml .
RUN pip install --no-cache-dir uv
RUN uv pip install --system --no-cache -r pyproject.toml
RUN pip install -qU chromadb langchain-chroma



COPY src/client/ ./client/
COPY src/schema/ ./schema/
COPY src/streamlit_app.py .

CMD ["streamlit", "run", "streamlit_app.py"]
