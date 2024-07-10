FROM python:3.11-slim-bookworm
LABEL description="speedEST"
EXPOSE 8501
WORKDIR /app

RUN mkdir ~/.streamlit

RUN apt-get update && apt-get install -y procps

COPY requirements/base.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r base.txt
ENV PYTHONPATH "/"

COPY src/ ./src/
COPY .streamlit ./.streamlit
COPY VERSION ./
ENTRYPOINT ["streamlit", "run"]
CMD ["src/Home.py"]