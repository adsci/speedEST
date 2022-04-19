FROM python:3.10-slim-bullseye
LABEL description="speedEST container"
LABEL maintainer="Adam Sciegaj @adsci"
EXPOSE 8501
WORKDIR /app


RUN mkdir ~/.streamlit
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > /root/.streamlit/config.toml'

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH "/"

COPY src/ ./src/
ENTRYPOINT ["streamlit", "run"]
CMD ["src/speedEST.py"]