FROM python:3.10-slim-buster

RUN pip install --upgrade pip
WORKDIR /app
COPY [".", "./"]
RUN pip install -r requirements.txt

EXPOSE 9696
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "scripts.main:app"]