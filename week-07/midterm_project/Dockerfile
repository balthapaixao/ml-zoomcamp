FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv
WORKDIR /app
COPY ["requirements.txt", "./"]
RUN pip install -r requirements.txt

COPY ["predict.py", "model1.bin", "dv.bin", "./"]

EXPOSE 9696
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "scripts.predict:app"]