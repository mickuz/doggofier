FROM python:3.7

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY ./doggofier/models.py /app/doggofier/models.py
COPY ./data/categories.json /app/data/categories.json
COPY ./models /app/models
COPY ./app /app/app

WORKDIR /app

CMD gunicorn -b 0.0.0.0:$PORT app.wsgi:app --timeout 0