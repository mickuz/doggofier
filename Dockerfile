FROM python:3.7

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY ./doggofier /app/doggofier
COPY ./models /app/models
COPY ./run.py /app/run.py

WORKDIR /app

EXPOSE 5000

ENTRYPOINT [ "python" ]
CMD [ "run.py" ]

