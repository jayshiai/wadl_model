FROM python:3.11-alpine

ADD requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt


# Make port 5000 available to the world outside this container
EXPOSE 5000

ADD app.py /var/server/app.py

CMD python /var/server/app.py
