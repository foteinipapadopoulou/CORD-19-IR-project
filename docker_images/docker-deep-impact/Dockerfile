FROM python:3.7.12-slim-bullseye

WORKDIR /usr/src/app
COPY requirements_for_di.txt ./
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/

RUN apt-get -y install git

RUN pip install -r requirements_for_di.txt

COPY .. .
RUN chmod +x /usr/src/app