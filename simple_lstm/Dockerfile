FROM ubuntu:16.04
MAINTAINER Prabhu Manicakavelu <prabhu.manickavelu@gmail.com>


RUN apt-get update && apt-get upgrade -y

RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    g++ \
    bc \
    cmake \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    wget \
    python \
    python-pip \
    libsqlite3-dev \
    libbz2-dev

RUN pip install virtualenv

RUN wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tar.xz
RUN tar -xf Python-3.7.4.tar.xz
RUN cd Python-3.7.4 && ./configure && make altinstall

RUN virtualenv -p python3.7 /python_env

RUN /python_env/bin/pip install Keras tensorflow==2.0.0 Flask

ADD . /app
WORKDIR /app
CMD ["/python_env/bin/python","web_service.py"]