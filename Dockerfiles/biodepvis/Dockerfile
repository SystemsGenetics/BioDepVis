FROM nvidia/cuda:9.0-devel
MAINTAINER Ben Shealy <btsheal@clemson.edu>

# create docker user
RUN useradd --create-home --shell /bin/bash --groups sudo docker

# install package dependencies
RUN apt-get update -qq \
	&& apt-get install -qq -y git libxv1 qt5-default

# install turbovnc
ADD turbovnc_2.2.1_amd64.deb .

RUN dpkg -i turbovnc_2.2.1_amd64.deb \
	&& rm turbovnc_2.2.1_amd64.deb

# install virtualgl
ADD virtualgl_2.6.1_amd64.deb .

RUN dpkg -i virtualgl_2.6.1_amd64.deb \
	&& rm virtualgl_2.6.1_amd64.deb

# install BioDepVis
WORKDIR /opt

RUN git clone https://github.com/SystemsGenetics/BioDepVis.git \
	&& cd BioDepVis \
	&& make -s install

WORKDIR /home/docker
