FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
  
ENV TZ Asia/Seoul
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get -y install python3 \
    python3-pip \
    python3-dev \
    git ssh vim

# Install Dependencies of Miniconda
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN echo "root:password" | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes #prohibit-password/' /etc/ssh/sshd_config

RUN mkdir -p ~/miniconda3
RUN apt-get install -y wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh
RUN ~/miniconda3/bin/conda init bash
RUN ~/miniconda3/bin/conda init zsh

RUN service ssh restart

ENTRYPOINT [ "/bin/bash" ]
