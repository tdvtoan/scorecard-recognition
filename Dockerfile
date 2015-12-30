FROM python:2.7
MAINTAINER Toan Tran <tdvtoan@gmail.com>

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libjpeg62-turbo-dev libtiff5-dev libjasper-dev libpng12-dev python-opencv python-dev libatlas-base-dev gfortran \
    python-tk \ 
    tk8.5-dev \
    tcl8.5-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/{apt,dpkg,cache,log}/
COPY vendor /usr/local/src
# Application requirements
ADD pip-cache/ /usr/src/pip-cache/

RUN cd /usr/src/ \
    && pip install --no-index --find-links=pip-cache -r pip-cache/requirements.txt \
    && rm -rf pip-cache/

# Build and install opencv
RUN	cd /usr/local/src/opencv \
	&& mkdir build \
	&& cd build \
	&& cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=ON \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=/usr/local/src/opencv_contrib/modules \
	-D BUILD_EXAMPLES=ON .. \
	&& make -j8 \
	&& make install \
	&& ldconfig \
	&& rm -rf /usr/local/src/opencv

RUN mkdir -p /usr/src/app/{project, train_data}
WORKDIR /usr/src/app
ENV APP_SETTINGS=project.config.DevelopmentConfig

# Add src
COPY train_data /usr/src/app/train_data
COPY manage.py /usr/src/app
COPY project/ /usr/src/app/project

CMD ["python","./manage.py", "runserver","-h","0.0.0.0","-p","5000"]