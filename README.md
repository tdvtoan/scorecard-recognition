# Overview

Golf Scorecard Recognition running by Flask.

## Quick Start

```bash
cd /var/opt
git clone https://github.com/khanhicetea/flask-skeleton.git ledis
cd /var/opt/ledis
```

build image:
```bash
docker build -t sc:1.0 .
```

run container
```bash
docker run --name sc -d  -p 80:5000 sc:1.0
```


run auto-reloading container
```
docker run --name sc -d -p 80:5000  -v /var/opt/ledis/project/:/usr/src/app/project sc:1.0
```


## Flask particulars

### Basics

1. Install the requirements
2. Install opencv 3.0

### Set Environment Variables

Update *config.py*, and then run:

```sh
$ export APP_SETTINGS="project.config.DevelopmentConfig"
```

or

```sh
$ export APP_SETTINGS="project.config.ProductionConfig"
```

### Run the Application

```sh
$ python manage.py runserver
```

### Testing

Without coverage:

```sh
$ python manage.py test
```

With coverage:

```sh
$ python manage.py cov
```
