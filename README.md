# Grokking Challenge Finale

In this challenge, we will attempt to build an in-memory datastore. To make things
simple, let's follow the popular in-memory database Redis and try to build a
stripped down version of it.

Your job is to build a **Ledis** (Lite Redis) datastore that supports these
data structures: **string**, **list**, **set**.

The use of ready-made databases or libraries that handle the main gist of the challenge (Redis, Riak, RocksDB, LevelDB, PostgreSQL, MySQL etc) are not allowed.

However, the use of any other libraries/framework that help with the individual components of your implementation is allowed.

## Quick Start

```bash
cd /var/opt
git clone https://github.com/khanhicetea/flask-skeleton.git ledis
cd /var/opt/ledis
```

build image:
```bash
docker build -t ledis:1.0 .
```

run container
```bash
docker run --name ledis -d -e "LEDIS_PATH=/var/data/ledis" -v /var/data/ledis:/var/data/ledis -p 80:5000 ledis:1.0
```


run auto-reloading container
```
docker run --name ledis -d -p 80:5000 -e "LEDIS_PATH=/var/data/ledis" -v /var/data/ledis:/var/data/ledis -v /var/opt/ledis/project/:/usr/src/app/project ledis:1.0
```


## Flask particulars

### Set Environment Variables

Update *config.py*, and then run:

```sh
$ export APP_SETTINGS="project.config.DevelopmentConfig"
```

or

```sh
$ export APP_SETTINGS="project.config.ProductionConfig"
```

### Create DB

```sh
$ python manage.py create_db
$ python manage.py db init
$ python manage.py db migrate
$ python manage.py create_admin
$ python manage.py create_data
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
