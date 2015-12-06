# Ledis - coding challenge

## Quick Start

build image:
```bash
docker build -t ledis:1.0 .
```

run container
```bash
docker run --name ledis -d -p 80:5000 ledis:1.0
```

run git-sync
```bash
docker run -d -e "GIT_SYNC_REPO=https://github.com/khanhicetea/flask-skeleton.git" -e "GIT_SYNC_DEST=/git" -e "GIT_SYNC_BRANCH=master" -e "GIT_SYNC_DEST=/git" -v /var/opt/ledis:/git --name=git-sync git-sync
```

run auto-reloading container
```
docker run --name ledis -d -p 80:5000 -v /var/opt/ledis/project/:/usr/src/app/project ledis:1.0
```

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
