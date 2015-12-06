FROM python:2-onbuild
ENV APP_SETTINGS=project.config.DevelopmentConfig
CMD ["python","./manage.py", "runserver","-h","0.0.0.0","-p","5000"]
