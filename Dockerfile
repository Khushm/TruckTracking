FROM python:3

MAINTAINER ____

WORKDIR /app
COPY requirements.txt ./
RUN pip install cython
RUN pip install numpy
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "main.py"]
CMD ["python", "main.py", "--host=127.0.0.1"]
ENTRYPOINT ["echo", "Hello World"]
ENTRYPOINT ["python", "main.py"]

#CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
#CMD echo 'Hello world'