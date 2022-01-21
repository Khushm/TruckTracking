FROM python:3

WORKDIR /app
COPY requirements.txt ./
RUN pip install cython
RUN pip install numpy
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "main.py"]
