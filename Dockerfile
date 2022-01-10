FROM registry.baidubce.com/paddlepaddle/paddle:2.2.1-gpu-cuda11.2-cudnn8

WORKDIR /app
COPY requirements.txt ./
RUN pip install cython
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "main.py"]
