FROM python:3.11-slim
# FROM public/ecr.aws/lambda/python:3.12

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY models/model.pkl .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

