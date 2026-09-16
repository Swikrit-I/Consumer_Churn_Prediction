FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /consumer_churn

COPY pyproject.toml /consumer_churn/
COPY src /consumer_churn/src
RUN pip install --no-cache-dir -e .

EXPOSE 80

ENV PYTHONPATH=/consumer_churn/src
COPY . /consumer_churn/
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]
