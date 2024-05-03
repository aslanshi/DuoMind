FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]