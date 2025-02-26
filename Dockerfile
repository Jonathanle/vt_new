FROM nvidia/cuda:11.8.0-base-ubuntu22.04
RUN apt-get update && apt-get install -y python3.9 python3-pip


WORKDIR /app
COPY . .

# Add your dependencies
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
