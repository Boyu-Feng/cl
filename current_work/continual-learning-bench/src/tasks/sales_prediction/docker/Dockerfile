FROM python:3.12-slim

RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    numpy==2.1.3 \
    scipy==1.14.1 \
    scikit-learn==1.6.1 \
    matplotlib==3.10.0

WORKDIR /app
RUN mkdir -p /app/data

COPY check_time.py /app/check_time.py

CMD ["bash"]
