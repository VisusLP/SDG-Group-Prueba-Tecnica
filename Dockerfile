FROM apache/airflow:2.10.4
# Switch to root to install system packages
USER root

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir "apache-airflow==2.10.4" -r requirements.txt
RUN pip install 'apache-airflow[statsd]'
# RUN pip install -r requirements.txt
