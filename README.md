# SDG-Group Technical Challenge

## Setup Instructions

To set up the environment, follow these steps:

1. Create necessary directories:
    ```sh
    mkdir -p ./dags ./logs ./plugins ./config
    ```

2. Set the Airflow user ID:
    ```sh
    echo -e "AIRFLOW_UID=$(id -u)" > .env
    ```

3. Initialize Airflow:
    ```sh
    docker compose up airflow-init
    ```

4. Start the Airflow services:
    ```sh
    docker compose up -d
    ```

5. Connect Postgres to Airflow:

Go to `localhost:8080`, click on the menu `Admin > Connections` and create a new one with the following settings:

- **Connection Id**: `postgres_default`
- **Connection Type**: `Postgres`
- **Host**: `postgres`
- **Database**: `airflow`
- **Login**: `airflow`
- **Password**: `airflow`
- **Port**: `5432`

6. Connect Prometheus to Grafana and load the dashboard

Go to `localhost:3000`, click on the menu `Connections` and search Prometheus.
Fill the Prometheus server URL (Should be `http://localhost:9090` but sometimes filling the entire ip address works better) and save it.

Open the dashboard code located in [`/grafana_dashboard_json/dashboard.json`](./grafana_dashboard_json/dashboard.json) and copy it.
Then go to `Dashboards > New > Import` and paste the JSON code.
Finally, open the dashboard created (You might need to enter inside each panel the first time for it to load)

## Additional Configuration

The maximum amount of cores is set by `AIRFLOW__CORE__PARALLELISM` in the [`docker-compose.yaml`](./docker-compose.yaml). If your machine is more powerful, you can increase it.

Additionally, all DAGs have `max_active_tasks=1` configured to reduce resource usage. You can change this setting if you wish.