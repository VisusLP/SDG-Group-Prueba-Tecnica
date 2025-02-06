import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shap

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

TARGET = "churn"
IDENTIFIER = "Customer_ID"


def get_postgres_connection():
    # Create and cache the hook and engine
    hook = PostgresHook(postgres_conn_id="postgres_default")
    engine = hook.get_sqlalchemy_engine()

    return engine


def load_from_postgres(table_name):
    """
    Load a Pandas DataFrame from PostgreSQL.
    """
    engine = get_postgres_connection()
    table_name = table_name.lower()
    print(f"Loading DataFrame from {table_name} in PostgreSQL.")
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)


def load_model_from_postgres(model_name, table_name='lgbm_models'):
    """
    Load a trained model from PostgreSQL.
    """
    engine = get_postgres_connection()

    model_name = model_name.lower()
    table_name = table_name.lower()
    print(f"Loading model '{model_name}' from {table_name} in PostgreSQL.")

    # Retrieve the model binary data
    with engine.connect() as conn:
        result = conn.execute(
            f"SELECT model_data FROM {table_name} WHERE model_name = %s",
            (model_name,)
        ).fetchone()

    if result is None:
        raise ValueError(
            f"Model '{model_name}' not found in table '{table_name}'.")

    # Deserialize the model using pickle
    model_data = result[0]
    model = pickle.loads(model_data)

    return model


def run_model_analysis():
    # Uncomment this to load the model from MLFlow
    # mlflow.set_tracking_uri("http://mlflow:5000")
    # model_uri = "models:/Customer Churn Final Model/latest"
    # # Load the model
    # model = mlflow.pyfunc.load_model(model_uri)
    print("Running model analysis")
    model = load_model_from_postgres(f"final_model")
    x_valid = load_from_postgres("x_valid")
    y_valid = load_from_postgres("y_valid")

    # Plot feature importance
    print("Plotting feature importance")
    lgb.plot_importance(model, importance_type='gain', figsize=(6, 6), max_num_features=20, title='Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('/opt/airflow/dags/graphs/feature_importances.png')
    plt.close()

    # Calculate SHAP values
    print("Calculating SHAP values")
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(x_valid)
    # Plot SHAP summary plot
    shap.summary_plot(shap_values, x_valid)
    plt.savefig('/opt/airflow/dags/graphs/shap_summary_plot.png')
    plt.close()

    # Calculate lift table
    print("Calculating lift table")
    y_pred = model.predict(x_valid)
    df = pd.DataFrame({
        'probability': y_pred,
        'target': y_valid.squeeze()  # Flatten y_valid to 1D
    })
    df = df.sort_values(by='probability', ascending=False)
    df['decile'] = pd.qcut(df['probability'].rank(method='first', ascending=False), 10, labels=False)

    positives = df['target'].sum()
    df_group = df.groupby('decile')['target'].agg(['count', 'sum'])

    df_group['min_probability'] = df.groupby('decile')['probability'].min()
    df_group['max_probability'] = df.groupby('decile')['probability'].max()
    df_group['percentage'] = df_group['sum'] / positives * 100
    df_group['gain'] = df_group['percentage'].cumsum() / (df_group['count'].cumsum() / df_group['count'].sum()) / 100

    gain_d10 = df_group.loc[0, 'gain']
    print(f"Decile 10 gain: {gain_d10}")
    df_group.to_csv('/opt/airflow/dags/graphs/lift_table.csv')

    # Calculate confusion matrix
    print("Calculating confusion matrix")
    cm = confusion_matrix(y_valid.squeeze(), (y_pred > 0.5).astype(int))
    print("Plotting confusion matrix")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array([0, 1]))
    fig, ax = plt.subplots(figsize=(3, 3))
    disp.plot(ax=ax)
    plt.title('Confusion Matrix')
    plt.savefig('/opt/airflow/dags/graphs/confusion_matrix.png')
    plt.close()
    print("Model analysis complete")

# DAG Definition
default_args = {"owner": "airflow", "start_date": datetime(2025, 1, 23)}
dag = DAG(
    "evaluate_final_model",
    default_args=default_args,
    max_active_tasks=1,
    schedule_interval=None,
)
start = PythonOperator(
    task_id="run_model_analysis",
    python_callable=run_model_analysis,
    dag=dag,)

start