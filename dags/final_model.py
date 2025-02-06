import pickle
import lightgbm as lgb
import mlflow
import optuna
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

TARGET = "churn"
IDENTIFIER = "Customer_ID"
N_OPTUNA_TRIALS = 10


def get_postgres_connection():
    # Create and cache the hook and engine
    hook = PostgresHook(postgres_conn_id="postgres_default")
    engine = hook.get_sqlalchemy_engine()

    return engine


def save_to_postgres(table_name, df, replace=False):
    engine = get_postgres_connection()
    table_name = table_name.lower()
    print(
        f"Saving DataFrame to {table_name} with shape {df.shape} in PostgreSQL.")

    if replace:
        with engine.begin() as conn:  # Ensures transaction safety
            conn.execute(f'DROP TABLE IF EXISTS {table_name} CASCADE;')

    df.to_sql(table_name, engine,
              if_exists="replace" if replace else "append", index=False)


def load_from_postgres(table_name):
    """
    Load a Pandas DataFrame from PostgreSQL.
    """
    engine = get_postgres_connection()
    table_name = table_name.lower()
    print(f"Loading DataFrame from {table_name} in PostgreSQL.")
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)


def save_model_to_postgres(model, model_name, table_name='lgbm_models', replace=False):
    """
    Save a trained model to PostgreSQL as a binary object (pickle format).
    """
    engine = get_postgres_connection()

    model_name = model_name.lower()
    table_name = table_name.lower()
    print(f"Saving model '{model_name}' to {table_name} in PostgreSQL.")

    # Serialize the model using pickle
    model_bytes = pickle.dumps(model)

    # Create the table if it doesn't exist
    if replace:
        with engine.begin() as conn:  # Ensures transaction safety
            conn.execute(f'DROP TABLE IF EXISTS {table_name} CASCADE;')

    # Insert or update the model in the database
    with engine.begin() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                model_id SERIAL PRIMARY KEY,
                model_name VARCHAR(255) UNIQUE,
                model_data BYTEA
            );
            """
        )
        conn.execute(
            f"""
            INSERT INTO {table_name} (model_name, model_data)
            VALUES (%s, %s)
            ON CONFLICT (model_name)
            DO UPDATE SET model_data = EXCLUDED.model_data;
            """,
            (model_name, model_bytes)
        )


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


def load_data():
    data_path = '/opt/airflow/dags/data/dataset.csv'
    df = pd.read_csv(data_path, sep=';', decimal=',')
    print("Dataset loaded.")
    save_to_postgres("original_dataset", df, replace=True)


def categorical_encoding():
    df = load_from_postgres("original_dataset")
    # One-hot encode the 'area' column
    df['area'].fillna('UNKNOWN', inplace=True)
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(df[['area']])
    ohe_cols = ohe.get_feature_names_out(['area'])
    ohe_df = pd.DataFrame(ohe.transform(df[['area']]).toarray(), columns=ohe_cols)
    df = pd.concat([df.copy(), ohe_df.copy()], axis=1)
    df.drop(columns=['area'], inplace=True)
    df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

    # Label encode the remaining categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.to_list()
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    categorical_cols.extend(ohe_cols)

    save_to_postgres("label_encoder", df, replace=True)
    save_to_postgres("categorical_cols", pd.DataFrame(
        {"col": categorical_cols}), replace=True)


def data_split():
    df = load_from_postgres("label_encoder")

    features = [col for col in df.columns if col not in [TARGET, IDENTIFIER]]

    X_train, X_valid, y_train, y_valid = train_test_split(
        df[features], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET])

    save_to_postgres("x_train", X_train, replace=True)
    save_to_postgres("x_valid", X_valid, replace=True)
    save_to_postgres("y_train", y_train, replace=True)
    save_to_postgres("y_valid", y_valid, replace=True)


def train_model():

    X_train = load_from_postgres("x_train")
    X_valid = load_from_postgres("x_valid")
    y_train = load_from_postgres("y_train")
    y_valid = load_from_postgres("y_valid")
    categorical_cols = load_from_postgres("categorical_cols")["col"].tolist()

    cat_cols = [col for col in categorical_cols if col in X_train.columns]

    def objective(trial):
        # Define the parameter search space
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'max_bin': trial.suggest_int('max_bin', 64, 512),
            'min_data_in_bin': trial.suggest_int('min_data_in_bin', 1, 10),
            'verbose': -1,
        }
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train, label=y_train, categorical_feature=cat_cols)
        valid_data = lgb.Dataset(
            X_valid, label=y_valid, categorical_feature=cat_cols, reference=train_data)
        # Train the model
        model = lgb.train(
            param,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(100),
            ],
        )
        # Evaluate performance
        preds = model.predict(X_valid)
        auc = roc_auc_score(y_valid, preds)
        return auc

    # Create the study and optimize
    # 'maximize' since AUC should be high
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, n_jobs=1)
    # Best parameters and score
    print("Best Parameters:", study.best_params)
    print("Best AUC:", study.best_value)

    best_params = study.best_params
    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    final_model = lgb.train(
        best_params,
        lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols),
        valid_sets=[lgb.Dataset(X_valid, label=y_valid,
                                categorical_feature=cat_cols)],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ],
    )
    save_model_to_postgres(final_model, f"final_model")
    save_to_postgres(f"final_best_params", pd.DataFrame(
        study.best_params, index=[0]), replace=True)


def evaluate_model():
    X_valid = load_from_postgres("x_valid")
    y_valid = load_from_postgres("y_valid")
    final_model = load_model_from_postgres("final_model")

    pred_probs = final_model.predict(X_valid)
    pred_labels = (pred_probs > 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_valid, pred_probs),
        "accuracy": accuracy_score(y_valid, pred_labels),
        "f1_score": f1_score(y_valid, pred_labels),
        "precision": precision_score(y_valid, pred_labels),
        "recall": recall_score(y_valid, pred_labels),
        "log_loss": log_loss(y_valid, pred_probs)
    }

    print("Evaluation metrics:", metrics)
    save_to_postgres(f"final_metrics", pd.DataFrame(
        metrics, index=[0]), replace=True)


def get_feature_importance_mlflow(model):
    importance_df = pd.DataFrame({
        "Feature": model.feature_name(),
        "Importance": model.feature_importance(importance_type="gain")
    })

    # Sort by importance and get the top 20
    importances = importance_df.sort_values(by="Importance", ascending=False)

    importances_dict = importances.set_index("Feature")["Importance"].to_dict()
    feature_importance_metrics = {
        f"feature_importance_{feature_name}": imp_value
        for feature_name, imp_value in importances_dict.items()
    }
    return feature_importance_metrics


def register_model():
    model = load_model_from_postgres(f"final_model")
    metrics = load_from_postgres(f"final_metrics")
    X_train = load_from_postgres(f'x_train')
    best_params = load_from_postgres(f"final_best_params")
    best_params = best_params.to_dict(orient='records')[0]
    X_valid = load_from_postgres(f'x_valid')
    X_valid_sample = X_valid.copy()
    X_valid_sample = X_valid_sample.astype("float64")

    signature = infer_signature(X_train, model.predict(X_train))

    feature_importance_metrics = get_feature_importance_mlflow(model)

    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment_name = "Final Customer Churn Predictions"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        print(
            f"Active Experiment: {mlflow.get_experiment_by_name(experiment_name)}")
        mlflow.lightgbm.log_model(model, f"final_model", input_example=X_valid_sample.head(), signature=signature)
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(feature_importance_metrics)

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/lightgbm_final_model"
        registered_model = mlflow.register_model(
            model_uri, f"Customer Churn Final Model")
        print(
            f"Model registered with name: {registered_model.name}, version: {registered_model.version}")


# DAG Definition
default_args = {"owner": "airflow", "start_date": datetime(2025, 1, 23)}
dag = DAG(
    "final_model",
    default_args=default_args,
    max_active_tasks=1,
    schedule_interval=None,
)
start = PythonOperator(
    task_id="load_data",
    python_callable=load_data,
    dag=dag,)

# Label encoding (mandatory step)
encode = PythonOperator(
    task_id="categorical_encoding",
    python_callable=categorical_encoding,
    dag=dag,)

split = PythonOperator(
    task_id="train_test_split",
    python_callable=data_split,
    dag=dag,)

train = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,)

evaluate = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag,
)

register = PythonOperator(
    task_id="register_model",
    python_callable=register_model,
    provide_context=True,
    dag=dag,
)

start >> encode >> split >> train >> evaluate >> register
