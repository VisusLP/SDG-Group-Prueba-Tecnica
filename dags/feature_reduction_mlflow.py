import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
import pickle

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from mlflow.models import infer_signature
from datetime import datetime
from scipy.stats import zscore
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor


TARGET = "churn"
IDENTIFIER = "Customer_ID"
N_OPTUNA_TRIALS = 10
CORRELATION = False
MULTICOLLINEARITY = False
OUTLIER_OPTION = 'none'

def get_postgres_connection():
    # Create and cache the hook and engine
    hook = PostgresHook(postgres_conn_id="postgres_default")
    engine = hook.get_sqlalchemy_engine()
    
    return engine

def save_to_postgres(table_name, df, replace=False):
    engine = get_postgres_connection()
    table_name = table_name.lower()
    print(f"Saving DataFrame to {table_name} with shape {df.shape} in PostgreSQL.")

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
        raise ValueError(f"Model '{model_name}' not found in table '{table_name}'.")

    # Deserialize the model using pickle
    model_data = result[0]
    model = pickle.loads(model_data)

    return model

def load_data():
    """
    Load the customer churn dataset from a CSV file.
    """
    data_path = '/opt/airflow/dags/data/dataset.csv'
    df = pd.read_csv(data_path, sep=';', decimal=',')
    print("Dataset loaded.")
    save_to_postgres("original_dataset", df, replace=True)


def preprocessing():
    X_train = load_from_postgres("x_train")
    X_valid = load_from_postgres("x_valid")
    y_train = load_from_postgres("y_train")
    y_valid = load_from_postgres("y_valid")
        
    print(f"- Correlation Analysis: {CORRELATION}")
    if CORRELATION:
        print("Dropping correlated features...")
        low_corr_features = load_from_postgres("low_corr_features")["col"].tolist()
        drop_cols = [col for col in low_corr_features if col in X_train.columns]
        X_train = X_train.drop(columns=drop_cols)
        X_valid = X_valid.drop(columns=drop_cols)

    print(f"- Multicollinearity Analysis: {MULTICOLLINEARITY}")
    if MULTICOLLINEARITY:
        print("Dropping features with high VIF...")
        high_vif = load_from_postgres("high_vif")["col"].tolist()
        drop_cols = [col for col in high_vif if col in X_train.columns]
        X_train = X_train.drop(columns=drop_cols)
        X_valid = X_valid.drop(columns=drop_cols)
        
    print(f"- Outlier Detection: {OUTLIER_OPTION}")
    features = [col for col in X_train.columns if col not in [TARGET, IDENTIFIER]]
    if OUTLIER_OPTION == 'none':
        print("Skipping outlier detection...")
    elif OUTLIER_OPTION == 'whole':
        print("Running outlier detection on the whole dataset...")
        # Merge features and target
        train = pd.concat([X_train, y_train], axis=1)
        valid = pd.concat([X_valid, y_valid], axis=1)
        # Add identifier columns
        train["set"] = "train"
        valid["set"] = "valid"
        # Merge datasets
        df = pd.concat([train, valid])
        # Perform outlier detection
        df_cleaned = outlier_detection(df)
        # Separate them back into train and test sets
        train_cleaned = df_cleaned[df_cleaned["set"] == "train"].drop(columns=["set"])
        valid_cleaned = df_cleaned[df_cleaned["set"] == "valid"].drop(columns=["set"])
        # Separate features and target
        X_train = train_cleaned[features]
        y_train = train_cleaned[TARGET]
        X_valid = valid_cleaned[features]
        y_valid = valid_cleaned[TARGET]
        print("Dataset shape after outlier detection:", X_train.shape)
    elif OUTLIER_OPTION == 'train':
        print("Running outlier detection on the training set...")
        train = pd.concat([X_train, y_train], axis=1)
        train = outlier_detection(train)
        X_train = train[features]
        y_train = train[TARGET]
        X_valid = X_valid[features]
    print("Preprocessing completed.")
    save_to_postgres(f'x_train_preprocessed', X_train, replace=True)
    save_to_postgres(f'x_valid_preprocessed', X_valid, replace=True)
    save_to_postgres(f'y_train_preprocessed', y_train, replace=True)
    save_to_postgres(f'y_valid_preprocessed', y_valid, replace=True)


def outlier_detection(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    z_scores = np.abs(zscore(df[numeric_cols]))
    threshold = 3
    df = df[~(z_scores > threshold).any(axis=1)]
    return df

def correlation_analysis(df):
    print("Analyzing correlation...")
    correlation_matrix = df.corr()
    target_corr = correlation_matrix[TARGET].sort_values(ascending=False)
    low_corr_features = target_corr[target_corr.abs() < 0.01].index
    print(f"Dropping of features with low correlation: {low_corr_features.tolist()}")
    return low_corr_features.to_list()

def multicollinearity_analysis(df):
    print("Analyzing multicollinearity...")
    X = df.drop(columns=[TARGET]).fillna(0)
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(
        X.values, i) for i in range(X.shape[1])]
    high_vif = vif_data[vif_data['VIF'] > 10]['Feature']
    print(f"Dropping of features with high VIF: {high_vif.tolist()}")
    return high_vif.to_list()

def label_encoder(**kwargs):
    """
    Loads the data and performs the preprocessing steps.
    It then saves the preprocessed data to X-Com.
    """
    df = load_from_postgres("original_dataset")
    print("Encoding categorical variables...")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    print("Categorical encoding completed.")
    save_to_postgres("label_encoder", df, replace=True)
    save_to_postgres("categorical_cols", pd.DataFrame({"col": categorical_cols}), replace=True)

def data_split(**kwargs):
    df = load_from_postgres("label_encoder")

    # Starting to calculate multicollinearity to save the affected features
    if MULTICOLLINEARITY:
        multicollinearity_analysis(df)

    # Starting to calculate correlation to save the affected features
    if CORRELATION:
        correlation_analysis(df)

    features = [col for col in df.columns if col not in [TARGET, IDENTIFIER]]

    X_train, X_valid, y_train, y_valid = train_test_split(
            df[features], df[TARGET], test_size=0.2, random_state=42)
    save_to_postgres("x_train", X_train, replace=True)
    save_to_postgres("x_valid", X_valid, replace=True)
    save_to_postgres("y_train", y_train, replace=True)
    save_to_postgres("y_valid", y_valid, replace=True)

def get_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_pred),
    }
    return metrics

def get_feature_importance_mlflow(model):
    importance_df = pd.DataFrame({
        "Feature": model.feature_name(),
        "Importance": model.feature_importance(importance_type="gain")  # or "split"
    })

    # Sort by importance and get the top 20
    importances = importance_df.sort_values(by="Importance", ascending=False)

    importances_dict = importances.set_index("Feature")["Importance"].to_dict()
    feature_importance_metrics = {
        f"feature_importance_{feature_name}": imp_value
        for feature_name, imp_value in importances_dict.items()
    }
    return feature_importance_metrics


def train_model():

    X_train = load_from_postgres(f'X_train_preprocessed')
    X_valid = load_from_postgres(f'X_valid_preprocessed')
    y_train = load_from_postgres(f'y_train_preprocessed')
    y_valid = load_from_postgres(f'y_valid_preprocessed')
    categorical_cols = load_from_postgres('categorical_cols')['col'].tolist()

    cat_cols = [col for col in categorical_cols if col in X_train.columns]

    # Get initial list of features
    feature_list = X_train.columns.tolist()
    original_features = feature_list.copy()
    
    iteration = 0

    while len(feature_list) > 10:  # Ensure at least 10 features remain
        print(f"Training with {len(feature_list)} features...")

        X_train_subset = X_train[feature_list]
        X_valid_subset = X_valid[feature_list]

        cat_cols = [col for col in categorical_cols if col in X_train_subset.columns]

        def objective(trial):
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

            train_data = lgb.Dataset(X_train_subset, label=y_train, categorical_feature=cat_cols)
            valid_data = lgb.Dataset(X_valid_subset, label=y_valid, categorical_feature=cat_cols, reference=train_data)

            model = lgb.train(
                param,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(100)],
            )

            preds = model.predict(X_valid_subset)
            return roc_auc_score(y_valid, preds)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, n_jobs=1)

        best_params = study.best_params
        best_params['objective'] = 'binary'
        best_params['metric'] = 'auc'

        model = lgb.train(
            best_params,
            lgb.Dataset(X_train_subset, label=y_train, categorical_feature=cat_cols),
            valid_sets=[lgb.Dataset(X_valid_subset, label=y_valid, categorical_feature=cat_cols)],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
        )

        X_valid_sample = X_valid_subset.copy()
        X_valid_sample = X_valid_sample.astype("float64")

        pred_probs = model.predict(X_valid_subset)
        pred_labels = (pred_probs > 0.5).astype(int) 

        metrics = get_metrics(y_valid, pred_labels)

        signature = infer_signature(X_train_subset, model.predict(X_train_subset))
        feature_importance_metrics = get_feature_importance_mlflow(model)
    
        mlflow.set_tracking_uri("http://mlflow:5000")
        experiment_name = "Customer Churn Prediction"

        mlflow.set_experiment(experiment_name)
        model_name = f"model_all" if original_features == feature_list else f"model_top_{len(feature_list)}"

        with mlflow.start_run():
            print(f"Active Experiment: {mlflow.get_experiment_by_name(experiment_name)}")
            mlflow.lightgbm.log_model(model, f"model_", input_example=X_valid_sample.head(), signature=signature)
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.log_metrics(feature_importance_metrics)

            model_uri = f"runs:/{mlflow.active_run().info.run_id}/lightgbm_{model_name}"
            registered_model = mlflow.register_model(model_uri, f"Preprocessing_Optimizer_{model_name}")
            print(f"Model registered with name: {registered_model.name}, version: {registered_model.version}")


        print(f"Model v{iteration} AUC: {study.best_value}")

        # Remove 10 least important features
        feature_importances = pd.DataFrame({'feature': feature_list, 'importance': model.feature_importance()})
        feature_importances = feature_importances.sort_values(by='importance', ascending=True)

        feature_list = feature_importances.iloc[10:]['feature'].tolist()  # Keep only top features
        iteration += 1

    print("Feature selection completed!")


# DAG Definition
default_args = {"owner": "airflow", "start_date": datetime(2025, 1, 23)}
dag = DAG(
    "feature_reduction_mlflow",
    default_args=default_args,
    max_active_tasks=1,
    schedule_interval=None,
    )
start = PythonOperator(
    task_id="load_data",
    python_callable=load_data,
    dag=dag,)

label_encoding = PythonOperator(
    task_id="label_encoder",
    python_callable=label_encoder,
    dag=dag,)

split = PythonOperator(
    task_id="train_test_split",
    python_callable=data_split,
    dag=dag,)

outliers = PythonOperator(
    task_id="preprocessing",
    python_callable=preprocessing,
    dag=dag,)

train = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,)

start >> label_encoding >> split >> outliers >> train
