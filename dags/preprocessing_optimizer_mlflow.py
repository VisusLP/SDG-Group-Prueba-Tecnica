import pandas as pd
import pickle
import numpy as np
import lightgbm as lgb
import mlflow
import optuna

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime
from itertools import product
from mlflow.models import infer_signature
from scipy.stats import zscore
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

TARGET = "churn"
IDENTIFIER = "Customer_ID"
N_OPTUNA_TRIALS = 10

# List of preprocessing steps
steps = {
    # Outlier detection options
    "outlier_detection": ["none", "whole", "train"],
    # Include/Exclude correlation analysis
    "correlation_analysis": [True, False],
    # Include/Exclude multicollinearity analysis
    "multicollinearity_analysis": [True, False],
}
# Generate all combinations
combinations = list(product(
    steps["outlier_detection"],
    steps["correlation_analysis"],
    steps["multicollinearity_analysis"]
))


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


def preprocess_combination(outlier_option, correlation, multicollinearity, **kwargs):
    X_train = load_from_postgres("x_train")
    X_valid = load_from_postgres("x_valid")
    y_train = load_from_postgres("y_train")
    y_valid = load_from_postgres("y_valid")

    print(f"- Correlation Analysis: {correlation}")
    if correlation:
        print("Dropping correlated features...")
        low_corr_features = load_from_postgres(
            "low_corr_features")["col"].tolist()
        drop_cols = [
            col for col in low_corr_features if col in X_train.columns]
        X_train = X_train.drop(columns=drop_cols)
        X_valid = X_valid.drop(columns=drop_cols)

    print(f"- Multicollinearity Analysis: {multicollinearity}")
    if multicollinearity:
        print("Dropping features with high VIF...")
        high_vif = load_from_postgres("high_vif")["col"].tolist()
        drop_cols = [col for col in high_vif if col in X_train.columns]
        X_train = X_train.drop(columns=drop_cols)
        X_valid = X_valid.drop(columns=drop_cols)

    print(f"- Outlier Detection: {outlier_option}")
    features = [col for col in X_train.columns if col not in [
        TARGET, IDENTIFIER, 'outlier']]
    if outlier_option == 'none':
        print("Skipping outlier detection...")
    elif outlier_option == 'whole':
        print("Running outlier detection on the whole dataset...")
        # Merge datasets
        train = pd.concat([X_train, y_train], axis=1)
        valid = pd.concat([X_valid, y_valid], axis=1)
        # Add identifier columns
        train["set"] = "train"
        valid["set"] = "valid"
        # Merge datasets
        df = pd.concat([train, valid])
        # Perform outlier detection (assuming it returns a cleaned dataset)
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
    elif outlier_option == 'train':
        print("Running outlier detection on the training set...")
        train = pd.concat([X_train, y_train], axis=1)
        train = outlier_detection(train)
        X_train = train[features]
        y_train = train[TARGET]
        X_valid = X_valid[features]
    print("Preprocessing completed.")

    save_to_postgres(
        f'x_train_{outlier_option}_{correlation}_{multicollinearity}', X_train, replace=True)
    save_to_postgres(
        f'x_valid_{outlier_option}_{correlation}_{multicollinearity}', X_valid, replace=True)
    save_to_postgres(
        f'y_train_{outlier_option}_{correlation}_{multicollinearity}', y_train, replace=True)
    save_to_postgres(
        f'y_valid_{outlier_option}_{correlation}_{multicollinearity}', y_valid, replace=True)

    return {"outlier_option": outlier_option, "correlation": correlation, "multicollinearity": multicollinearity}


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
    save_to_postgres("low_corr_features", pd.DataFrame(
        {"col": low_corr_features}), replace=True)


def multicollinearity_analysis(df):
    print("Analyzing multicollinearity...")
    X = df.drop(columns=[TARGET]).fillna(0)
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(
        X.values, i) for i in range(X.shape[1])]
    high_vif = vif_data[vif_data['VIF'] > 10]['Feature']
    save_to_postgres("high_vif", pd.DataFrame({"col": high_vif}), replace=True)


def label_encoder():
    df = load_from_postgres("original_dataset")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    save_to_postgres("label_encoder", df, replace=True)
    save_to_postgres("categorical_cols", pd.DataFrame(
        {"col": categorical_cols}), replace=True)


def additional_analysis():
    df = load_from_postgres("label_encoder")
    # Starting to calculate multicollinearity to save the affected features
    multicollinearity_analysis(df)

    # Starting to calculate correlation to save the affected features
    correlation_analysis(df)

    # Starting to calculate outliers to save the affected rows
    features = [col for col in df.columns if col not in [TARGET, IDENTIFIER]]

    X_train, X_valid, y_train, y_valid = train_test_split(
        df[features], df[TARGET], test_size=0.2, random_state=42)
    # Save the preprocessed data to XCom
    save_to_postgres("x_train", X_train, replace=True)
    save_to_postgres("x_valid", X_valid, replace=True)
    save_to_postgres("y_train", y_train, replace=True)
    save_to_postgres("y_valid", y_valid, replace=True)


def train_model(**kwargs):

    preprocessing_data = kwargs['ti'].xcom_pull(
        task_ids=kwargs['preprocessing_task_id'])

    outlier_option = preprocessing_data['outlier_option']
    correlation = preprocessing_data['correlation']
    multicollinearity = preprocessing_data['multicollinearity']

    print(
        f"Training with: Outlier={outlier_option}, Correlation={correlation}, Multicollinearity={multicollinearity}")

    X_train = load_from_postgres(
        f'x_train_{outlier_option}_{correlation}_{multicollinearity}')
    X_valid = load_from_postgres(
        f'x_valid_{outlier_option}_{correlation}_{multicollinearity}')
    y_train = load_from_postgres(
        f'y_train_{outlier_option}_{correlation}_{multicollinearity}')
    y_valid = load_from_postgres(
        f'y_valid_{outlier_option}_{correlation}_{multicollinearity}')
    categorical_cols = load_from_postgres('categorical_cols')['col'].tolist()

    cat_cols = [col for col in categorical_cols if col in X_train.columns]

    print(f"Training set:\n{X_train.shape}\n")
    print(f"Validation set:\n{X_valid.shape}\n")
    print(f"Categorical columns: {cat_cols}")
    unique_features = {col: X_train[col].unique() for col in cat_cols}
    print(f"Unique values for categorical columns:\n{unique_features}")

    def objective(trial):
        # Define the parameter search space
        param = {
            'objective': 'binary',  # For binary classification
            'metric': 'auc',        # Optimize for AUC
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
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ],
    )
    save_model_to_postgres(
        final_model, f"model_{outlier_option}_{correlation}_{multicollinearity}")
    save_to_postgres(f"best_params_{outlier_option}_{correlation}_{multicollinearity}", pd.DataFrame(
        study.best_params, index=[0]), replace=True)


def evaluate_model(**kwargs):
    preprocessing_data = kwargs['ti'].xcom_pull(
        task_ids=kwargs['preprocessing_task_id'])
    outlier_option = preprocessing_data['outlier_option']
    correlation = preprocessing_data['correlation']
    multicollinearity = preprocessing_data['multicollinearity']
    print(
        f"Evaluating with: Outlier={outlier_option}, Correlation={correlation}, Multicollinearity={multicollinearity}")

    X_valid = load_from_postgres(
        f'x_valid_{outlier_option}_{correlation}_{multicollinearity}')
    y_valid = load_from_postgres(
        f'y_valid_{outlier_option}_{correlation}_{multicollinearity}')

    final_model = load_model_from_postgres(
        f"model_{outlier_option}_{correlation}_{multicollinearity}")
    pred_probs = final_model.predict(X_valid)
    pred_labels = (pred_probs > 0.5).astype(int)

    auc = roc_auc_score(y_valid, pred_probs)
    accuracy = accuracy_score(y_valid, pred_labels)
    f1 = f1_score(y_valid, pred_labels)
    precision = precision_score(y_valid, pred_labels)
    recall = recall_score(y_valid, pred_labels)
    logloss = log_loss(y_valid, pred_probs)

    metrics = {
        "auc": auc,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "log_loss": logloss
    }

    print("Evaluation metrics:", metrics)
    save_to_postgres(f"metrics_{outlier_option}_{correlation}_{multicollinearity}", pd.DataFrame(
        metrics, index=[0]), replace=True)


def get_feature_importance_mlflow(model):
    importance_df = pd.DataFrame({
        "Feature": model.feature_name(),
        "Importance": model.feature_importance(importance_type="gain")
    })

    # Sort by importance and get the top 20
    importances = importance_df.sort_values(by="Importance", ascending=False)

    # Convert to dictionary for easy XCom push
    importances_dict = importances.set_index("Feature")["Importance"].to_dict()
    feature_importance_metrics = {
        f"feature_importance_{feature_name}": imp_value
        for feature_name, imp_value in importances_dict.items()
    }
    return feature_importance_metrics


def register_model(**kwargs):
    preprocessing_data = kwargs['ti'].xcom_pull(
        task_ids=kwargs['preprocessing_task_id'])
    outlier_option = preprocessing_data['outlier_option']
    correlation = preprocessing_data['correlation']
    multicollinearity = preprocessing_data['multicollinearity']
    print(
        f"Registering model with: Outlier={outlier_option}, Correlation={correlation}, Multicollinearity={multicollinearity}")

    model = load_model_from_postgres(
        f"model_{outlier_option}_{correlation}_{multicollinearity}")
    metrics = load_from_postgres(
        f"metrics_{outlier_option}_{correlation}_{multicollinearity}")
    X_train = load_from_postgres(
        f'x_train_{outlier_option}_{correlation}_{multicollinearity}')
    best_params = load_from_postgres(
        f"best_params_{outlier_option}_{correlation}_{multicollinearity}")
    best_params = best_params.to_dict(orient='records')[0]
    X_valid = load_from_postgres(
        f'x_valid_{outlier_option}_{correlation}_{multicollinearity}')
    X_valid_sample = X_valid.copy()
    X_valid_sample = X_valid_sample.astype(
        "float64")  # Convert integers to float

    signature = infer_signature(X_train, model.predict(X_train))

    feature_importance_metrics = get_feature_importance_mlflow(model)

    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment_name = "Preprocessing Optimization"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        print(
            f"Active Experiment: {mlflow.get_experiment_by_name(experiment_name)}")
        mlflow.lightgbm.log_model(model, f"model_{outlier_option}_{correlation}_{multicollinearity}",
                                  input_example=X_valid_sample.head(), signature=signature)
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(feature_importance_metrics)

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/lightgbm_model_{outlier_option}_{correlation}_{multicollinearity}"
        registered_model = mlflow.register_model(
            model_uri, f"Preprocessing_Optimizer_Model_{outlier_option}_{correlation}_{multicollinearity}")
        print(
            f"Model registered with name: {registered_model.name}, version: {registered_model.version}")


# DAG Definition
default_args = {"owner": "airflow", "start_date": datetime(2025, 2, 2)}
dag = DAG(
    "preprocessing_optimizer_mlflow",
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

analysis = PythonOperator(
    task_id="additional_analysis",
    python_callable=additional_analysis,
    dag=dag,)
branching = BranchPythonOperator(
    task_id="branching",
    python_callable=lambda: [
        f"preprocess_{i}" for i in range(len(combinations))],
    dag=dag,
)
# Preprocessing combinations
preprocessing_tasks = []
training_tasks = []
evaluating_tasks = []
registering_tasks = []
for i, combination in enumerate(combinations):
    task = PythonOperator(
        task_id=f"preprocess_{i}",
        python_callable=preprocess_combination,
        op_kwargs={
            "outlier_option": combination[0],
            "correlation": combination[1],
            "multicollinearity": combination[2],
        },
        dag=dag,
    )
    preprocessing_tasks.append(task)
    train_task = PythonOperator(
        task_id=f"train_{i}",
        python_callable=train_model,
        op_kwargs={
            "preprocessing_task_id": f"preprocess_{i}",
        },
        provide_context=True,
        dag=dag,
    )
    training_tasks.append(train_task)
    eval_task = PythonOperator(
        task_id=f"evaluate_{i}",
        python_callable=evaluate_model,
        op_kwargs={
            "preprocessing_task_id": f"preprocess_{i}",
        },
        provide_context=True,
        dag=dag,
    )
    evaluating_tasks.append(eval_task)
    register_task = PythonOperator(
        task_id=f"register_{i}",
        python_callable=register_model,
        op_kwargs={
            "preprocessing_task_id": f"preprocess_{i}",
        },
        provide_context=True,
        dag=dag,
    )
    registering_tasks.append(register_task)

start >> label_encoding >> analysis >> branching
for i in range(len(preprocessing_tasks)):
    branching >> preprocessing_tasks[i] 
    preprocessing_tasks[i] >> training_tasks[i]
    training_tasks[i] >> evaluating_tasks[i]
    evaluating_tasks[i] >> registering_tasks[i]
