import os
from random import random, randint
import mlflow
from mlflow import log_artifacts, log_param, log_metric
import mlflow.sklearn
import numpy as np
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

# Укажите учетные данные AWS через переменные окружения
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ["AWS_REGION"] = "eu-central-1"

# Установить URI и эксперимент MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment('mlflow_test')

if __name__ == '__main__':
    # Создание каталога для артефактов
    if not os.path.exists('outputs'):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("Hello world")

    # Начало нового запуска
    mlflow.end_run()
    with mlflow.start_run():

        # Логирование артефактов
        log_artifacts("outputs")

        # Обучение модели линейной регрессии
        X_train = np.random.rand(100, 1)
        for i, value in enumerate(X_train):
            log_param(f"X_train_{i}", value[0])
        y_train = 2 * X_train + 1 + np.random.randn(100, 1) * 0.1
        model = LinearRegression()
        model.fit(X_train, y_train) # Метод наименьших квадратов

        # Прогнозирование
        y_pred = model.predict(X_train)

        # Вычисление метрик
        mse = mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)

        # Логирование метрик
        log_metric("mse", mse)
        log_metric("r2", r2)

        log_artifacts("outputs")  # Загружаем в MLflow как артефакт

        # Логирование модели
        input_example = np.array([[0.5]])
        signature = infer_signature(input_example, model.predict(input_example))
        mlflow.sklearn.log_model(model, "linear_regression_model", signature=signature, input_example=input_example)

        # Регистрация модели в Model Registry
        client = MlflowClient()
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/linear_regression_model"
        model_name = "linear_regression_model"

        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            print(f"Model {model_name} already exists in registry.")

        model_version = client.create_model_version(model_name, model_uri, "Generated during training.")
        print(f"Model version {model_version.version} registered successfully.")
