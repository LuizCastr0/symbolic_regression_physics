## criando a aplicação com MLFLOW

import numpy as np
import mlflow
from pysr import PySRRegressor

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("symbolic_regression_physics")



data = np.loadtxt("../data/raw/dados_queda_livre.csv", delimiter=",", skiprows=1)
t_ql = data[:, 0].reshape(-1, 1)
y_ql = data[:, 1]

with mlflow.start_run(run_name="queda_livre"):
    params = dict(
        niterations=100,
        maxsize=20,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        sistema="queda_livre",
        equacao_esperada="y = 100 - 4.9 * t^2",
    )
    mlflow.log_params(params)

    model_ql = PySRRegressor(
        niterations=params["niterations"],
        maxsize=params["maxsize"],
        binary_operators=params["binary_operators"],
        unary_operators=params["unary_operators"],
        model_selection="best",
        temp_equation_file=True,
        delete_tempfiles=True,
        output_directory=None,
        variable_names=["t"],
    )
    model_ql.fit(t_ql, y_ql)

    best_ql = model_ql.get_best()
    mlflow.log_metric("loss", float(best_ql.loss))
    mlflow.log_metric("complexity", int(best_ql.complexity))
    mlflow.log_text(best_ql.equation, "equacao_encontrada.txt")
    print(f"[Queda livre] Equação: {best_ql.equation}")

data = np.loadtxt("../data/raw/dados_massa_mola.csv", delimiter=",", skiprows=1)
t_mm = data[:, 0].reshape(-1, 1)
y_mm = data[:, 1]

with mlflow.start_run(run_name="oscilador_harmonico"):
    params = dict(
        niterations=100,
        maxsize=20,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "cos"],
        sistema="oscilador_harmonico",
        equacao_esperada="x = 0.1 * cos(3.16 * t)",
    )
    mlflow.log_params(params)

    model_mm = PySRRegressor(
        niterations=params["niterations"],
        maxsize=params["maxsize"],
        binary_operators=params["binary_operators"],
        unary_operators=params["unary_operators"],
        model_selection="best",
        temp_equation_file=True,
        delete_tempfiles=True,
        output_directory=None,
        variable_names=["t"],
    )
    model_mm.fit(t_mm, y_mm)

    best_mm = model_mm.get_best()
    mlflow.log_metric("loss", float(best_mm.loss))
    mlflow.log_metric("complexity", int(best_mm.complexity))
    mlflow.log_text(best_mm.equation, "equacao_encontrada.txt")
    print(f"[Oscilador harmônico] Equação: {best_mm.equation}")


data = np.loadtxt("../data/raw/dados_pendulo_amortecido.csv", delimiter=",", skiprows=1)
t_pa = data[:, 0].reshape(-1, 1)
y_pa = data[:, 1]

with mlflow.start_run(run_name="pendulo_amortecido"):
    params = dict(
        niterations=100,
        maxsize=30,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "cos", "sin", "exp", "log"],
        sistema="pendulo_amortecido",
        equacao_esperada="theta = 0.52 * exp(-0.1 * t) * cos(3.1 * t)",
    )
    mlflow.log_params(params)

    model_pa = PySRRegressor(
        niterations=params["niterations"],
        maxsize=params["maxsize"],
        binary_operators=params["binary_operators"],
        unary_operators=params["unary_operators"],
        model_selection="best",
        temp_equation_file=True,
        delete_tempfiles=True,
        output_directory=None,
        variable_names=["t"],
    )
    model_pa.fit(t_pa, y_pa)

    best_pa = model_pa.get_best()
    mlflow.log_metric("loss", float(best_pa.loss))
    mlflow.log_metric("complexity", int(best_pa.complexity))
    mlflow.log_text(best_pa.equation, "equacao_encontrada.txt")
    print(f"[Pêndulo amortecido] Equação: {best_pa.equation}")


data = np.loadtxt("../data/raw/dados_resfriamento_newton.csv", delimiter=",", skiprows=1)
t_rn = data[:, 0].reshape(-1, 1)
T_rn = data[:, 1]

with mlflow.start_run(run_name="resfriamento_newton"):
    params = dict(
        niterations=500,
        maxsize=9,
        binary_operators=["+", "-", "*"],
        unary_operators=["exp"],
        sistema="resfriamento_newton",
        equacao_esperada="T = 25 + 75 * exp(-0.1 * t)",
        constraints="exp: 1",
        denoise=True,
        optimizer_iterations=60,
        optimizer_nrestarts=20,
    )
    mlflow.log_params(params)

    model_rn = PySRRegressor(
        niterations=params["niterations"],
        maxsize=params["maxsize"],
        binary_operators=params["binary_operators"],
        unary_operators=params["unary_operators"],
        constraints={"exp": 1},
        nested_constraints={"exp": {"exp": 0}},
        optimizer_iterations=params["optimizer_iterations"],
        optimizer_nrestarts=params["optimizer_nrestarts"],
        denoise=params["denoise"],
        model_selection="best",
        temp_equation_file=True,
        delete_tempfiles=True,
        output_directory=None,
        variable_names=["t"],
    )
    model_rn.fit(t_rn.astype(np.float32), T_rn.astype(np.float32))

    best_rn = model_rn.get_best()
    mlflow.log_metric("loss", float(best_rn.loss))
    mlflow.log_metric("complexity", int(best_rn.complexity))
    mlflow.log_text(best_rn.equation, "equacao_encontrada.txt")

