# save_model_if_better.py
import os
import pickle

from homework.src._internals.compare_models import compare_models
from homework.src._internals.save_model import save_model


def save_model_if_better(model, x_test, y_test, save_path="models/estimator.pkl"):
    """Save the model if it performs better than the existing one."""
    best_model = None
    if os.path.exists(save_path):
        try:
            with open(save_path, "rb") as file:
                best_model = pickle.load(file)
        except (EOFError, pickle.UnpicklingError):
            print("⚠️ El archivo del modelo existente está vacío o dañado. Se ignorará.")
            best_model = None

    best_model = compare_models(model, best_model, x_test, y_test)
    save_model(best_model, save_path)
