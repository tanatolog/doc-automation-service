"""Модуль для функций рисования графиков обучения"""
import matplotlib.pyplot as plt
import numpy as np

def plot_losses(train_losses, valid_losses, save_path: str):
    """Рисование графика потерь при обучении"""
    plt.style.use("classic")
    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_losses, color="blue", label="Training loss")
    ax.plot(valid_losses, color="red", label="Validation loss")
    ax.set(title="Зависимость потерь от эпох", xlabel="Эпохи", ylabel="Потери")
    ax.legend()
    plt.style.use("default")
    plt.savefig(save_path)


def plot_acc(accuracy, save_path: str):
    """Рисование графика точности при тестировании"""
    plt.style.use("classic")
    accuracy = np.array(accuracy)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(accuracy, color="purple", label="Model Accuracy")
    ax.set(title="Зависимость точности от эпох", xlabel="Эпохи", ylabel="Точность")
    ax.legend()
    plt.style.use("default")
    plt.savefig(save_path)
