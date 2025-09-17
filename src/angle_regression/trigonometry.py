"""Модуль для функций работы с углами синусами и косинусами"""
import math
import numpy as np
import torch

def angle_to_sincos(angle: float):
    """
    Перевод угла в градусах в sin-cos представление

    :param angle: угол в градусах
    :return: sin, cos
    """
    rad = math.radians(angle)
    return math.sin(rad), math.cos(rad)


def sincos_to_angle(sin_val: float, cos_val: float):
    """
    Перевод из синус-косинус представления обратно в угол в градусах.

    :param sin_val: синус угла
    :param cos_val: косинус угла
    :return: угол в градусах от -180 до 180
    """
    rad = math.atan2(sin_val, cos_val)
    deg = math.degrees(rad)
    return deg


def decode_angles(data: torch.Tensor):
    """
    Преобразует тензор углов (2) в форме sin-cos в углы в градусах в диапазоне [-180, 180].

    :param preds: тензор углов (2) или (N, 2)
    :return: список углов в градусах list[float] или угол в градусах float (если одно значение) 
    """
    data = data.detach().cpu().numpy()  # безопасно извлекаем значения
    data = np.asarray(data)

    if data.ndim == 1 and data.shape[0] == 2: # одиночное предсказание
        return sincos_to_angle(data[0], data[1])
    elif data.ndim == 2 and data.shape[1] == 2: # массив предсказаний
        return [sincos_to_angle(sin, cos) for sin, cos in data]
    else:
        raise ValueError(f"Ожидался тензор формы (2,) или (N, 2), но получено: {data.shape}")