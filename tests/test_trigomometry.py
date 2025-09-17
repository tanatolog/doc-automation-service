"""Тестирование trigonometry"""
import math
import torch
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from angle_regression.trigonometry import angle_to_sincos, sincos_to_angle, decode_angles


def test_angle_to_sincos(): 
    """Проверка обратимости преобразования угол для разных значений"""
    angles = [-180, -90, 0, 45, 90, 180]
    for angle in angles:
        sin_val, cos_val = angle_to_sincos(angle)
        angle_back = sincos_to_angle(sin_val, cos_val)
        assert math.isclose(angle, angle_back, abs_tol=1e-6)


def test_decode_angles():
    """Проверка decode_angles для одиночного предсказания"""
    angle = 30
    sin_val, cos_val = angle_to_sincos(angle)
    tensor = torch.tensor([sin_val, cos_val], dtype=torch.float32)
    decoded = decode_angles(tensor)
    assert math.isclose(decoded, angle, abs_tol=1e-6)


def test_decode_angles_batch():
    """Проверка decode_angles для батча предсказаний"""
    angles = [0, 45, 90, -90, 180]
    data = []
    for a in angles:
        sin_val, cos_val = angle_to_sincos(a)
        data.append([sin_val, cos_val])
    tensor = torch.tensor(data, dtype=torch.float32)
    
    decoded = decode_angles(tensor)
    assert len(decoded) == len(angles)
    for d, a in zip(decoded, angles):
        assert math.isclose(d, a, abs_tol=1e-6)


def test_decode_angles_wrong():
    """Проверка что decode_angles выдает ValueError"""
    tensor = torch.tensor([1.0], dtype=torch.float32)
    with pytest.raises(ValueError):
        decode_angles(tensor)

    tensor = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    with pytest.raises(ValueError):
        decode_angles(tensor)