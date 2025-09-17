"""Тестирование coordinates"""
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from fields_detector.coordinates import xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy


def test_xy_to_cxcy():
    """Проверка преобразования из координат границ в центральные координаты"""
    xy = torch.tensor([[0.0, 0.0, 2.0, 4.0]]) # один прямоугольник
    cxcy = xy_to_cxcy(xy)
    xy_back = cxcy_to_xy(cxcy)

    assert torch.allclose(xy_back, xy, atol=1e-6)


def test_cxcy_to_xy():
    """Проверка преобразования из cxcy в xy"""
    cxcy = torch.tensor([[1.0, 2.0, 2.0, 4.0]])
    xy = cxcy_to_xy(cxcy)
    cxcy_back = xy_to_cxcy(xy)

    assert torch.allclose(cxcy_back, cxcy, atol=1e-6)


def test_cxcy_to_gcxgcy():
    """Проверка, что смещения относительно приоров равны нулю, если cxcy == priors"""
    priors = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    cxcy = priors.clone()
    gcxgcy = cxcy_to_gcxgcy(cxcy, priors)

    assert torch.allclose(gcxgcy, torch.zeros_like(gcxgcy), atol=1e-6)

    cxcy_back = gcxgcy_to_cxcy(gcxgcy, priors)
    assert torch.allclose(cxcy_back, cxcy, atol=1e-6)


def test_cxcy_to_gcxgcy_nontrivial():
    """Проверка обратимости"""
    priors = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    cxcy = torch.tensor([[0.6, 0.4, 0.4, 0.1]])
    gcxgcy = cxcy_to_gcxgcy(cxcy, priors)

    # проверка обратимости
    cxcy_back = gcxgcy_to_cxcy(gcxgcy, priors)
    assert torch.allclose(cxcy_back, cxcy, atol=1e-6)


def test_batch_support():
    """Проверка поддержки батчей"""
    xy = torch.tensor([
        [0.0, 0.0, 2.0, 2.0],
        [1.0, 1.0, 3.0, 5.0],
    ])
    cxcy = xy_to_cxcy(xy)
    xy_back = cxcy_to_xy(cxcy)

    assert xy_back.shape == xy.shape
    assert torch.allclose(xy_back, xy, atol=1e-6)