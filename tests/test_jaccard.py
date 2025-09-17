"""Тестирование jaccard"""
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from fields_detector.jaccard import find_intersection, find_jaccard_indx


def test_intersection_full_overlap():
    """Проверка для совпадающих рамок"""
    box1 = torch.tensor([[0, 0, 2, 2]], dtype=torch.float32)
    box2 = torch.tensor([[0, 0, 2, 2]], dtype=torch.float32)

    inter = find_intersection(box1, box2)
    assert inter.shape == (1, 1)
    assert torch.allclose(inter, torch.tensor([[4.0]]))  # площадь 2*2=4


def test_intersection_no_overlap():
    """Проверка непересекающихся рамок"""
    box1 = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
    box2 = torch.tensor([[2, 2, 3, 3]], dtype=torch.float32)

    inter = find_intersection(box1, box2)
    assert torch.all(inter == 0)


def test_jaccard_full_overlap():
    """Проверка коэф. Жаккара для совпадающих рамок"""
    box1 = torch.tensor([[0, 0, 2, 2]], dtype=torch.float32)
    box2 = torch.tensor([[0, 0, 2, 2]], dtype=torch.float32)

    jacc = find_jaccard_indx(box1, box2)
    assert jacc.shape == (1, 1)
    assert torch.allclose(jacc, torch.tensor([[1.0]]))  # полное совпадение


def test_jaccard_partial_overlap():
    """Проверка коэф. Жаккара для частично пересекающихся рамок"""
    # Пересекающиеся рамки
    box1 = torch.tensor([[0, 0, 2, 2]], dtype=torch.float32)
    box2 = torch.tensor([[1, 1, 3, 3]], dtype=torch.float32)

    jacc = find_jaccard_indx(box1, box2)
    assert jacc.shape == (1, 1)

    # пересечение = 1 (квадрат 1x1), площадь объединения = 4+4-1=7
    assert torch.allclose(jacc, torch.tensor([[1/7]]))


def test_jaccard_no_overlap():
    """Проверка коэф. Жаккара Жаккара для непересекающихся рамок"""
    box1 = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
    box2 = torch.tensor([[2, 2, 3, 3]], dtype=torch.float32)

    jacc = find_jaccard_indx(box1, box2)
    assert torch.all(jacc == 0)