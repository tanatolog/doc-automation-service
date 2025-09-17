"""Модуль для реализации коэффициента Жаккара"""
import torch

def find_intersection(set_1, set_2):
    """
    Нахождение площади пересечения. Для каждой рамки из set_1 относительно каждой рамки из set_2.

    :param set_1: тензор рамок (n1, 4)
    :param set_2: тензор рамок (n2, 4)
    :return: площадь пересечения каждой рамки из set_1 относительно каждой рамки из set_2 тензор (n1, n2)
    """
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0)) # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0)) # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0) # (n1, n2, 2)

    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1] # (n1, n2)


def find_jaccard_indx(set_1, set_2):
    """
    Нахождение коэффициента Жаккара. Для каждой рамки из set_1 относительно каждой рамки из set_2.

    :param set_1: тензор рамок (n1, 4)
    :param set_2: тензор рамок (n2, 4)
    :return: коэффициент Жаккара каждой рамки из set_1 относительно каждой рамки из set_2 тензор (n1, n2)
    """
    intersection = find_intersection(set_1, set_2) # нахождение площади пересечения (n1, n2)

    # нахождение площади каждой рамки
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1]) # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1]) # (n2)

    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection # нахождение площади объединения (n1, n2)

    return intersection / union # коефф Жаккара (n1, n2)