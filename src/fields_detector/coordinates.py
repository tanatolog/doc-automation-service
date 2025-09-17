"""В модуле содержаться функции для работы с координатами разных видов"""
import torch

def xy_to_cxcy(xy):
    """
    Преобразование рамок в координатах границ (x_min, y_min, x_max, y_max) в координаты центра (c_x, c_y, w, h).
    Работает как с дробными так и с абсолютными координатами.
    
    c_x = (x_max + x_min) / 2
    c_y = (y_max + y_min) / 2
    w = x_max - x_min
    h = y_max - y_min

    :param xy: тензор рамок в координатах границ (n_boxes, 4)
    :return: тензор рамок в координатах центра (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2, # c_x, c_y
                       xy[:, 2:] - xy[:, :2]], 1) # w, h


def cxcy_to_xy(cxcy):
    """
    Преобразование рамок в координатах центра (c_x, c_y, w, h) в координаты границ (x_min, y_min, x_max, y_max).
    Работает как с дробными так и с абсолютными координатами.
    
    x_min = c_x - w / 2
    y_min = c_y - h / 2
    x_max = c_x + w / 2
    y_max = c_y + h / 2

    :param cxcy: тензор рамок в координатах центра (n_boxes, 4)
    :return: тензор рамок в координатах границ (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2), # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1) # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Получение смещений относительно приоров из рамок в дробных центральных координатах.
    Используется в обучении. Модель учится корректировать приоры путём предсказания смещений.

    :param cxcy: тензор рамок в дробных центральных координатах (n_priors, 4)
    :param priors_cxcy: тензор приоров в дробных центральных координатах (n_priors, 4)
    :return: тензор смещений относительно приоров (n_priors, 4)
    """
    # 10 и 5 - это варианты, они найдены эмпирически и помогают стабилизировать градиенты
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10), # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1) # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Получение координат рамок в дробных координатах центра из смещений для приоров предсказанных моделью.

    :param gcxgcy: смещения относительно приоров, предсказанные моделью (n_priors, 4)
    :param priors_cxcy: приоры в дробных координатах центра (n_priors, 4)
    :return: рамки в дробных координатах центра (n_priors, 4)
    """
    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2], # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1) # w, h