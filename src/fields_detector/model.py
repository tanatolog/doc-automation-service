"""Модуль для описания нейросетевой модели детектора объектов"""
from torch import nn
import torch.nn.functional as func
from math import sqrt
from itertools import product as product
import torchvision
import torch

from fields_detector.jaccard import find_jaccard_indx
from fields_detector.coordinates import xy_to_cxcy, cxcy_to_xy, gcxgcy_to_cxcy, cxcy_to_gcxgcy
from utils.constants import DEVICE

def decimate(tensor, n: list):
    """Уменьшение тензора путем децимации (сохранение каждого n значения). Используется для преобразования в слой меньшего размера."""

    assert tensor.dim() == len(n)

    for dim in range(tensor.dim()): # цикл по измерениям
        if n[dim] != None: # рефракторинг is not на !=
            tensor = tensor.index_select(dim=dim, index=torch.arange(start=0, end=tensor.size(dim), step=n[dim]).long()) # выбор эл по заданным индексам

    return tensor

class BaseLayers(nn.Module):
    """Класс базовой части модели для создания feature maps низкого уровня"""

    def __init__(self):
        """Конструктор. Класс базовой части модели для создания feature maps низкого уровня"""
        super(BaseLayers, self).__init__()

        # стандартные сверточные слои VGG
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # 64 фильтра 3х3х3, каждый проходит по всему изображению 3х300х300 с шагом 1, не уменьшает карту т.к по краям добавляется 1 пиксель (padding=1), выход (N, 64, 300, 300)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # (N, 64, 300, 300)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # проход с фильтром 2х2 по кажд каналу берем макс, получаем (N, 64, 150, 150)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # нечетная размерность
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) # ceiling (не floor) делаем нечетную размерность

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # сохраняет размер, потому что шаг (stride) равен 1 и отступы (padding) 1. Удаляет мелкие шумы, усиляет активные признаки

        # новые сверточные слои
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6) # padding=6 - пропуск 5 пикселей между ядрами, увеличивает поле восприятия без потери разрешения, позволяет учитывать широкий контекст

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1) # эквивалентен полносвязному слою, нужен чтобы загрузить в него предобученные веса

        self.load_pretrained_weights()

    def forward(self, image):
        """
        Прямой проход

        :param image: тензор изображений (N, 3, 300, 300)
        :return: низкоуровневые карты признаков conv4_3, conv7
        """
        out = func.relu(self.conv1_1(image)) # (N, 64, 300, 300)
        out = func.relu(self.conv1_2(out)) # (N, 64, 300, 300)
        out = self.pool1(out) # (N, 64, 150, 150)

        out = func.relu(self.conv2_1(out)) # (N, 128, 150, 150)
        out = func.relu(self.conv2_2(out)) # (N, 128, 150, 150)
        out = self.pool2(out) # (N, 128, 75, 75)

        out = func.relu(self.conv3_1(out)) # (N, 256, 75, 75)
        out = func.relu(self.conv3_2(out)) # (N, 256, 75, 75)
        out = func.relu(self.conv3_3(out)) # (N, 256, 75, 75)
        out = self.pool3(out) # (N, 256, 38, 38), без ceil_mode=True было бы 37

        out = func.relu(self.conv4_1(out)) # (N, 512, 38, 38)
        out = func.relu(self.conv4_2(out)) # (N, 512, 38, 38)
        out = func.relu(self.conv4_3(out)) # (N, 512, 38, 38)
        conv4_3_feats = out # сохранение карты признаков (N, 512, 38, 38)
        out = self.pool4(out) # (N, 512, 19, 19)

        out = func.relu(self.conv5_1(out)) # (N, 512, 19, 19)
        out = func.relu(self.conv5_2(out)) # (N, 512, 19, 19)
        out = func.relu(self.conv5_3(out)) # (N, 512, 19, 19)
        out = self.pool5(out) # сохраняем размерность (N, 512, 19, 19)

        out = func.relu(self.conv6(out)) # (N, 1024, 19, 19)

        conv7_feats = func.relu(self.conv7(out)) # (N, 1024, 19, 19)

        return conv4_3_feats, conv7_feats # низкоуровневые карты признаков

    def load_pretrained_weights(self):
        """
        Загрузка весов VGG16 предварительно обученных на ImageNet.
        Слои 1-5 точно копируются, слои 6-7 загружаются после подвыборки и децимации.
        """
        weights_dict = self.state_dict() # текущее состояние модели
        param_names = list(weights_dict.keys())

        pretrained_weights_dict = torchvision.models.vgg16(pretrained=True).state_dict() # предобученные веса
        pretrained_param_names = list(pretrained_weights_dict.keys())

        # копирование предобученных весов в слои 1-5 напрямую
        for i, param in enumerate(param_names[:-4]):
            weights_dict[param] = pretrained_weights_dict[pretrained_param_names[i]]

        # копирование в слой 6 путем децимации
        conv_fc6_weight = pretrained_weights_dict["classifier.0.weight"].view(4096, 512, 7, 7) # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_weights_dict["classifier.0.bias"] # (4096)
        weights_dict["conv6.weight"] = decimate(conv_fc6_weight, n=[4, None, 3, 3]) # (1024, 512, 3, 3)
        weights_dict["conv6.bias"] = decimate(conv_fc6_bias, n=[4]) # (1024)
        
        # копирование в слой 7 путем децимации
        conv_fc7_weight = pretrained_weights_dict["classifier.3.weight"].view(4096, 4096, 1, 1) # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_weights_dict["classifier.3.bias"] # (4096)
        weights_dict["conv7.weight"] = decimate(conv_fc7_weight, n=[4, 4, None, None]) # (1024, 1024, 1, 1)
        weights_dict["conv7.bias"] = decimate(conv_fc7_bias, n=[4]) # (1024)

        self.load_state_dict(weights_dict) # загрузка весов
        print("\nЗагружены Base layers\n")


class AdditionalLayers(nn.Module):
    """Класс дополнительных сверточных слоев для создания feature maps высокого уровня. Должны распологатся над base model."""

    def __init__(self):
        """Конструктор. Класс дополнительных сверточных слоев для создания feature maps высокого уровня"""
        super(AdditionalLayers, self).__init__()

        # дополнительные слои
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # уменьшение измерений т.к. шаг stride > 1

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # уменьшение измерений т.к. шаг stride > 1

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0) # уменьшение измерений т.к. заполнение padding = 0

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0) # уменьшение измерений т.к. заполнение padding = 0

        self.init_conv2d() 

    def init_conv2d(self):
        """Инициализация параметров слоёв"""

        for layer in self.children(): # цикл по прямым дочерним слоям
            if isinstance(layer, nn.Conv2d): # проверка на Conv2d слой
                nn.init.xavier_uniform_(layer.weight) # инициализация весов равномерным распределением Ксавье, подходит для симметричной активации (ReLU, Tanh). Поддерживает стабильную дисперсию градиентов.
                nn.init.constant_(layer.bias, 0.) # инициализация смещений 0

    def forward(self, conv7_feats):
        """
        Прямой проход.

        :param conv7_feats: feature map тензор (N, 1024, 19, 19)
        :return: feature maps высокого уровня
        """
        out = func.relu(self.conv8_1(conv7_feats)) # (N, 256, 19, 19)
        out = func.relu(self.conv8_2(out)) # (N, 512, 10, 10)
        conv8_2_feats = out # (N, 512, 10, 10)

        out = func.relu(self.conv9_1(out)) # (N, 128, 10, 10)
        out = func.relu(self.conv9_2(out)) # (N, 256, 5, 5)
        conv9_2_feats = out # (N, 256, 5, 5)

        out = func.relu(self.conv10_1(out)) # (N, 128, 5, 5)
        out = func.relu(self.conv10_2(out)) # (N, 256, 3, 3)
        conv10_2_feats = out # (N, 256, 3, 3)

        out = func.relu(self.conv11_1(out)) # (N, 128, 3, 3)
        conv11_2_feats = func.relu(self.conv11_2(out)) # (N, 256, 1, 1)

        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats # feature maps высокого уровня


class PredictionLayers(nn.Module):
    """Класс предсказательной части модели"""

    def __init__(self, n_classes: int):
        """
        Конструктор. Класс предсказательной части модели.

        :param n_classes: число классов объектов
        """
        super(PredictionLayers, self).__init__()

        self.n_classes = n_classes

        n_boxes = {"conv4_3": 4, # количество приоров для каждой позиции на карте
                   "conv7": 6,
                   "conv8_2": 6,
                   "conv9_2": 6,
                   "conv10_2": 4,
                   "conv11_2": 4}

        # предсказание смещений рамок объектов
        self.loc_box_conv4_3 = nn.Conv2d(512, n_boxes["conv4_3"] * 4, kernel_size=3, padding=1)
        self.loc_box_conv7 = nn.Conv2d(1024, n_boxes["conv7"] * 4, kernel_size=3, padding=1)
        self.loc_box_conv8_2 = nn.Conv2d(512, n_boxes["conv8_2"] * 4, kernel_size=3, padding=1)
        self.loc_box_conv9_2 = nn.Conv2d(256, n_boxes["conv9_2"] * 4, kernel_size=3, padding=1)
        self.loc_box_conv10_2 = nn.Conv2d(256, n_boxes["conv10_2"] * 4, kernel_size=3, padding=1)
        self.loc_box_conv11_2 = nn.Conv2d(256, n_boxes["conv11_2"] * 4, kernel_size=3, padding=1)

        # предсказание классов в рамках
        self.class_conv4_3 = nn.Conv2d(512, n_boxes["conv4_3"] * n_classes, kernel_size=3, padding=1)
        self.class_conv7 = nn.Conv2d(1024, n_boxes["conv7"] * n_classes, kernel_size=3, padding=1)
        self.class_conv8_2 = nn.Conv2d(512, n_boxes["conv8_2"] * n_classes, kernel_size=3, padding=1)
        self.class_conv9_2 = nn.Conv2d(256, n_boxes["conv9_2"] * n_classes, kernel_size=3, padding=1)
        self.class_conv10_2 = nn.Conv2d(256, n_boxes["conv10_2"] * n_classes, kernel_size=3, padding=1)
        self.class_conv11_2 = nn.Conv2d(256, n_boxes["conv11_2"] * n_classes, kernel_size=3, padding=1)

        self.init_conv2d()

    def init_conv2d(self):
        """Инициализация параметров"""

        for c in self.children(): # цикл по прямым дочерним слоям
            if isinstance(c, nn.Conv2d): # проверка на Conv2d слой
                nn.init.xavier_uniform_(c.weight) # инициализация весов равномерным распределением Ксавье, подходит для симметричной активации (ReLU, Tanh). Поддерживает стабильную дисперсию градиентов.
                nn.init.constant_(c.bias, 0.) # инициализация смещений 0

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        """
        Прямой проход

        :param conv4_3_feats: тензор карты признаков (N, 512, 38, 38)
        :param conv7_feats: тензор карты признаков (N, 1024, 19, 19)
        :param conv8_2_feats: тензор карты признаков (N, 512, 10, 10)
        :param conv9_2_feats: тензор карты признаков (N, 256, 5, 5)
        :param conv10_2_feats: тензор карты признаков (N, 256, 3, 3)
        :param conv11_2_feats: тензор карты признаков (N, 256, 1, 1)
        :return: 8732 варианта предсказания смещений рамок и классов в них
        """
        batch_size = conv4_3_feats.size(0)

        # предсказание смещений для приоров
        pred_box_conv4_3 = self.loc_box_conv4_3(conv4_3_feats) # предсказания (N, 16, 38, 38)
        pred_box_conv4_3 = pred_box_conv4_3.permute(0, 2, 3, 1).contiguous() # изменение формы тензора (N, 38, 38, 16)
        pred_box_conv4_3 = pred_box_conv4_3.view(batch_size, -1, 4) # выпрямляем тензор, получение 5776 предсказаний рамок по 4 координаты каждая (N, 5776, 4)

        pred_box_conv7 = self.loc_box_conv7(conv7_feats) # (N, 24, 19, 19)
        pred_box_conv7 = pred_box_conv7.permute(0, 2, 3, 1).contiguous() # (N, 19, 19, 24)
        pred_box_conv7 = pred_box_conv7.view(batch_size, -1, 4) # (N, 2166, 4)

        pred_box_conv8_2 = self.loc_box_conv8_2(conv8_2_feats)# (N, 24, 10, 10)
        pred_box_conv8_2 = pred_box_conv8_2.permute(0, 2, 3, 1).contiguous()# (N, 10, 10, 24)
        pred_box_conv8_2 = pred_box_conv8_2.view(batch_size, -1, 4)# (N, 600, 4)

        pred_box_conv9_2 = self.loc_box_conv9_2(conv9_2_feats) # (N, 24, 5, 5)
        pred_box_conv9_2 = pred_box_conv9_2.permute(0, 2, 3, 1).contiguous() # (N, 5, 5, 24)
        pred_box_conv9_2 = pred_box_conv9_2.view(batch_size, -1, 4) # (N, 150, 4)

        pred_box_conv10_2 = self.loc_box_conv10_2(conv10_2_feats) # (N, 16, 3, 3)
        pred_box_conv10_2 = pred_box_conv10_2.permute(0, 2, 3, 1).contiguous() # (N, 3, 3, 16)
        pred_box_conv10_2 = pred_box_conv10_2.view(batch_size, -1, 4) # (N, 36, 4)

        pred_box_conv11_2 = self.loc_box_conv11_2(conv11_2_feats) # (N, 16, 1, 1)
        pred_box_conv11_2 = pred_box_conv11_2.permute(0, 2, 3, 1).contiguous() # (N, 1, 1, 16)
        pred_box_conv11_2 = pred_box_conv11_2.view(batch_size, -1, 4) # (N, 4, 4)

        # предсказание классов в рамках
        pred_class_conv4_3 = self.class_conv4_3(conv4_3_feats) # (N, 4 * n_classes, 38, 38)
        pred_class_conv4_3 = pred_class_conv4_3.permute(0, 2, 3, 1).contiguous() #  изменение формы тензора (N, 38, 38, 4 * n_classes)
        pred_class_conv4_3 = pred_class_conv4_3.view(batch_size, -1, self.n_classes) # выпрямляем тензор, получение 5776 предсказаний классов (N, 5776, n_classes)

        pred_class_conv7 = self.class_conv7(conv7_feats) # (N, 6 * n_classes, 19, 19)
        pred_class_conv7 = pred_class_conv7.permute(0, 2, 3, 1).contiguous() # (N, 19, 19, 6 * n_classes)
        pred_class_conv7 = pred_class_conv7.view(batch_size, -1, self.n_classes) # (N, 2166, n_classes)

        pred_class_conv8_2 = self.class_conv8_2(conv8_2_feats) # (N, 6 * n_classes, 10, 10)
        pred_class_conv8_2 = pred_class_conv8_2.permute(0, 2, 3, 1).contiguous() # (N, 10, 10, 6 * n_classes)
        pred_class_conv8_2 = pred_class_conv8_2.view(batch_size, -1, self.n_classes) # (N, 600, n_classes)

        pred_class_conv9_2 = self.class_conv9_2(conv9_2_feats) # (N, 6 * n_classes, 5, 5)
        pred_class_conv9_2 = pred_class_conv9_2.permute(0, 2, 3, 1).contiguous() # (N, 5, 5, 6 * n_classes)
        pred_class_conv9_2 = pred_class_conv9_2.view(batch_size, -1, self.n_classes) # (N, 150, n_classes)

        pred_class_conv10_2 = self.class_conv10_2(conv10_2_feats) # (N, 4 * n_classes, 3, 3)
        pred_class_conv10_2 = pred_class_conv10_2.permute(0, 2, 3, 1).contiguous() # (N, 3, 3, 4 * n_classes)
        pred_class_conv10_2 = pred_class_conv10_2.view(batch_size, -1, self.n_classes) # (N, 36, n_classes)

        pred_class_conv11_2 = self.class_conv11_2(conv11_2_feats) # (N, 4 * n_classes, 1, 1)
        pred_class_conv11_2 = pred_class_conv11_2.permute(0, 2, 3, 1).contiguous() # (N, 1, 1, 4 * n_classes)
        pred_class_conv11_2 = pred_class_conv11_2.view(batch_size, -1, self.n_classes) # (N, 4, n_classes)

        # сбор 8732 предсказаний в порядке их следования
        locs = torch.cat([pred_box_conv4_3, pred_box_conv7, pred_box_conv8_2, pred_box_conv9_2, pred_box_conv10_2, pred_box_conv11_2], dim=1) # (N, 8732, 4)
        classes_scores = torch.cat([pred_class_conv4_3, pred_class_conv7, pred_class_conv8_2, pred_class_conv9_2, pred_class_conv10_2, pred_class_conv11_2], dim=1) # (N, 8732, n_classes)

        return locs, classes_scores


class ObjectDetector(nn.Module):
    """Класс нейросети - детектора объектов"""

    def __init__(self, n_classes):
        """Конструктор. Класс нейросети - детектора объектов"""
        super(ObjectDetector, self).__init__()
        self.n_classes = n_classes # число классов

        # слои сети
        self.base_layers = BaseLayers()
        self.additional_layers = AdditionalLayers()
        self.pred_layers = PredictionLayers(n_classes)
        # у признаков с нижних уровней (ранние слои сверточной сети) (conv4_3) значения активаций могут быть очень разными по масштабу, что затрудняет обучение.
        self.normalize_layer = nn.Parameter(torch.FloatTensor(1, 512, 1, 1)) # поэтому создаем обучаемый слой нормализации
        nn.init.constant_(self.normalize_layer, 20) # значения инициализируются константой 20

        self.priors_cxcy = self.create_prior_boxes() # координаты приоров в дробных центральных координатах

    def forward(self, image):
        """
        Прямой проход

        :param image: тензор изображения (N, 3, 300, 300)
        :return: 8732 предсказаний рамок и их классов
        """
        # получение низкоуровневых карт
        conv4_3_feats, conv7_feats = self.base_layers(image) # (N, 512, 38, 38), (N, 1024, 19, 19)

        # нормализация conv4_3
        L2_norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt() # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / L2_norm # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.normalize_layer # (N, 512, 38, 38)

        # получение уровневых карт
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.additional_layers(conv7_feats) # (N, 512, 10, 10), (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # слои предсказания
        locs, classes_scores = self.pred_layers(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        return locs, classes_scores

    def create_prior_boxes(self):
        """Создание приоров для карт признаков. Приоры возвращаются в дробных центральных координатах."""

        fmap_dims = {"conv4_3": 38, # карты признаков и их размеры
                     "conv7": 19,
                     "conv8_2": 10,
                     "conv9_2": 5,
                     "conv10_2": 3,
                     "conv11_2": 1}

        obj_scales = {"conv4_3": 0.1, # площадь покрытия приором относительно карты
                      "conv7": 0.2,
                      "conv8_2": 0.375,
                      "conv9_2": 0.55,
                      "conv10_2": 0.725,
                      "conv11_2": 0.9}

        aspect_ratios = {"conv4_3": [1., 2., 0.5], # соотношения сторон приоров
                         "conv7": [1., 2., 3., 0.5, .333],
                         "conv8_2": [1., 2., 3., 0.5, .333],
                         "conv9_2": [1., 2., 3., 0.5, .333],
                         "conv10_2": [1., 2., 0.5],
                         "conv11_2": [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys()) # список имен карт
        prior_boxes = []

        for k, fmap in enumerate(fmaps): # цикл по картам
            for i in range(fmap_dims[fmap]): # цикл по измерениям карты
                for j in range(fmap_dims[fmap]): # цикл по измерениям карты
                    # координаты центра приора
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]: # цикл по приорам
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        if ratio == 1.: # для соотношения сторон 1 добавим еще приор с площадью покрытия равной среднему геометрическому масштаба текущей карты объектов и масштаба следующей карты объектов
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            except IndexError: # для последней карты следующей карты нет
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale]) # доп приор

        prior_boxes = torch.FloatTensor(prior_boxes).to(DEVICE) # преобразование в тензор, загрузка на девайс (8732, 4)
        #prior_boxes.clamp_(0, 1) # (8732, 4); this line has no effect; see Remarks section in tutorial

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Расшифровка предсказаний сети. 8732 предсказания смещений рамок и 8732 предсказания классов.
        Так же производится декодирование координат, отбор по порогу, подавление пересечений и сортировка по уверенности.

        :param predicted_locs: предсказания смещений для каждого приора (N, 8732, 4)
        :param predicted_scores: предсказания классов (N, 8732, n_classes)
        :param min_score: мин порог принадлежности к классу
        :param max_overlap: максимальное перекрытие, которое могут иметь две рамки, чтобы та, которая имеет более низкую оценку, не была заблокирована
        :param top_k: если во всех классах обнаруживается много результатов, оставьте только k лучших
        :return: списки длинны batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = func.softmax(predicted_scores, dim=2) # приведение логитов к вероятностям, сумма = 1 (N, 8732, n_classes)

        # списки для хранения окончательных прогнозируемых рамок, названий объектов и оценок для всех изображений
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for img in range(batch_size): # цикл по изображениям
            # декодирование координат рамок из приоров и смещений для каждого приора в рамки в дробных координатах границ 
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[img], self.priors_cxcy)) # (8732, 4)

            # списки для хранения рамок, названий и оценок на одном изображении
            image_boxes = list()
            image_labels = list()
            image_scores = list()
            #max_scores, best_label = predicted_scores[i].max(dim=1) # (8732)

            for c in range(1, self.n_classes): # цикл по классам, кроме фона 0

                # оставляем рамки и классы, оценкци которых превышают мин порог min_score
                class_scores = predicted_scores[img][:, c] # оценки объектов в приорах для одного класса, одного изображения (8732)
                score_above_min_score = class_scores > min_score # (byte)
                n_above_min_score = score_above_min_score.sum().item()

                if n_above_min_score == 0:
                    continue

                class_scores = class_scores[score_above_min_score] # отбрасываем лишние оценки (n_above_min_score), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score] # отбрасываем лишние рамки (n_above_min_score, 4)

                # сортировка оценок
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True) # сортировка оценок по убыванию (n_above_min_score), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind] # пререстановка рамок в соответсвие оценкам (n_min_score, 4)

                overlap = find_jaccard_indx(class_decoded_locs, class_decoded_locs) # нахождение сетки перекрытия между каждой парой рамок (n_above_min_score, n_min_score)
                # это нужно чтобы отбросить рамки, которые выделяют один и тот же объект на изображении

                # Non-Maximum Suppression (NMS) максимальное подавление
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(DEVICE) # маска подавления, 1 подавить, 0 не подавлять (n_above_min_score)

                for box in range(class_decoded_locs.size(0)): # цикл по каждой рамке, чтобы пройти по каждой строчке сетки перекрытия

                    if suppress[box] == 1: # если рамка уже помечена для удаления
                        continue

                    # сравнение перекрытий для этой рамки со всеми другими рамками, нахождение перекрытий больших макс перекрытия
                    suppress = torch.max(suppress, overlap[box] > max_overlap) # пометка соотв рамок для удаления, max сохраняет ранее отмеченные поля, как операция или
                    suppress[box] = 0 # убираем отметку этой рамкой, т.к она перекрывается сама с собой с коэф = 1

                # сохранение неподавленных рамок для этого класса
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(DEVICE))
                image_scores.append(class_scores[1 - suppress])

            if len(image_boxes) == 0: # если объект ни в одном из классов не найден, сохранение как фон
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(DEVICE))
                image_labels.append(torch.LongTensor([0]).to(DEVICE))
                image_scores.append(torch.FloatTensor([0.]).to(DEVICE))

            # объединяем списки тензоров в отдельные тензоры
            image_boxes = torch.cat(image_boxes, dim=0) # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0) # (n_objects)
            image_scores = torch.cat(image_scores, dim=0) # (n_objects)
            n_objects = image_scores.size(0)

            if n_objects > top_k: # оставляем только k объектов с наивысшей оценкой вероятности
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k] # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k] # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k] # (top_k)

            # сохраняем обработанные объекты к спискам всех изображений
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores # списки длинны batch_size


class MultiLoss(nn.Module):
    """
    Класс реализующий комбинированную функцию потерь

    Это комбинация:
    (1) потерь (ошибки предсказания) координат рамок;
    (2) потерь (ошибки предсказания) класса объекта.
    """

    def __init__(self, priors_cxcy, threshold: float = 0.5, neg_pos_ratio: int = 3, alpha: float = 1.):
        """Конструктор. Класс реализующий комбинированную функцию потерь"""
        super(MultiLoss, self).__init__()
        self.priors_cxcy = priors_cxcy # приоры в дробных центральных координатах
        self.priors_xy = cxcy_to_xy(priors_cxcy) # приоры в дробных граничных координатах
        self.threshold = threshold # порог, если коэф Жаккара больше, то рамка содержит объект
        self.neg_pos_ratio = neg_pos_ratio # на кажд. положительный пример берем не более стольки отрицательных
        # Hard Negative Mining выбираем только самые трудные отрицательные примеры, у которых наибольшая ошибка предсказания
        self.alpha = alpha # вес ошибки рамок

        self.smooth_l1 = nn.L1Loss() # ф потерь для ошибки координат рамок
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False) # ф потерь для ошибки классов

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Комбинированная функция потерь.

        :param predicted_locs: тензор предсказаний рамок для кажд приора, полученный из модели (N, 8732, 4)
        :param predicted_scores: тензор предсказаний классов для кажд приора, полученный из модели (N, 8732, n_classes)
        :param boxes: истинные рамки объектов в дробных координатах границ, список из N тензоров
        :param labels: истинные названия объектов, список из N тензоров
        :return: комбинированные потери, скаляр
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        # сбор истинных значений
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(DEVICE) # пустой тензор для истинных смещений (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(DEVICE) # пустой тензор для истинных названий объектов (N, 8732)
        for i in range(batch_size): # цикл по изображениям в батче
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_indx(boxes[i], self.priors_xy) # коэф жаккара между всеми объектами и приорами (n_objects, 8732)

            # для кажд приора найти макс коэф и объект, кот соотносится с этим коэф
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0) # каждый приор выбирает лучший объект (8732)

            # нужно чтобы кажд объект получил хотя бы 1 приор. Но может возникнуть ситуация, когда ни один приор не назначится объекту:
            # 1. когда объект не является лучшим по всем параметрам
            # 2. когда все приоры объекта становятся фоном, т.к не удовлетворяют порогу threshold
            # чтобы обработать это нужно

            _, prior_for_each_object = overlap.max(dim=1) # индекс приора, наиболее соответствующего каждому объекту, каждый объект выбирает лучший приор (N)

            # лучшие приоры принудительно получают свои объекты
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(DEVICE) # назначаем каждому объекту приор с макс коэф (исправляем 1 случай)
            overlap_for_each_prior[prior_for_each_object] = 1. # для того чтобы приор соотвествовал порогу искусственно увеличим их коэф (исправляем 2 случай)

            label_for_each_prior = labels[i][object_for_each_prior] # название для каждого приора (8732)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0 # если коэф меньше порога, то назначаем прору класс фон (8732)

            true_classes[i] = label_for_each_prior
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy) # кодируем координаты рамок в виде смещений для приоров (8732, 4)

        positive_priors = true_classes != 0 # тензор для истинных названий объектов без приоров фона (N, 8732)

        # Потери координат рамок

        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors]) # ф потерь вычисляется только для не фоновых приоров, скаляр
        # если predicted_locs имеет форму (N, 8732, 4), то predicted_locs[positive_priors] будет иметь форму (total positives, 4)

        # Потери классификации

        # потери вычисляются из положительных приоров и наиболее сложных отрицательных приоров на кажд изображении
        # используем neg_pos_ratio * n_positives негативных приоров, на кот потери макс
        # это называется Hard Negative Mining - концентрируемся на самых сложных негативах в каждом изображении и минимизируем дисбаланс классов

        n_positives = positive_priors.sum(dim=1) # количество позитивных приоров на кажд изображении (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives # количество сложных негативных приоров на кажд изображении (N)

        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1)) # нахождение потерь для приоров (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors) # изменение вида тензора (N, 8732)

        conf_loss_pos = conf_loss_all[positive_priors] # потери на приорах, где действительно были объекты (sum(n_positives))

        # потери сложных негативных приоров
        conf_loss_neg = conf_loss_all.clone() # (N, 8732)
        conf_loss_neg[positive_priors] = 0. # обнуление позитивных приоров (N, 8732)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True) # сортируем сложные негативные приоры по убыванию потерь (N, 8732)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(DEVICE) # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1) # маска (N, 8732), где True — приор попал в top-k hardest
        conf_loss_hard_neg = conf_loss_neg[hard_negatives] # извлечение потерь только для сложных негативных приоров (sum(n_hard_negatives))

        # суммируем потери по позитивным и сложным негативным приорам и нормализуем по общему числу позитивов
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float() # () скаляр

        # Общие потери
        return conf_loss + self.alpha * loc_loss
