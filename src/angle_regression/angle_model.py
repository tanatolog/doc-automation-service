"""Модуль для модели регрессии угла поворота"""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as Func

class AngleRegressionModel(nn.Module):
    """
    Нейросетевая модель для регрессии угла поворота.

    Архитектура:
        - Сверточные слои с предобученными весами.
        - Полносвязные слои для регрессии синуса и косинуса угла.

    :return: Тензор размера (batch_size, 2), соответствующий [sin(angle), cos(angle)]
    """

    def __init__(self, dropout_prob: float):
        """
        Конструктор. Нейросетевая модель для регрессии угла поворота.
            
        Архитектура:
            - Сверточные слои с предобученными весами.
            - Полносвязные слои для регрессии синуса и косинуса угла.

        :param dropout_prob: вероятность отключения нейронов
        :return: Тензор размера (batch_size, 2), соответствующий [sin(angle), cos(angle)]
        """
        super().__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT) # загрузка предобученной ResNet18 с весами ImageNet

        # Базовая модель
        # Используем слои ResNet
        # layer0: conv1, bn1, relu, maxpool
        # layer1, layer2, layer3
        self.base_model = nn.Sequential(
            self.backbone.conv1, #  3 -> 64, kernel 7x7, шаг 2, паддинг 3, (batch_size, 64, 112, 112)
            self.backbone.bn1, # нормализация (batch_size, 64, 112, 112)
            self.backbone.relu, # relu (batch_size, 64, 112, 112)
            self.backbone.maxpool, # maxpool 64 -> 64, kernel 3х3, шаг 2, паддинг 1, (batch_size, 64, 56, 56)
            self.backbone.layer1, # первый сверточный слой resnet (batch_size, 64, 56, 56)
            self.backbone.layer2, # второй сверточный слой resnet (batch_size, 128, 28, 28)
            self.backbone.layer3, # третий сверточный слой resnet (batch_size, 256, 14, 14)
            self.backbone.layer4 # четвертый сверточный слой resnet (batch_size, 512, 7, 7)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (batch_size, 512, 7, 7) -> (batch_size, 512, 1, 1)
        self.flatten = nn.Flatten()               # (batch_size, 512, 1, 1) -> (batch_size, 512)

        # Полносвязная голова
        self.regressor_predictor = nn.Sequential(
            nn.Linear(512, 512), # (batch_size, 512)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, 2)  # выход: sin(angle), cos(angle)
        )

    def forward(self, x):
        """
        Прямой проход.

        :param x: батч изображений (batch_size, 3, 300, 300)
        :return: тензор (batch_size, 2)
        """
        features = self.base_model(x) # (batch_size, 256, 7, 7)
        x = self.pool(features) # (batch_size, 256, 1, 1)
        x = self.flatten(x) # (batch_size, 256)
        output = self.regressor_predictor(x) # (batch_size, 2)
        return output
    

class AngleLoss(nn.Module):
    """
    Функция потерь для регрессии угла поворота.
    Основана на косинусном сходстве между предсказанным и истинным вектором [sin, cos].

    Loss = 1 - cos(pred, target)
    """

    def __init__(self, eps: float = 1e-7):
        """
        Конструктор. Функция потерь для регрессии угла поворота.
        Основана на косинусном сходстве между предсказанным и истинным вектором [sin, cos].

        Loss = 1 - cos(pred, target)
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход.

        :param pred: предсказания модели, тензор формы (batch_size, 2)
        :param target: истинные значения угла, тензор формы (batch_size, 2)
        :return: скалярная функция потерь
        """
        # L2 нормализация (по длине), нормализация длинны, чтобы сравнение происходило по направлению вектора (углу), а не по длине
        pred_norm = Func.normalize(pred, dim=1, eps=self.eps)
        target_norm = Func.normalize(target, dim=1, eps=self.eps)

        cosine_sim = (pred_norm * target_norm).sum(dim=1) # скалярное произведение векторов, дает cos
        loss = 1.0 - cosine_sim # 1 - cos(θ)
        return loss.mean()
    

if __name__ == "__main__":
    model = AngleRegressionModel(0.4)
    dummy = torch.randn(4, 3, 224, 224) # батч из 4 изображений
    out = model(dummy)
    print(out.shape)