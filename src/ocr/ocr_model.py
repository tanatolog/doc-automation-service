"""Модуль для OCR модели"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights
from ocr.attention import Attention
from typing import List

class OCRModel(nn.Module):
    """
    OCR модель для распознавания текста на изображениях.

    OCR модель состоит из:
        - предобученные слои ResNet18
        - двунаправленного управляемого рекуррентного модуля (GRU)
        - механизм внимания (Attention)

    Parameters:
        resolution: размер входных изображений (ширина, высота)
        dims: размерность признаков после линейного слоя перед GRU
        num_chars: количество классов символов (без blank)
        use_attention: включить механизм внимания
        grayscale: использовать одноцветные изображения
    """

    def __init__(
        self,
        classes: List[str], # список классов, которые модель будет (предсказывать вместе с blank классом)
        resolution: tuple = (180, 50), # размер входных изображений
        dims: int = 256, # размерность признаков в горле
        use_attention: bool = True,
        grayscale: bool = False,
        dropout_prob: float = 0.5, # добавил
        gru_layers: int = 2 # добавил
    ):
        """
        Конструктор. OCR модель для распознавания текста на изображениях.

        OCR модель состоит из:
        - предобученные слои ResNet18
        - двунаправленного управляемого рекуррентного модуля (GRU)
        - механизм внимания (Attention)

        Parameters:
            classes: список классов, которые модель будет (предсказывать вместе с blank классом)
            resolution: размер входных изображений (ширина, высота)
            dims: размерность признаков после линейного слоя перед GRU
            use_attention: включить механизм внимания
            grayscale: использовать одноцветные изображения
        
        """
        super().__init__()

        self.resolution = resolution # размер входных изображений
        self.grayscale = grayscale
        self.use_attention = use_attention
        self.classes = classes # список классов, которые модель будет (предсказывать вместе с blank классом)
        self.dropout_prob = dropout_prob # добавил

        # Загрузка предобученной ResNet18 с весами ImageNet
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Базовый модуль сети, карты признаков
        self.convnet = resnet18(weights=ResNet18_Weights.DEFAULT) # загрузка resnet обученной на imagenet

        if grayscale: # заменяется conv1 на 1 канальный
            self.convnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Используем слои ResNet до layer2 (для более глубоких признаков)
        # layer0: conv1, bn1, relu, maxpool
        # layer1, layer2
        self.features = nn.Sequential(
            self.convnet.conv1, #  1 -> 64, kernel 7x7, шаг 2, паддинг 3, (batch_size, 64, 25, 90)
            self.convnet.bn1, # нормализация (batch_size, 64, 25, 90)
            self.convnet.relu, # relu (batch_size, 64, 25, 90)
            self.convnet.maxpool, # maxpool 64 -> 64, kernel 3х3, шаг 2, паддинг 1, (batch_size, 64, 13, 45)
            self.convnet.layer1, # первый сверточный слой resnet (batch_size, 64, 13, 45)
            self.convnet.layer2 # второй сверточный слой resnet (batch_size, 128, 7, 23)
        )

        # адаптация выходов ResNet под вход в GRU
        # Размер входа для линейного слоя - считаем динамически
        linear_input_size = self._calc_linear_layer()
        self.linear = nn.Linear(linear_input_size, dims)
        self.bn_linear = nn.BatchNorm1d(dims)
        self.dropout = nn.Dropout(self.dropout_prob)

        # GRU: двунаправленный, 2 слоя
        self.gru = nn.GRU(
            input_size=dims,
            hidden_size=dims // 2,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Attention
        if use_attention:
            self.attention = Attention(dims=dims)
        
        # Финальный классификатор по временной оси
        # Линейный классификатор, который на каждом временном шаге предсказывает логиты для всех символов включая blank
        self.projection = nn.Linear(dims, len(self.classes))


    def _calc_linear_layer(self): # добавил
        width, height = self.resolution
        channels = 1 if self.grayscale else 3
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width) # (batch_size, 1, 50, 180)
            x = self.features(dummy_input) # (batch_size, 64, 13, 45)
            # Перемещение размерностей: (B, C, H, W) -> (batch_size, W, C*H)
            x = x.permute(0, 3, 1, 2).contiguous() # (batch_size, 23, 128, 7)
            x = x.view(x.size(0), x.size(1), -1) # (batch_size, 23, 896)
            return x.shape[-1] # 832
        

    def base_model(self, x):
        """Прямой проход через базовые слои"""
        x = self.features(x) # (batch_size, 128, 7, 23)

        # Преобразование: (B, C, H, W) -> (B, W, C*H)
        x = x.permute(0, 3, 1, 2).contiguous() # (batch_size, 23, 128, 7)
        x = x.view(x.size(0), x.size(1), -1) # (batch_size, 23, 896)

        # Линейный слой + batchnorm + relu + dropout
        x = self.linear(x) # линейный слой, 896 -> 256 (batch_size, 23, 256)

        # Преобразование для BatchNorm1d: (batch_size * 23, C)
        B, T, C = x.shape
        x = x.contiguous().view(B * T, C) # (batch_size * 23, 256)
        x = self.bn_linear(x) # (batch_size * 23, 256)
        x = x.view(B, T, C) # (batch_size, 23, 256)

        x = F.relu(x)
        x = self.dropout(x)

        return x
    

    def forward(self, images, targets=None, target_lengths=None):
        """
        Прямой проход.

        1. Кодирует пакет изображений в вектор признаков размера (batch_size, 45, 256)
        2. Моделирование последовательности в GRU, которое обрабатывает вектор признаков и выдает выходные данные на каждом временном шаге
        3. Механизм внимания, применяемый на временных этапах, подготовленных GRU
        4. Преобразование уровня привлечения внимания (или GRU) в тензор с вероятностным распределением классов

        :param: images: батч изображений формы (batch_size, 1, 50, 180)
        :param: targets: список целевых строк (n_targets) (только при обучении)
        :param: target_lengths: список длинн каждой строки (только при обучении)
        """
        features = self.base_model(images) # (batch_size, 23, 256)
        gru_out, _ = self.gru(features)  # (batch_size, 23, 256)

        if self.use_attention:
            x, _ = self.attention(gru_out, features) # (batch_size, 23, 256)
        else:
            x = gru_out

        logits = self.projection(x) # прогрнозируем (batch_size, 23, num_classes)

        if (targets != None) and (target_lengths != None): # если обучение
            log_probs = logits.permute(1, 0, 2) # (23, batch_size, num_classes) меняем вид тензора

            # фиксированная длина выхода сети (таймстепы)
            input_lengths = torch.full(
                size = (log_probs.size(1),), 
                fill_value = log_probs.size(0),
                dtype = torch.long, 
                device = log_probs.device
            )

            loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths) # функция потерь
            return logits, loss

        return logits, None # если предсказание (batch_size, 23, num_classes) 


    @staticmethod
    def ctc_loss(x, targets, input_lengths, target_lengths):
        """
        Функция потерь Connectionist Temporal Classification (CTC).

        :param: x: тензор предсказаний (batch_size, 45, num_classes) до softmax
        :param: targets: тензор (n_targets) содержащий все истинные последовательности меток классов
        :param: input_lengths: (batch_size) фиксированная длина выхода сети (таймстепы)
        :param: target_lengths: длины истинных последовательностей меток классов
        """
        log_probs = F.log_softmax(x, dim=2) # преобразование логитов в вероятности

        # Создание ctc
        loss_fn = nn.CTCLoss(blank=0, zero_infinity=True) # 0 это метка класса blank

        # Вычисление потерь
        loss = loss_fn(log_probs, targets, input_lengths, target_lengths)

        return loss
    

    def decode_predictions(self, predictions: torch.Tensor) -> list:
        """
        Декодирование предсказаний сети (с функцией ctc).

        1: два (или более) повторяющихся символа объединяются в один экземпляр, если только
        они не разделены blank - это компенсирует тот факт, что RNN выполняет классификацию
        для каждой полосы, представляющей часть цифры (таким образом, создаются дубликаты).

        2: Несколько последовательных blank объединяются в один - это компенсирует
        интервал до, после или между цифрами

        :param predictions (Tensor): тензор предсказаний сети (batch_size, sequence_length, num_classes)

        :return texts (list): список строк, где каждая строка является расшифрованным предсказанием для примера в батче
        """
        predictions = torch.softmax(predictions, dim=2) # преобразование в вероятности
        predictions = torch.argmax(predictions, dim=2).detach().cpu().numpy() # получаем метки наиболее вероятных классов (batch, seq)

        texts = []
        blank_token = self.classes[0] # первый символ в списке классов — это blank (по соглашению)

        for batch_seq in predictions: # цикл по батчам
            prev_char = None
            decoded = []

            for idx in batch_seq:
                char = self.classes[idx]

                if char != blank_token and char != prev_char:
                    decoded.append(char)

                prev_char = char

            texts.append("".join(decoded))

        return texts


if __name__ == "__main__":
    x = torch.randn((1, 3, 50, 180))

    model = OCRModel(
        dims = 256, 
        classes = 35, 
        use_attention = True, 
        grayscale = True,
        dropout_prob = 0.5,
        gru_layers = 2
    )

    output, loss = model(x)
    print(output.shape, loss)
