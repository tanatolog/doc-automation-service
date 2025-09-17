"""Модуль для обучения модели детекции объектов"""
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import os

from fields_detector.model import ObjectDetector, MultiLoss
from fields_detector.bb_dataset import BbDataset
from utils.constants import LABEL_MAP, DEVICE
from utils.log_file import LogFile

# Параметры датасета
DATASET_PATH = "data/fields_dataset/" # путь к папке с датесетом
USE_DIFFICULT = True # использовать трудные объекты

# Параметры модели
CLASSES_AMOUNT = len(LABEL_MAP) # количество классов объектов
NET_FOLDER_PATH = "models/"
MODEL_NAME = "fields_det_test"

# Параметры обучения
PRETRAINED_MODEL_PATH = None # "models/fields_det_11_112.pth" # путь к предобученной модели, чтобы дообучить
BATCH_SIZE = 8 # размер батча
EPOCHS_AMOUNT = 500 # колличество эпох обучения
WORKERS_AMOUNT = 4 # колличество потоков, кот будут загружать данные из датасета, исп. в DataLoader
LOG_FREQ = 10 # статус обучения выводиться каждые __ батчей
LEARNING_RATE = 0.0001 # скорость обучения
DECAY_LR = [20, 40, 60, 80, 100, 120, 140] # снижение скорости обучения происходит после этих эпох
DECAY_LR_KOEFF = 0.5 # снижение скорости обучения до этой доли от текущей
MOMENTUM = 0.9 # момент
WEIGHT_DECAY = 5e-4 # коэффициент регуляризации L2 при обучении модели, добавляет штраф за большие веса в функцию потерь
GRAD_CLIP = None # обрезка градиентов, включить при взрыве градиентов. Он может произойти при больших размерах батча (больше 32) - об этом свидетельсвует ошибка сортировки в MuliLoss 
cudnn.benchmark = True # настройка оптимизации свёрточных операций PyTorch, работает только на GPU и при одинаковых размерах входных данных
FORSED_LR_DECAY_KOEF = 1 # при дообучении модели можно отрегулировать скорость обучения


def adjust_learning_rate(optimizer, scale: float, log_file: LogFile = None):
    """
    Масштабирование скорости обучения на коэффициент.

    :param optimizer: оптимизатор, скорость обучения которого должна быть снижена;
    :param scale: коэффициент, на который умножается скорость обучения.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * scale

    print("Скорость обучения изменена. LR = %f\n" % (optimizer.param_groups[1]["lr"],))
    if log_file != None:
        log_file.open()
        log_file.write("Скорость обучения изменена. LR = %f\n" % (optimizer.param_groups[1]["lr"],))
        log_file.close()


class MetricTracker(object):
    """Класс для отслеживания метрик. Отслеживает последнее значение, сумму, количество элементов, среднее значение"""

    def __init__(self):
        """Конструктор. Класс для отслеживания метрик. Отслеживает последнее значение, сумму, количество элементов, среднее значение."""
        self.reset()

    def reset(self):
        """Обнуление всех значений"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Обновление значений"""
        self.val = val # последнее значение метрики
        self.sum += val * n # сумма метрик
        self.count += n # колличество
        self.avg = self.sum / self.count # среднее значение

def clip_gradient(optimizer, grad_clip: float):
    """
    Обрезание градиентов, вычисленных при обратном распространении, чтобы избежать резкого увеличения градиентов

    :param optimizer: оптимизатор с градиентами
    :param grad_clip: максимальное допустимое значение градиента
    """
    for group in optimizer.param_groups:
        for param in group["params"]:

            if param.grad != None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def train(train_loader, model: ObjectDetector, criterion: MultiLoss, optimizer, epoch: int, log_file: LogFile):
    """
    Одна эпоха обучения.

    :param train_loader: DataLoader для тренировачных данных
    :param model: нейросетевая модель
    :param criterion: Multiloss
    :param optimizer: оптимизатор
    :param epoch: номер текущей эпохи
    """
    model.train() # перевод модели в режим обучения, активация dropout

    batch_time = MetricTracker() # время выполнения одной итерации обучения, прямой + обратный проход
    data_time = MetricTracker() # время загрузки данных
    losses = MetricTracker() # потери

    start = time.time()

    for i, (images, boxes, labels, _) in enumerate(train_loader): # цикл по батчам
        data_time.update(time.time() - start)

        # загрузка на девайс
        images = images.to(DEVICE) # (batch_size, 3, 300, 300)
        boxes = [b.to(DEVICE) for b in boxes]
        labels = [l.to(DEVICE) for l in labels]

        # прямой проход
        predicted_locs, predicted_scores = model(images) # (N, 8732, 4), (N, 8732, n_classes)

        # потери
        loss = criterion(predicted_locs, predicted_scores, boxes, labels) # скаляр

        # обратный проход
        optimizer.zero_grad() # обнуление градиентов
        loss.backward() # вычисление градиентов

        if GRAD_CLIP != None: # обрезка градиентов
            clip_gradient(optimizer, GRAD_CLIP)

        # обновление весов
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time() # сброс времени

        # логирование
        if i % LOG_FREQ == 0:
            print("Epoch: [{0}][{1}/{2}]\t"
                  "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  "Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(epoch, i, len(train_loader),
                                                                batch_time = batch_time,
                                                                data_time = data_time, 
                                                                loss = losses
                                                                )
            )

            log_file.open()
            log_file.write("Epoch: [{0}][{1}/{2}]\t"
                  "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  "Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(epoch, i, len(train_loader),
                                                                batch_time = batch_time,
                                                                data_time = data_time, 
                                                                loss = losses
                                                                )
            )
            log_file.close()
            
    del predicted_locs, predicted_scores, images, boxes, labels # удаление переменных, чтобы PyTorch освободил память GPU


def save_model(epoch: int, model: ObjectDetector, optimizer, filename: str):
    """
    Сохранение модели

    :param epoch: номер эпохи
    :param model: модель
    :param optimizer: оптимизатор
    :param filename: кка сохранить модель
    """
    state = {"epoch": epoch,
             "model": model,
             "optimizer": optimizer}
    torch.save(state, filename)
    print(f"\n{epoch}: модель {os.path.basename(filename)} сохранена\n")


def main():
    """Обучение"""
    start_epoch: int = 0

    log = LogFile(os.path.join(NET_FOLDER_PATH, MODEL_NAME + ".txt"))
    log.open()
    log.write("Начало")
    log.write(f"BATCH_SIZE: {BATCH_SIZE}, LR: {LEARNING_RATE}, CLIP {GRAD_CLIP}")
    log.close()

    # Инициализация модели или загрузка предобученной модели
    if PRETRAINED_MODEL_PATH is None: # инициализация
        start_epoch = 0
        model = ObjectDetector(n_classes=CLASSES_AMOUNT)
        biases = list()
        not_biases = list()

        # фильтр параметров
        for param_name, param in model.named_parameters(): # цикл по всем параметрам

            if param.requires_grad: # если обучаемый параметр

                if param_name.endswith(".bias"): # если биас
                    biases.append(param)
                else:
                    not_biases.append(param)

        # задаем отдельный learning rate для биасов
        optimizer = torch.optim.SGD(

            params = [
                {"params": biases, "lr": 2 * LEARNING_RATE}, 
                {"params": not_biases}
            ],

            lr = LEARNING_RATE, 
            momentum = MOMENTUM, 
            weight_decay = WEIGHT_DECAY
        )
        
    else: # загрузка предобученной модели
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, weights_only=False)
        start_epoch = checkpoint["epoch"] + 1
        print("\nПредобученная модель загружена\n")
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]
        adjust_learning_rate(optimizer, FORSED_LR_DECAY_KOEF, log)

    # загрузка на девайс
    model = model.to(DEVICE)
    criterion = MultiLoss(priors_cxcy=model.priors_cxcy).to(DEVICE)

    # загрузчик данных
    train_dataset = BbDataset(
        DATASET_PATH,
        split = "train",
        keep_difficult = USE_DIFFICULT
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = BATCH_SIZE, 
        shuffle = True,
        collate_fn = train_dataset.collate_fn, # передача кастомной функции
        num_workers = WORKERS_AMOUNT,
        pin_memory = True
    )

    # Преобразование итераций в эпохи
    epochs = EPOCHS_AMOUNT
    decay_lr = DECAY_LR

    for epoch in range(start_epoch, epochs): # цикл по эпохам

        if epoch in decay_lr: # снижение скорости обучения при данных эпохах
            adjust_learning_rate(optimizer, DECAY_LR_KOEFF, log)

        # одна эпоха обучения
        train(
            train_loader = train_loader,
            model = model,
            criterion = criterion,
            optimizer = optimizer,
            epoch = epoch,
            log_file = log
        )
        save_model(epoch, model, optimizer, os.path.join(NET_FOLDER_PATH, MODEL_NAME + "_" + str(epoch) + ".pth")) # сохранение модели


if __name__ == "__main__":

    try:
        torch.cuda.empty_cache()
        main()
    except Exception:
        print("Ищи ошибку")