"""Модуль для обучения и тестирования модели регрессии угла поворота"""
import copy
import torch
from sklearn import model_selection
from rich.console import Console
from rich import print
import os
import torch.backends.cudnn as cudnn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from rich.progress import track

from utils.constants import DEVICE, ANGLE_IMAGE_SIZE, ANGLE_DROPOUT
from utils.log_file import LogFile
from utils.logging_config import general_table, predictions_table
from utils.plot import plot_acc, plot_losses
from angle_regression.angle_dataset import AngleDataset
from angle_regression.angle_model import AngleRegressionModel, AngleLoss
from angle_regression.trigonometry import decode_angles

cudnn.benchmark = True # настройка оптимизации свёрточных операций PyTorch, работает только на GPU и при одинаковых размерах входных данных
NUM_WORKERS = 4
ACCURACY = 5.0 # погрешность расчета точности

# Пути
DATASET_PATH = "data/angle_dataset/images/"
SAVE_MODEL_AS = "models/angle_det_test.pth"
SAVE_LOG_AS = "models/angle_det_test.txt"

# Конфигурация обучения
MAX_ANGLE = 180.0
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS_AMOUNT = 100
CLIPPING = 0 # обрезка градиентов, если 0, то без обрезки. Оставить, если наблюдаются пики значений loss или NaN


def build_dataloaders(dataset_path: str, max_angle: float, image_size: tuple, batch_size: int, num_workers: int):
    """Создание обучающей и тестировачной выборки. Создание классов датасетов и загрузчиков dataloader"""

    image_paths = sorted([ # получение всех путей к изображениям
        os.path.join(dataset_path, fname)
        for fname in os.listdir(dataset_path)
        if fname.lower().endswith(".jpg")
    ])

    # Разделение датасета на обучающую и тестовую выборки 95\5
    (train_imgs_paths, test_imgs_paths) = model_selection.train_test_split(image_paths, test_size=0.05, random_state=42)

    # Обучающий датасет
    train_dataset = AngleDataset(
        image_paths = train_imgs_paths,
        max_angle = max_angle,
        image_size = image_size
    )

    # Обучающий загрузчик
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = True,
        drop_last=True, # отбросить последний батч, если он неполный
    )

    # Тестовый датасет
    test_dataset = AngleDataset(
        image_paths = test_imgs_paths,
        max_angle = max_angle,
        image_size = image_size
    )

    # Тестовый загрузчик
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = False
    )

    return train_loader, test_loader


def train_fn(model: AngleRegressionModel, loss_func: AngleLoss, data_loader: DataLoader, optimizer: Optimizer, device: torch.device, clipping: int) -> float:
    """
    Выполнение одной эпохи обучения модели.

    :param model: Обучаемая модель
    :param loss_func: Класс функции потерь
    :param data_loader: DataLoader, предоставляющий батчи обучающих данных
    :param optimizer: Оптимизатор
    :param device: Устройство для вычислений (CPU или CUDA)
    :param clipping: Параметр обрезки градиентов

    :return fin_loss: Среднее значение функции потерь за эпоху
    """
    model.train()
    fin_loss = 0

    for data in track(data_loader, description="😪 Training..."):
        images, targets = data
        images = images.to(device)
        targets = targets.to(device)

        # Алгоритм ОРО
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_func(predictions, targets)
        loss.backward()

        if clipping != 0: # обрезка градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)

        optimizer.step()
        fin_loss += loss.item()

    return fin_loss / len(data_loader)


def eval_fn(model: AngleRegressionModel, loss_func: AngleLoss, data_loader: DataLoader, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor], float]:
    """
    Оценка модели.

    :param model: Обучаемая модель
    :param loss_func: Класс функции потерь
    :param data_loader: DataLoader, предоставляющий батчи обучающих данных
    :param device: Устройство для вычислений (CPU или CUDA)

    :return eval_preds: Список предсказаний
    :return eval_targets: Список целей
    :return eval_loss: Среднее значение функции потерь
    """
    model.eval()
    eval_loss = 0
    eval_preds = []
    eval_targets = []
    #print("\n!eval!\n")

    with torch.no_grad():

        for data in track(data_loader, description="🤔 Testing ..."):
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)
            loss = loss_func(predictions, targets)

            eval_loss += loss.item()
            eval_preds.append(predictions)
            eval_targets.append(targets)

    return eval_preds, eval_targets, eval_loss / len(data_loader)


def angular_diff(a: float, b: float):
    """Вычисляет минимальную циклическую разность между двумя углами в градусах (от 0 до 180°) для вычисления точности модели"""
    return abs(((a - b + 180) % 360) - 180)


def run_training(console: Console):

    log = LogFile(SAVE_LOG_AS)
    log.open()
    print(f"Конфикурация:\nimage_size = {ANGLE_IMAGE_SIZE}\nmax_angle = {MAX_ANGLE}\nbatch_size = {BATCH_SIZE}\ndropout = {ANGLE_DROPOUT}\nlearning_rate = {LEARNING_RATE}\nclipping = {CLIPPING}\n")
    log.write(f"Конфикурация:\nimage_size = {ANGLE_IMAGE_SIZE}\nmax_angle = {MAX_ANGLE}\nbatch_size = {BATCH_SIZE}\ndropout = {ANGLE_DROPOUT}\nlearning_rate = {LEARNING_RATE}\nclipping = {CLIPPING}\n")
    log.close()

    # Dataset, dataloaders
    train_loader, test_loader = build_dataloaders(
        dataset_path = DATASET_PATH,
        max_angle = MAX_ANGLE,
        image_size = ANGLE_IMAGE_SIZE,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS
    )

    # Model, optimizer, scheduler, loss
    model = AngleRegressionModel(dropout_prob = ANGLE_DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=4, verbose=True)
    loss_func = AngleLoss()

    best_acc = 0.0
    train_loss_data = []
    valid_loss_data = []
    accuracy_data = []

    # Обучение
    for epoch in range(EPOCHS_AMOUNT):

        train_loss = train_fn(
            model = model,
            loss_func = loss_func, 
            data_loader = train_loader, 
            optimizer = optimizer, 
            device = DEVICE,
            clipping = CLIPPING
        )

        train_loss_data.append(train_loss)

        valid_preds, valid_targets, test_loss = eval_fn(
            model = model,
            loss_func = loss_func, 
            data_loader = test_loader, 
            device = DEVICE
        )

        valid_loss_data.append(test_loss)

        valid_decoded_preds = []
        valid_decoded_targets = []
        for i in range(len(valid_targets)):
            valid_decoded_preds.extend(decode_angles(valid_preds[i]))
            valid_decoded_targets.extend(decode_angles(valid_targets[i]))

        combined = list(zip(valid_decoded_targets, valid_decoded_preds))

        table = predictions_table()
        for idx in combined: # вывод таблицы результатов валидации
            table.add_row(f"{idx[0]:.2f}", f"{idx[1]:.2f}")

        console.print(table)

        # Подсчет точности
        diffs = [angular_diff(a, b) for a, b in zip(valid_decoded_targets, valid_decoded_preds)]
        accuracy = sum(d < ACCURACY for d in diffs) / len(diffs)
        accuracy_data.append(accuracy)

        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), f"{SAVE_MODEL_AS}checkpoint-{(best_acc*100):.2f}.pth")
            plot_losses(train_loss_data, valid_loss_data, f"{SAVE_MODEL_AS}checkpoint-{(best_acc*100):.2f}loss.png")
            plot_acc(accuracy_data, f"{SAVE_MODEL_AS}checkpoint-{(best_acc*100):.2f}acc.png")

            print(f"Модель сохранена: {SAVE_MODEL_AS}checkpoint-{(best_acc*100):.2f}.pth")
            log.open()
            log.write(f"Модель сохранена: {SAVE_MODEL_AS}checkpoint-{(best_acc*100):.2f}.pth")
            log.close()

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(test_loss)
        new_lr = optimizer.param_groups[0]['lr']

        log.open()
        if new_lr != prev_lr:
            message = f"Learning rate reduced from {prev_lr:.6f} to {new_lr:.6f}"
            print(message)
            log.write(message)

        table = general_table()
        table.add_row(str(epoch), str(train_loss), str(test_loss), str(accuracy), str(best_acc))
        log.write_table(table)
        log.close()

    # сохранение модели
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), SAVE_MODEL_AS)
    plot_losses(train_loss_data, valid_loss_data, f"{SAVE_MODEL_AS}loss.png")
    plot_acc(accuracy_data, f"{SAVE_MODEL_AS}acc.png")


if __name__ == "__main__":
    console = Console()

    try:
        torch.cuda.empty_cache()
        run_training(console)
    except Exception:
        console.print_exception()