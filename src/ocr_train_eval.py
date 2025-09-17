"""Модуль для обучения и тестирования OCR"""
import copy
import torch
from sklearn import metrics
from rich.console import Console
from rich import print
from types import SimpleNamespace
from sklearn import preprocessing
import glob
import os
from sklearn import model_selection
import torch.backends.cudnn as cudnn
import re

from ocr.engine import train_fn, eval_fn
from utils.logging_config import general_table, predictions_table
from ocr.ocr_model import OCRModel
from utils.plot import plot_acc, plot_losses
from ocr.ocr_dataset import OcrDataset
from utils.log_file import LogFile
from utils.constants import DEVICE, CLASSES, OCR_GRU_LAYERS, OCR_DIMS, OCR_DROPOUT

cudnn.benchmark = True # настройка оптимизации свёрточных операций PyTorch, работает только на GPU и при одинаковых размерах входных данных

# Конфигурация
CFG = SimpleNamespace(
    processing=SimpleNamespace(
        device=DEVICE,
        image_width=180,
        image_height=50
    ),
    training=SimpleNamespace(
        lr=0.0003,
        batch_size=16,
        num_workers=4,
        num_epochs=100
    ),
    bools=SimpleNamespace(
        DISPLAY_ONLY_WRONG_PREDICTIONS=True,
        VIEW_INFERENCE_WHILE_TRAINING=True,
        SAVE_CHECKPOINTS=True
    ),
    paths=SimpleNamespace(
        dataset_dir="./data/ocr_dataset/images/",
        save_model_as="./models/ocr_test.pth",
        save_log_as="./models/ocr_test.txt"
    ),
    model=SimpleNamespace(
        use_attention = True,
        gray_scale = True,
        dims = OCR_DIMS, # 512
        dropout_prob = OCR_DROPOUT,
        gru_layers = OCR_GRU_LAYERS # 4
    )
)

def build_dataloaders(cfg):
    """Создание обучающей и тестировачной выборки. Создание классов датасетов и загрузчиков dataloader."""

    image_files = glob.glob(os.path.join(cfg.paths.dataset_dir, "*.png")) # пути ко всем png файлам в указанной директории

    # Получаем истинные строки из имен файлов
    original_targets = [
        re.sub(r' \(\d+\)', '', os.path.splitext(os.path.basename(x))[0])
        for x in image_files
    ]

    # Разбивание истинных строк на символы
    targets = [[c for c in x] for x in original_targets]

    # Нахождение названий всех классов символов
    targets_flat = [c for clist in targets for c in clist]
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(targets_flat) # все классы в датасете

    # Назначаем каждому классу метку от 0 до n_classes
    targets_encoded = [label_encoder.transform(seq).tolist() for seq in targets]
    targets_encoded = [ [t + 1 for t in seq] for seq in targets_encoded ] # сдвиг на 1, т.к. 0 зарезервирован под blank

    # Разделение датасета на обучающую и тестовую выборки 95\5
    (train_imgs, test_imgs, train_targets, test_targets, _, test_original_targets) = model_selection.train_test_split(
        image_files, targets_encoded, original_targets, test_size=0.001, random_state=42
    )

    # Обучающий датасет
    train_dataset = OcrDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(cfg.processing.image_height, cfg.processing.image_width),
        grayscale=cfg.model.gray_scale,
    )

    # Обучающий загрузчик
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True,
        collate_fn=train_dataset.custom_collate_fn, # добавлен кастомный collate
    )

    # Тестовый датасет
    test_dataset = OcrDataset(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(cfg.processing.image_height, cfg.processing.image_width),
        grayscale=cfg.model.gray_scale,
    )

    # Тестовый загрузчик
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=False,
        collate_fn=test_dataset.custom_collate_fn,
    )

    return train_loader, test_loader, test_original_targets, label_encoder.classes_


def run_training(cfg, console: Console):

    log = LogFile(cfg.paths.save_log_as)
    log.open()
    print(f"Конфикурация:\n{cfg}")
    log.write(f"Конфигурайия:\n{cfg}")

    # Dataset, dataloaders
    train_loader, test_loader, test_original_targets, classes = build_dataloaders(cfg)

    print(f"Количество классов в датасете: {len(classes)}")
    log.write(f"Количество классов в датасете: {len(classes)}")
    print(f"Классы: {classes}")
    log.write(f"Классы: {classes}")
    log.close()

    # Добавление класса blank
    training_classes = ["∅"]
    training_classes.extend(classes)

    assert training_classes == CLASSES # если списки не совпадают, то что то пошло не так

    # Model, optimizer, scheduler
    device = cfg.processing.device

    model = OCRModel(
        classes = training_classes,
        resolution = (cfg.processing.image_width, cfg.processing.image_height),
        dims = cfg.model.dims,
        use_attention = cfg.model.use_attention,
        grayscale = cfg.model.gray_scale,
        dropout_prob = cfg.model.dropout_prob,
        gru_layers = cfg.model.gru_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=4, verbose=True)

    best_acc = 0.0
    train_loss_data = []
    valid_loss_data = []
    accuracy_data = []

    # Обучение
    for epoch in range(cfg.training.num_epochs):
        train_loss = train_fn(model, train_loader, optimizer, device)
        train_loss_data.append(train_loss)

        valid_preds, test_loss = eval_fn(model, test_loader, device)
        valid_loss_data.append(test_loss)

        valid_captcha_preds = []

        for vp in valid_preds:

            current_preds = model.decode_predictions(vp)
            valid_captcha_preds.extend(current_preds)

        combined = list(zip(test_original_targets, valid_captcha_preds))

        if cfg.bools.VIEW_INFERENCE_WHILE_TRAINING:
            table = predictions_table()

            for idx in combined:

                if cfg.bools.DISPLAY_ONLY_WRONG_PREDICTIONS:

                    if idx[0] != idx[1]:
                        table.add_row(idx[0], idx[1])

                else:
                    table.add_row(idx[0], idx[1])

            console.print(table)

        accuracy = metrics.accuracy_score(test_original_targets, valid_captcha_preds)
        accuracy_data.append(accuracy)

        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

            if cfg.bools.SAVE_CHECKPOINTS:
                torch.save(model.state_dict(), f"{cfg.paths.save_model_as}checkpoint-{(best_acc*100):.2f}.pth")
                plot_losses(train_loss_data, valid_loss_data, f"{cfg.paths.save_model_as}checkpoint-{(best_acc*100):.2f}loss.png")
                plot_acc(accuracy_data, f"{cfg.paths.save_model_as}checkpoint-{(best_acc*100):.2f}acc.png")

                print(f"Модель сохранена: {cfg.paths.save_model_as}checkpoint-{(best_acc*100):.2f}.pth")
                log.open()
                log.write(f"Модель сохранена: {cfg.paths.save_model_as}checkpoint-{(best_acc*100):.2f}.pth")
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
        #console.print(table)
        log.write_table(table)
        log.close()

    # сохранение модели
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), cfg.paths.save_model_as)
    plot_losses(train_loss_data, valid_loss_data, f"{cfg.paths.save_model_as}loss.png")
    plot_acc(accuracy_data, f"{cfg.paths.save_model_as}acc.png")


if __name__ == "__main__":
    console = Console()

    try:
        torch.cuda.empty_cache()
        run_training(CFG, console)
    except Exception:
        console.print_exception()