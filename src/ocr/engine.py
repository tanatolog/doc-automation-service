from rich.progress import track
from typing import Any, List, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_fn(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, device: torch.device) -> float:
    """
    Обучает модель с использованием CTC или CrossEntropy.

    Аргументы:
        model (nn.Module): Обучаемая модель.
        data_loader (DataLoader): DataLoader, предоставляющий батчи обучающих данных.
        optimizer (Optimizer): Оптимизатор.
        device (torch.device): Устройство для вычислений (CPU или CUDA).

    Возвращает:
        fin_loss (float): Среднее значение функции потерь за эпоху.
    """
    model.train()
    fin_loss = 0

    for data in track(data_loader, description="😪 Training..."):
        images = data["images"].to(device)

        for key, value in data.items():
            data[key] = value.to(device) if isinstance(value, torch.Tensor) else value

        optimizer.zero_grad()

        images = data["images"]
        targets = data["targets"]
        target_lengths = data["target_lengths"]

        model = model.to(device)
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        logits, loss = model(images=images, targets=targets, target_lengths=target_lengths)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        fin_loss += loss.item()

    return fin_loss / len(data_loader)



def eval_fn(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[List[Any], float]:
    """
    Оценка модели на предоставленных данных.

    Параметры:
        model (nn.Module): Модель PyTorch, которую необходимо оценить.
        data_loader (DataLoader): Объект PyTorch DataLoader, предоставляющий батчи данных для оценки.
        device (torch.device): Устройство PyTorch (например, 'cpu' или 'cuda'), на которое будут загружены данные и модель.

    Возвращает:
        fin_preds (list): Список предсказаний, сделанных моделью на данных для оценки.
        fin_loss (float): Среднее значение функции потерь по всем батчам данных.
    """
    model.eval()
    with torch.no_grad():
        fin_loss = 0
        fin_preds = []
        for data in track(data_loader, description="🤔 Testing ..."):
            for key, value in data.items():
                data[key] = value.to(device)

            batch_preds, loss = model(**data)
            fin_loss += loss.item()
            fin_preds.append(batch_preds)
        return fin_preds, fin_loss / len(data_loader)
