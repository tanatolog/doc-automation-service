"""Модуль датасета для обучения и тестирования OCR"""
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
from typing import Tuple

ImageFile.LOAD_TRUNCATED_IMAGES = True # разрешить загрузку повреждённых изображений

class OcrDataset:
    """Класс датасета для обучения и тестирования OCR, который может использоватся DataLoader для загрузки данных"""

    def __init__(self, image_paths: list[str], targets: list[str], resize: Tuple[int, int], grayscale: bool = False):
        """Конструктор. Класс датасета для обучения и тестирования OCR, который может использоватся DataLoader для загрузки данных."""

        self.image_paths = image_paths # пути к изображениям
        self.targets = targets # метки (классы) для каждого изображения
        self.resize = resize # размер для изменения изображения (высота, ширина)
        self.grayscale = grayscale # флаг для конвертации в оттенки серого

        # Нормализация по среднему и стандартному отклонению для RGB
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        # Аугментация: только нормализация (без трансформаций)
        self.aug = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        # Если grayscale=True, задаётся преобразование для оттенков серого + преобразование в тензор
        if grayscale:
            self.transform = transforms.Compose([
                transforms.Grayscale(), 
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) 
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item: int):
        """Возвращение эл датасета по индексу"""

        image = Image.open(self.image_paths[item]).convert("RGB")
        targets = self.targets[item]

        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)

        if self.grayscale:
            image = self.transform(image)
        else:
            image = np.array(image)
            augmented = self.aug(image=image)
            image = augmented["image"]
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)

        return {
            "images": image, # (1, 50, 180)
            "targets": torch.tensor(targets, dtype=torch.long),
        }


    def custom_collate_fn(self, batch):
        """Функция объединения батча. Нужна так как на изображениях текст разной длинны."""

        images = [item["images"] for item in batch]
        targets = [item["targets"] for item in batch]

        images = torch.stack(images) # (batch_size, 50, 180)
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets_concat = torch.cat(targets)

        return {
            "images": images, # изображения (batch_size, 50, 180)
            "targets": targets_concat, # метки классов подряд (total_target_len)
            "target_lengths": target_lengths, # длина строки для кажд. изображения (batch_size)
        }
