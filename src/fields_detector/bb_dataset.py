"""Этот модуль предназначен для описания класса датасета для работы с данными"""
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision.transforms.functional as Tfunctional
import random

from fields_detector.jaccard import find_jaccard_indx

DATASET_PATH = "data/fields_dataset/"
SPLIT = "train"
USE_DIFFICULT = True

def expand(image, boxes, filler):
    """
    Выполнение операции уменьшения масштаба, поместив изображение на холст большего размера.
    Помогает научиться распознавать объекты меньшего размера.

    :param image: тензор изображения вида (3, original_h, original_w)
    :param boxes: тензор ограничивающих рамок в координатах границ вида (n_objects, 4)
    :param filler: RBG заполнитель холста, лист [R, G, B]
    :return: расширенное изображение, измененные координаты рамок
    """
    # рассчет размеров расширенного изображения
    original_h = image.size(1) # изначальная высота
    original_w = image.size(2) # изначальная длинна
    max_scale = 3 # макс возможное увеличение
    scale = random.uniform(1, max_scale) # рассчет увеличения
    new_h = int(scale * original_h) # новая высота
    new_w = int(scale * original_w) # новая длинна

    # создание изображения с помощью наполнителя
    filler_tensor = torch.tensor(filler, dtype=torch.float32) / 255.0  # (3,)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float32) * filler_tensor.view(3, 1, 1) # создание холста (3, new_h, new_w)
    # Не использовать expand() вот так new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # потому что все расширенные значения делят одну память, так изменение одного пикселя изменит все

    # размещение исходного изображения в случайных координатах на холсте
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # рассчет новых координат рамок
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)  # (n_objects, 4)

    return new_image, new_boxes


def random_crop(image, boxes, labels, difficulties):
    """
    Выполняет случайное обрезание изображения. Обучение распознавания крупных объектов и фрагментов.
    Объекты могут быть вырезаны полностью.

    :param image: тензор изображения вида (3, original_h, original_w)
    :param boxes: тензор ограничивающих рамок в координатах границ вида (n_objects, 4)
    :param labels:названия объектов, тензор (n_objects)
    :param difficulties: трудность обнаружения объектов, тензор (n_objects)
    :return: обрезанное изображение, измененные координаты рамок, измененные названия, измененные трудности
    """
    original_h = image.size(1) # изначальная высота
    original_w = image.size(2) # изначальная длинна

    # выбирор минимального перекрытия до тех пор, пока не будет найдена удачная обрезка
    while True:
        # случайно рисует значения мин перекрытич
        min_overlap = random.choice([0., .1, .3, .5, .7, .9]) # None это отсутсвие обрезки

        if min_overlap is None:
            return image, boxes, labels, difficulties

        max_trials = 50 # пробуется 50 раз
        for _ in range(max_trials):
            # рассчет новых размеров изображения
            min_scale = 0.3 # размеры обрезки должны быть [0.3, 1] от исходных
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            aspect_ratio = new_h / new_w # соотношение сторон должно быть в [0.5, 2]
            if not 0.5 < aspect_ratio < 2:
                continue

            # рассчет координат обрезки изображения
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom]) # (4)

            # рассчет перекрытия Жаккара между обрезкой и ограничивающими рамками
            jaccard_indx = find_jaccard_indx(crop.unsqueeze(0), boxes) # (1, n_objects)
            jaccard_indx = jaccard_indx.squeeze(0) # (n_objects)

            if jaccard_indx.max().item() < min_overlap:
                continue

            new_image = image[:, top:bottom, left:right] # образание изображения (3, new_h, new_w)
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2. # нахождение центров оригинальных рамок (n_objects, 2)

            # нахождение рамок, центры которых находятся в обрезке
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (bb_centers[:, 1] < bottom) # (n_objects)

            if not centers_in_crop.any(): # если нет центров рамок в обрезке, то след итерация
                continue

            # удаление рамок, центры которых не попадают в обрезку
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # расчет новых координат рамок в обрезке
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] - [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] - [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties

def resize(image, boxes, dims=(300, 300), return_fraction_coords=True):
    """
    Изменение размера изображения. Так же можно получить дробные координаты рамок

    :param image: PIL Image
    :param boxes: ограничивающие рамки в координатах границ, тензор (n_objects, 4)
    :return: измененное изображение, измененные координаты рамок в дробных координатах координатах границ или координатах границ
    """
    new_image = Tfunctional.resize(image, dims) # изменение размера

    # изменение координат
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims # координаты рамок в дробных координатах

    if return_fraction_coords == False: # рефракторинг not на ==
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims # координаты рамок в координатах границ

    return new_image, new_boxes


def photometric_distort(image):
    """Изменяет яркость, контрастность, насыщенность и тон с вероятностью 50% в случайном порядке"""

    new_image = image

    # искажения
    distortions = [Tfunctional.adjust_brightness,
                   Tfunctional.adjust_contrast,
                   Tfunctional.adjust_saturation,
                   Tfunctional.adjust_hue]

    random.shuffle(distortions)

    for distort in distortions: # цикл по искажениям
        if random.random() < 0.5: # вероятность 50%

            if distort.__name__ == "adjust_hue": # здесь исправлено is на ==
                param = random.uniform(-18 / 255., 18 / 255.)
            else:
                param = random.uniform(0.5, 1.5)

            new_image = distort(new_image, param) # произвести искажение

    return new_image


def examlpe_preprocessing(image, boxes, labels, difficulties, split: str):
    """
    Функция предварительной обработки данных.

    :param image: PIL Image
    :param boxes: ограничивающие рамки в координатах границ, тензор (n_objects, 4)
    :param labels: названия объектов, тензор (n_objects)
    :param difficulties: трудность обнаружения объектов, тензор (n_objects)
    :param split: "TRAIN" или "TEST", применяются разные наборы преобразований
    :return: tuple: преобразованные image, координаты рамок в дробных координатах границ, названия объектов, трудности
    """
    assert split in {"TRAIN", "TEST"}

    mean = [0.485, 0.456, 0.406] # среднее значение данных ImageNet
    std = [0.229, 0.224, 0.225] # стандартное отклонение данных ImageNet

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties

    if split == "TRAIN": # только для тренировочной выборки
        new_image = photometric_distort(new_image) # серия случайных фотометрических искажений

        mean_color = list(int(x) for x in new_image.resize((1,1)).getpixel((0,0))) # средний цвет изображения

        new_image = Tfunctional.to_tensor(new_image) # преобразование pil image в тензор

        if random.random() < 0.5: # расширение изображения с вероятностью 50%, тренировка обнаружения небольших объектов
            new_image, new_boxes = expand(new_image, boxes, filler=mean_color)

        if random.random() < 0.5: # случайное обрезание (уменьшение), тренировка обнаружения частичных объектов
            new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels, new_difficulties)

        new_image = Tfunctional.to_pil_image(new_image) # преобразование тензора в pil image

    # приведение изображения к размеру 300х300, преобразование абсолютных координат рамок в дробную форму
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

    new_image = Tfunctional.to_tensor(new_image) # преобразование pil image в тензор
    new_image = Tfunctional.normalize(new_image, mean=mean, std=std) # нормализация по среднему значению и стандартному отклонению

    return new_image, new_boxes, new_labels, new_difficulties


class BbDataset(Dataset):
    """Класс датасета, который может использоваться в PyTorch DataLoader для формирования батчей"""

    def __init__(self, dataset_path: str, split: str, keep_difficult: bool):
        """
        Конструктор. Класс датасета, который может использоваться в PyTorch DataLoader для формирования батчей

        :param dataset_path: путь к папке с json файлами, которые описывают датасет
        :param split: значение "TRAIN" или "TEST", определяет какая выборка будет заргужена
        :param keep_difficult:  включать ли размеченные как "difficult" объекты
        """
        self.split = split.upper()

        assert self.split in {"TRAIN", "TEST"}

        #self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # чтение json файлов
        with open(os.path.join(dataset_path, self.split + "_images.json"), "r") as j:
            self.images = json.load(j)
        with open(os.path.join(dataset_path, self.split + "_objects.json"), "r") as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects) # размер должен быть одинаков


    def __getitem__(self, i):
        """Возвращает один элемент по индексу i, применяется автоматически при обучении через DataLoader"""
        image = Image.open(self.images[i], mode="r") # чтение изображения
        image = image.convert("RGB") # преобразование в 3-канальный rgb

        # получение объектов этого изображения (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects["boxes"])  # формат тензора (n_objects, 4)
        labels = torch.LongTensor(objects["labels"])  # (n_objects)
        difficulties = torch.ByteTensor(objects["difficulties"])  # (n_objects)

        if not self.keep_difficult: # отбрасываем сложные объекты
            boxes = boxes[1 - difficulties] # создается булева маска, чтобы оставить только простые объекты
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        image, boxes, labels, difficulties = examlpe_preprocessing(image, boxes, labels, difficulties, split=self.split) # предобработка

        return image, boxes, labels, difficulties # возврат тензоров image ([3, H, W]), boxes [N, 4], labels [N], difficulties [N]


    def __len__(self):
        """Возвращает количество изображений в датасете, необходим для корректной работаты с DataLoader"""
        return len(self.images)


    def collate_fn(self, batch):
        """
        Функция объединения батча, используется DataLoader. Описывает как объеденить тензоры разной размерности. 
        Так как число объектов на изображениях может отличаться, то размеры boxes, labels, difficulties различны.

        :param batch: N наборов из __getitem__()
        :return: тензор изображений, списки тензоров разного размера bounding boxes, labels, difficulties
        """
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for img in batch: # цикл по изображениям в батче
            images.append(img[0])
            boxes.append(img[1])
            labels.append(img[2])
            difficulties.append(img[3])

        images = torch.stack(images, dim=0) # приведение к общему тензору размера (N, 3, H, W)

        return images, boxes, labels, difficulties # тензор (N, 3, 300, 300), 3 списка по N тензоров
    

if __name__ == "__main__":
    dataset = BbDataset(
        dataset_path = DATASET_PATH,
        split = SPLIT,
        keep_difficult = USE_DIFFICULT
    )
    image_tensor, _, _, _ = dataset[402]

    # Обратная нормализация
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    unnormalized_image = image_tensor * std + mean

    # Преобразование обратно в PIL.Image и отображение
    image = Tfunctional.to_pil_image(unnormalized_image.clamp(0, 1))
    image.show()
