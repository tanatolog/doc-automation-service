"""Этот модуль предназначен для описания класса датасета для задачи определения угла поворота"""
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as Tfunctional

from angle_regression.trigonometry import angle_to_sincos, sincos_to_angle
from utils.constants import ANGLE_IMAGE_SIZE

DATASET_PATH = "data/angle_dataset/images"
MAX_ANGLE = 180


def photometric_distort(image):
    """Изменяет яркость, контрастность, насыщенность и тон с вероятностью 50% в случайном порядке"""
    # искажения
    distortions = [Tfunctional.adjust_brightness,
                   Tfunctional.adjust_contrast,
                   Tfunctional.adjust_saturation,
                   Tfunctional.adjust_hue]

    random.shuffle(distortions)

    for distort in distortions: # цикл по искажениям
        if random.random() < 0.5: # вероятность 50%

            if distort.__name__ == "adjust_hue":
                param = random.uniform(-18 / 255., 18 / 255.)
            else:
                param = random.uniform(0.5, 1.5)

            image = distort(image, param) # произвести искажение

    return image


def expand(image, filler):
    """
    Выполнение операции уменьшения масштаба, поместив изображение на холст большего размера.
    Помогает научиться распознавать объекты меньшего размера.

    :param image: тензор изображения вида (3, original_h, original_w)
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

    return new_image


def random_crop(image: Image, scale_range=(0.6, 1.0)):
    """
    Выполняет случайное обрезание изображения. Обучение распознавания крупных объектов и фрагментов.
    
    :param image: PIL Image
    :param scale_range: диапазон масштабов по ширине и высоте (по умолчанию от 60% до 100%)
    :return: обрезанное изображение
    """
    w, h = image.size

    scale_w = random.uniform(*scale_range)
    scale_h = random.uniform(*scale_range)

    new_w = int(w * scale_w)
    new_h = int(h * scale_h)

    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)

    return image.crop((left, top, left + new_w, top + new_h))


class AngleDataset(Dataset):
    """Класс датасета, который может использоваться в PyTorch DataLoader для формирования батчей"""

    def __init__(self, image_paths: list[str], max_angle: float, image_size: tuple):
        """
        Конструктор. Класс датасета, который может использоваться в PyTorch DataLoader для формирования батчей

        :param image_paths: лист путей к jpg изображениям
        :param max_angle: максимальный угол поворота (в обе стороны)
        :param image_size: выходной размер изображения (ширина, высота)
        """

        self.image_paths = image_paths
        self.max_angle = max_angle
        self.image_size = image_size


    def __getitem__(self, i): # откуда-то белые рамки появляются
        """Возвращает один элемент по индексу i, применяется автоматически при обучении через DataLoader"""
        image = Image.open(self.image_paths[i], mode="r") # чтение изображения
        image = image.convert("RGB") # преобразование в 3-канальный rgb

        # Работа с углом поворота
        angle_randomer = random.random()
        if angle_randomer <= 0.2: # рандомим угол поворота
            angle = 0.0
        elif angle_randomer <= 0.4:
            angle = random.choice([90.0, -90.0, 180.0, -180.0])
        else:
            angle = random.uniform(-self.max_angle, self.max_angle)

        sin, cos = angle_to_sincos(angle)
        #print(f"\nangle = {angle:.2f}, sin = {sin:.2f}, cos = {cos:.2f}\n")
        target = torch.tensor([sin, cos], dtype=torch.float32)

        # Аугментации
        image = photometric_distort(image)

        if random.random() < 0.5: # обрезание изображения с вероятностью 50%, обучение распознавания крупных объектов и фрагментов
            image = random_crop(image)

        mean_color_tup = tuple(int(x) for x in image.resize((1,1)).getpixel((0,0))) # средний цвет изображения
        image = image.rotate(angle, expand=True, fillcolor=mean_color_tup) # поворот с заполнением средним цветом

        image = Tfunctional.to_tensor(image) # преобразование в тензор

        if random.random() < 0.5: # расширение изображения с вероятностью 50%, тренировка обнаружения небольших объектов
            mean_color_list = [mean_color_tup[0], mean_color_tup[1], mean_color_tup[2]]
            image = expand(image, mean_color_list)

        image = Tfunctional.resize(image, self.image_size) # изменение размера

        # Нормализация
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = Tfunctional.normalize(image, mean=mean, std=std)

        return image, target # тензор (3, 224, 224), тензор (2)


    def __len__(self):
        """Возвращает количество изображений в датасете, необходим для корректной работаты с DataLoader"""
        return len(self.image_paths)
    

if __name__ == "__main__":
    image_paths = sorted([ # получение всех путей к изображениям
        os.path.join(DATASET_PATH, fname)
        for fname in os.listdir(DATASET_PATH)
        if fname.lower().endswith(".jpg")
    ])

    dataset = AngleDataset(
        image_paths = image_paths,
        max_angle = MAX_ANGLE,
        image_size = ANGLE_IMAGE_SIZE
    )

    image_tensor, target = dataset[123]
    angle = sincos_to_angle(target[0], target[1])
    print(f"Угол: {angle:.2f} градусов")

    # Обратная нормализация
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    unnormalized_image = image_tensor * std + mean

    # Преобразование обратно в PIL.Image и отображение
    image = Tfunctional.to_pil_image(unnormalized_image.clamp(0, 1))
    image.show()