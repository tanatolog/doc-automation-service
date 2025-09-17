"""Модуль для генерации синтетических данных для обучения OCR"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple
from PIL import ImageFilter

NUM_IMAGES = 10000
MIN_CONTRAST = 30 # 0 - черный 255 - белый
OUTPUT_DIR = "data/ocr_dataset/images/"
FONTS_DIR = "data/data_for_gen/fonts/"

class OCRDatasetGenerator:
    """Класс для генерации ситнетических изображений с текстом для обучения OCR"""

    def __init__(self, fonts_dir: str, output_dir: str, min_contrast: int):
        """Конструктор. Класс для генерации ситнетических изображений с текстом для обучения OCR."""
        self.fonts_dir = fonts_dir
        self.output_dir = output_dir
        self.min_contrast = min_contrast

        os.makedirs(self.output_dir, exist_ok=True) # создание папки, если нет

        self.font_paths = [
            os.path.join(self.fonts_dir, f)
            for f in os.listdir(self.fonts_dir)
            if f.lower().endswith(".ttf")
        ]

        if not self.font_paths: # если шрифтов в папке нет
            raise FileNotFoundError("Шрифты не найдены в данной папке")


    def __brightness(self, color: Tuple[int, int, int]) -> float:
        """Расчет яркости rgb цвета"""
        r, g, b = color
        return 0.299 * r + 0.587 * g + 0.114 * b


    def __generate_contrast_colors(self, contrast: int) -> Tuple:
        """Генерация случайного светлого цвета с порогом по яркости"""

        while True:
            color_1 = tuple(random.randint(0, 255) for _ in range(3))
            color_2 = tuple(random.randint(0, 255) for _ in range(3))

            if abs(self.__brightness(color_1) - self.__brightness(color_2)) >= contrast :
                return color_1, color_2


    def __apply_gaussian_noise(self, img: Image.Image, mean: int = 5, std: int = 16) -> Image.Image:
        """Добавление гауссовского шума к изображению"""
        np_img = np.array(img).astype(np.int16)
        noise = np.random.normal(mean, std, np_img.shape).astype(np.int16)
        noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)


    def generate_kod(self, num: int):
        """Генерация изображений с шаблоном 123-456"""

        for i in range(num):
            text = f"{random.randint(0, 999):03d}-{random.randint(0, 999):03d}"
            img = self._generate_image(text)
            filename = self.__create_img_name(text)
            img.save(os.path.join(self.output_dir, filename))


    def generate_seriya(self, num: int):
        """Генерация изображений с шаблоном 12   34"""

        for i in range(num):
            ser1 = random.randint(0, 99)
            ser2 = random.randint(0, 99)
            text = f"{ser1:02d}   {ser2:02d}"
            img = self._generate_image(text)
            text = f"{ser1:02d} {ser2:02d}"
            filename = self.__create_img_name(text)
            img.save(os.path.join(self.output_dir, filename))

    def generate_date(self, num: int):
        """Генерация изображений с шаблоном ДД.ММ.ГГГГ"""

        for i in range(num):
            text = f"{random.randint(1, 31):02d}.{random.randint(1, 12):02d}.{random.randint(1930, 2025)}"
            img = self._generate_image(text)
            filename = self.__create_img_name(text)
            img.save(os.path.join(self.output_dir, filename))


    def generate_number(self, num: int):
        """Генерация изображений с шаблоном 123456"""

        for i in range(num):
            text = f"{random.randint(0, 999999):06d}"
            img = self._generate_image(text)
            filename = self.__create_img_name(text)
            img.save(os.path.join(self.output_dir, filename))


    def generate_from_file(self, text_file_path: str, limit: int = None):
        """
        Генерирация изображений на основе строк из файла.
        
        :param text_file_path: путь к текстовому файлу с данными.
        :param limit: максимальное число строк для обработки (если None, обрабатываются все).
        """
        if not os.path.exists(text_file_path):
            raise FileNotFoundError(f"Файл {text_file_path} не найден.")

        with open(text_file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if limit:
            lines = lines[:limit]

        for text in lines:
            text = text.upper()
            img = self._generate_image(text)
            filename = self.__create_img_name(text)
            img.save(os.path.join(self.output_dir, filename))


    def __create_img_name(self, text: str) -> str:
        """Создание имени для изображения без Ё.,-"""
        text = text.replace("Ё", "Е")
        #text = text.replace(" ", "")
        text = text.replace("-", "")
        #text = text.replace(".", "")
        text = text.replace(",", "")
        return f"{text}.png"


    def generate_addres(self, text_file_path: str, limit: int = None):
        """Генерирация изображений с адресами на основе строк из файла."""

        if not os.path.exists(text_file_path):
            raise FileNotFoundError(f"Файл {text_file_path} не найден.")

        with open(text_file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if limit:
            lines = lines[:limit]

        for text in lines:

            # Удаление случайного числа слов (0–2), но оставляя хотя бы одно
            words = text.split()
            if len(words) > 1:
                remove_n = random.randint(0, min(2, len(words) - 1))

                if remove_n > 0:
                    words = words[:-remove_n]

                text = ' '.join(words)

            # Случайное добавление фраз
            add = random.randint(0, 3)
            if add == 0:
                prefixes = ["г. ", "гор. "]
                prefix = random.choice(prefixes)
                text = f"{prefix}{text}"

                # Обрезка текста до 24 символов, если нужно
                if len(text) > 24:
                    text = text[:24].rstrip()
                    if not text:
                        continue

            elif ((add == 1) or (add == 2)) and (len(words) == 1):
                suffixes = [" обл.", " области"]
                suffix = random.choice(suffixes)
                text = f"{text}{suffix}"

            text = text.upper()
            img = self._generate_image(text)
            filename = self.__create_img_name(text)
            img.save(os.path.join(self.output_dir, filename))


    def _generate_image(self, text: str) -> Image.Image:
        """Создание изображения с заданным текстом"""
        font_path = random.choice(self.font_paths) # случайный шрифт
        font_size = random.randint(10, 60) # случайный размер шрифта

        bg_color, text_color = self.__generate_contrast_colors(self.min_contrast) # цвет фона, цвет текста

        font = ImageFont.truetype(font_path, font_size) # загрузка шрифта

        # Вычисление размера текста
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Базовый размер изображения + запас под смещение
        padding = 30  # увеличен для диапазона смещений
        image_width = text_width + 2 * padding
        image_height = text_height + 2 * padding

        # Диапазоны случайных смещений
        max_dx = padding
        max_dy = padding
        dx = random.randint(-max_dx // 2, max_dx // 2)
        dy = random.randint(-max_dy // 2, max_dy // 2)

        # Координаты отрисовки с учётом смещения
        text_x = (image_width - text_width) // 2 + dx
        text_y = (image_height - text_height) // 2 + dy

        img = Image.new("RGB", (image_width, image_height), color=bg_color)
        draw = ImageDraw.Draw(img)
        draw.text((text_x, text_y), text, font=font, fill=text_color)

        # Случайный поворот
        angle = random.uniform(-2, 2)  # градусы
        img = img.rotate(angle, expand=True, fillcolor=bg_color)
        
        # Добавление гауссовского шума
        img = self.__apply_gaussian_noise(img)

        # Случайное размытие
        if random.random() < 0.4:
            blur_radius = random.uniform(0.5, 0.8)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        return img


if __name__ == "__main__":
    
    generator = OCRDatasetGenerator(
        fonts_dir = FONTS_DIR,
        output_dir = OUTPUT_DIR,
        min_contrast = MIN_CONTRAST,
    )

    generator.generate_kod(NUM_IMAGES)
    print("Код готов")
    generator.generate_addres("data/data_for_gen/addreses.txt", NUM_IMAGES + 4803)
    generator.generate_addres("data/data_for_gen/addreses.txt", int(NUM_IMAGES * 0.5))
    print("Адрес готов")
    generator.generate_from_file("data/data_for_gen/sex.txt")
    print("Пол готов")
    generator.generate_from_file("data/data_for_gen/midnames.txt", int(NUM_IMAGES * 0.5) + 5)
    print("Отчество готово")
    generator.generate_from_file("data/data_for_gen/names.txt", int(NUM_IMAGES * 0.5) + 4)
    print("Имя готово")
    generator.generate_from_file("data/data_for_gen/surnames.txt", NUM_IMAGES + 5000)
    print("Фамилия готова")
    generator.generate_seriya(NUM_IMAGES)
    print("Серия готов")
    generator.generate_number(NUM_IMAGES)
    print("Номер готов")
    generator.generate_date(NUM_IMAGES)
    print("Дата готов")
