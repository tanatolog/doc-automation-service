"""Модуль для класса разделения изображения с текстом на отдельные строки"""
import cv2
import numpy as np
import os
from PIL import Image


class LineSplitter:
    """Класс для разделения изображения на текстовые строки двумя различными методами"""

    def __init__(self, padding: int = 5, threshold_1: int = 100, threshold_2: int = 100, min_line_height_1: int = 15, min_line_height_2: int = 5):
        """
        Конструктор. Класс для разделения изображения на текстовые строки двумя различными методами
        
        :param padding: отступ сверху и снизу при вырезании строки
        :param threshold_1: порог черных пикселей для метода 1
        :param threshold_2: порог черных пикселей для метода 2
        :param min_line_height: минимальная высота строки в пикселях
        """
        self.padding = padding
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.min_line_height_1 = min_line_height_1
        self.min_line_height_2 = min_line_height_2


    def split_into_lines_1(self, image: Image.Image) -> list[Image.Image]:
        """
        Метод 1: Разделение строк на основе вертикальной суммы пикселей (проекция по оси Y).
        Хорошо работает на четко разделенных строках.

        :param image: pil image
        :return: лист строк pil image
        """
        image = np.array(image.convert("L"))

        # Бинаризация изображения (черный текст на белом фоне)
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Вертикальная проекция: сумма черных пикселей по строкам
        projection = np.sum(binary, axis=1)

        if np.sum(projection) == 0: # Нет текста
            return []  

        lines = []
        in_line = False
        H = image.shape[0]

        # Поиск участков с черными пикселями — потенциальных строк
        for y, val in enumerate(projection):

            if val > 0 and not in_line:
                start_y = y
                in_line = True
            elif val == 0 and in_line:
                end_y = y
                in_line = False

                if end_y - start_y > self.min_line_height_1:
                    y1 = max(0, start_y - self.padding)
                    y2 = min(H, end_y + self.padding)

                    # Извлечение строки и фильтрация по количеству черных пикселей
                    line_img = image[y1:y2, :]
                    line_bin = binary[y1:y2, :]
                    black_pixels = cv2.countNonZero(line_bin)

                    if black_pixels > self.threshold_1:
                        lines.append(Image.fromarray(line_img))

        return lines
    

    def split_into_lines_2(self, image: Image.Image) -> list[Image.Image]:
        """
        Метод 2: Разделение на строки с использованием горизонтальной проекции через cv2.reduce().
        Более устойчив к шуму, подходит при неявных разделениях строк.

        :param image: pil image
        :return: лист строк pil image
        """
        img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Бинаризация изображения
        _, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Горизонтальная проекция (усреднение по строкам)
        hist = cv2.reduce(threshed, 1, cv2.REDUCE_AVG).reshape(-1)

        if np.sum(hist) == 0: # Нет текста
            return []

        th = 60  # порог для поиска границ

        H, W = img.shape[:2]

        # Определение верхней и нижней границ строк
        uppers = [y for y in range(H - 1) if hist[y] <= th and hist[y + 1] > th]
        lowers = [y for y in range(H - 1) if hist[y] > th and hist[y + 1] <= th]

        lines = []

        for y1, y2 in zip(uppers, lowers):

            if (y2 > y1) and (y2 - y1 > self.min_line_height_2):
                y1 = max(0, y1 - self.padding)
                y2 = min(H, y2 + self.padding)

                # Извлечение строки и проверка на наличие достаточного количества текста
                line_img = img[y1:y2, :]
                line_bin = threshed[y1:y2, :]
                black_pixels = cv2.countNonZero(line_bin)

                if black_pixels > self.threshold_2:
                    line_pil = Image.fromarray(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
                    lines.append(line_pil)

        return lines
    

    def combo_split_into_lines(self, pil_image: Image.Image) -> list[Image.Image]:
        """
        Комбинированный метод: сначала используется быстрый и простой метод 1.
        Если он не дал результата — применяется метод 2.

        :param image: pil image
        :return: лист строк pil image
        """
        lines1 = self.split_into_lines_1(pil_image)
        lines2 = self.split_into_lines_2(pil_image)

        if (len(lines1) >= len(lines2)) or (len(lines2) > 3):
            return lines1
        else:
            return lines2


    def search_and_save_lines(self, path: str, output_dir_path: str):
        """
        Комбинированный метод: сначала используется быстрый и простой метод 1.
        Если он не дал результата — применяется метод 2.

        :param path: путь к изображению
        :param output_dir_path: путь к выходной папке
        :return: количество найденных строк
        """
        os.makedirs(output_dir_path, exist_ok=True)
        image = Image.open(path)
        lines = self.combo_split_into_lines(image)

        for idx, line_img in enumerate(lines):
            line_img.save(os.path.join(output_dir_path, f"line_{idx+1}.png"))

        return len(lines)


# Пример запуска
if __name__ == "__main__":
    path = "ocr_dataset/texts/16.jpg"
    output_dir = "ocr_dataset/texts/lines"
    splitter = LineSplitter()
    _ = splitter.search_and_save_lines(path, output_dir)
