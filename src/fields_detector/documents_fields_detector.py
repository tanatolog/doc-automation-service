"""Модуль для класса распознавания полей паспорта РФ на изображении"""
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.serialization import add_safe_globals
import os
import glob

from utils.constants import DEVICE, REV_LABEL_MAP, LABEL_COLOR_MAP
from fields_detector.model import ObjectDetector

# пути к данным
IMAGE_PATH = "test.jpg"
MODEL_PATH = "models/fields_det_10_64.pth"

# Путь к папке с изображениями
INPUT_DIR = "app_test"
OUTPUT_DIR = "cropped_fields"

# параметры распознавания
MIN_SCORE = 0.3 # минимальный порог для того, чтобы предсказанная рамка считалась соответствующей определенному классу
MAX_OVERLAP = 0.3 # максимальное перекрытие, которое могут иметь две рамки, чтобы рамка с более низким баллом не удалялась с помощью Non-Maximum Suppression (NMS)

class DocumentFieldsDetector:
    """Класс для распознавания полей паспорта РФ на изображении"""

    def __init__(self, model_path: str, device: torch.device, color_map: dict[str, str], rev_label_map: dict[int, str], font_path: str = "./calibril.ttf"):
        """Конструктор. Класс для распознавания полей паспорта РФ."""
        self.device = device
        self.color_map = color_map
        self.rev_label_map = rev_label_map

        # Загрузка модели
        add_safe_globals([ObjectDetector])
        self.model = ObjectDetector(n_classes = len(rev_label_map))
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"\nЗагружена сжатая модель: {os.path.basename(model_path)}.\n")

        self.font = ImageFont.truetype(font_path, 15)

        # Трансформации
        self.resize = transforms.Resize((300, 300))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )


    def detect_objects_on_image(self, original_image: Image.Image, min_score: float, max_overlap: float, top_k: int = 200):
        """
        Обнаружение объектов на изображении.

        :param original_image: PIL Image
        :param min_score: минимальный порог для того, чтобы предсказанная рамка считалась соответствующей определенному классу
        :param max_overlap: максимальное перекрытие, которое могут иметь две рамки, чтобы рамка с более низким баллом не удалялась с помощью Non-Maximum Suppression (NMS)
        :param top_k: если классе обнаружено много результирующих данных, оставьте только k результатов
        :return: списки
        """
        image = self.normalize(self.to_tensor(self.resize(original_image))).to(self.device) # нормализация, на устройсво

        with torch.no_grad(): # Прямой проход
            predicted_locs, predicted_scores = self.model(image.unsqueeze(0))

        # Расшифровка предсказаний
        det_boxes, det_labels, det_scores = self.model.detect_objects(
            predicted_locs, predicted_scores,
            min_score=min_score,
            max_overlap=max_overlap,
            top_k=top_k
        )

        det_boxes = det_boxes[0].to("cpu") # перенос на cpu т.к PIL и ImageDraw поддерживают GPU

        # Приведение к изначальным координатам
        original_dims = torch.FloatTensor([
            original_image.width, original_image.height,
            original_image.width, original_image.height
        ]).unsqueeze(0)
        det_boxes = det_boxes * original_dims

        det_labels = [self.rev_label_map[l] for l in det_labels[0].to("cpu").tolist()] # приводим числовые метки классов к их названиям
        return det_boxes, det_labels
    
    
    def draw_boxes_on_image(self, original_image: Image.Image, det_boxes, det_labels, suppress=None):
        """
        Визуализация объектов на изображении.

        :param original_image: PIL Image
        :param suppress: список классов, которых не может быть на изображении, или которых не нужно предсказывать
        :return: размеченное изображение PIL Image
        """
        if not det_labels or det_labels == ["background"]: # если обнаружен только фон
            return original_image # вернуть исходное изображение

        annotated_image = original_image.copy()
        draw = ImageDraw.Draw(annotated_image)

        for i in range(det_boxes.size(0)): # цикл по всем найденым рамкам

            if suppress and det_labels[i] in suppress: # подавить ненужные классы
                continue

            # Рисование рамок
            box_location = det_boxes[i].tolist()
            color = self.color_map[det_labels[i]]
            draw.rectangle(xy=box_location, outline=color)
            draw.rectangle(xy=[l + 1. for l in box_location], outline=color) # вторая рамка со смещением в 1 пиксель, чтобы увеличить толщину линии

            # Подпись объектов в рамках
            label_text = det_labels[i].upper()
            bbox = self.font.getbbox(label_text)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

            text_location = [box_location[0] + 2., box_location[1] - text_height]
            textbox_location = [box_location[0], box_location[1] - text_height,
                                box_location[0] + text_width + 4., box_location[1]]

            draw.rectangle(xy=textbox_location, fill=color)
            draw.text(xy=text_location, text=label_text, fill="white", font=self.font)

        return annotated_image
    
    
    def crop_images_by_boxes(self, original_image: Image.Image, det_boxes, output_dir="cropped_fields"):
        """
        Обрезает исходное изображение по найденным рамкам и сохраняет каждое поле как отдельное изображение.

        :param original_image: PIL Image - исходное изображение
        :param det_boxes: tensor с координатами рамок (x_min, y_min, x_max, y_max)
        :param output_dir: директория для сохранения вырезанных изображений
        :return: список путей сохраненных изображений
        """
        os.makedirs(output_dir, exist_ok=True)
        cropped_paths = []

        for i in range(det_boxes.size(0)):
            box = det_boxes[i].tolist()
            # Преобразуем координаты в целые числа
            x_min, y_min, x_max, y_max = map(int, box)
            # Обрезаем изображение по рамке
            cropped_img = original_image.crop((x_min, y_min, x_max, y_max))
            # Сохраняем в файл
            save_path = os.path.join(output_dir, f"field_{i+1}.png")
            cropped_img.save(save_path)
            cropped_paths.append(save_path)

        return cropped_paths


if __name__ == "__main__":
    detector = DocumentFieldsDetector(
        model_path = MODEL_PATH,
        color_map = LABEL_COLOR_MAP,
        rev_label_map = REV_LABEL_MAP,
        device = DEVICE
    )

    """
    original = Image.open(IMAGE_PATH).convert("RGB")

    boxes, labels = detector.detect_objects_on_image(original, min_score=MIN_SCORE, max_overlap=MAX_OVERLAP)
    result = detector.draw_boxes_on_image(original, boxes, labels)
    result.show()
    """

    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + glob.glob(os.path.join(INPUT_DIR, "*.png"))

    for img_path in image_paths:
        try:
            original = Image.open(img_path).convert("RGB")

            boxes, labels = detector.detect_objects_on_image(
                original,
                min_score=MIN_SCORE,
                max_overlap=MAX_OVERLAP
            )

            # Имя без расширения
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_subdir = os.path.join(OUTPUT_DIR, base_name)
            os.makedirs(out_subdir, exist_ok=True)

            detector.crop_images_by_boxes(original, boxes, output_dir=out_subdir)
            print(f"Обработано: {img_path} → {out_subdir}")

        except Exception as e:
            print(f"Ошибка при обработке {img_path}: {e}")
