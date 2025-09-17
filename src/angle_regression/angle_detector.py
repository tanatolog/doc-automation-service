"""Модуль для класса-интерфейса модели регрессии угда поворота"""
import torch
from PIL import Image
import torchvision.transforms.functional as Tfunctional
import os

from angle_regression.angle_model import AngleRegressionModel
from utils.constants import DEVICE, ANGLE_DROPOUT, ANGLE_IMAGE_SIZE
from angle_regression.trigonometry import decode_angles

MODEL_PATH = "models/angle_det_test.pthcheckpoint-81.58.pth"
TEST_PATH = "data/angle_dataset/test"

class AngleDetector:
    """Класс-интерфейс для работы с моделью регрессии угла поворота"""

    def __init__(self, model_path: str, angle_dropout: float, image_size: tuple[int, int], device: torch.device):
        """Конструктор. Класс-интерфейс для работы с моделью регрессии угла поворота."""
        self.device = device
        self.image_size = image_size

        # Инициализация модели
        self.model = AngleRegressionModel(dropout_prob = angle_dropout).to(self.device)

        # Загрузка весов
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"\nЗагружена модель {model_path}\n")

        self.model.eval()


    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Предобработка изображения для подачи в моедель"""
        image = image.convert("RGB") # исходное RGB изображение
        image = Tfunctional.to_tensor(image) # преобразование в тензор (3, h, w)
        image = Tfunctional.resize(image, self.image_size) # изменение размера (3, 300, 300)
        image = image.unsqueeze(0).float()  # добавление размерности батча: (1, 3, 300, 300)

        return image.to(self.device) # на gpu

    def predict_from_path(self, image_path: str) -> list[float]:
        """Предсказание текста с изображения"""
        image = Image.open(image_path, mode="r") # чтение изображения
        image = self.preprocess_image(image)

        # Предсказание
        with torch.no_grad():
            preds = self.model(image)

        decoded_preds = decode_angles(preds)

        return decoded_preds
    
    
    def predict_from_image(self, pil_image: Image.Image) -> list[float]:
        """Предсказание текста с изображения PIL"""
        image = self.preprocess_image(pil_image)

        # Предсказание
        with torch.no_grad():
            preds = self.model(image)

        decoded_preds = decode_angles(preds)

        return decoded_preds


if __name__ == "__main__":
    angle_detector = AngleDetector(
        model_path = MODEL_PATH,
        angle_dropout = ANGLE_DROPOUT,
        device = DEVICE,
        image_size = ANGLE_IMAGE_SIZE
    )

    for filename in os.listdir(TEST_PATH):

        if filename.lower().endswith(".jpg"):
            filepath = os.path.join(TEST_PATH, filename)

            try:
                angle = angle_detector.predict_from_path(filepath)[0]
                print(f"{filename}: угол = {angle:.2f}°")
            except Exception as e:
                print(f"Ошибка обработки {filename}: {e}")

