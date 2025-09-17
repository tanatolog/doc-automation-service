"""Модуль для класса-интерфейса OCR модели"""
import torch
from PIL import Image
from torchvision import transforms
import os

from ocr.ocr_model import OCRModel
from utils.constants import CLASSES, DEVICE, OCR_DIMS, OCR_DROPOUT, OCR_GRU_LAYERS

MODEL_PATH = "models/ocr_3.pthcheckpoint-91.26.pth"
TEST_PATH = "data/ocr_dataset/test"

class TextDetector:
    """Класс-интерфейс для работы с OCR"""

    def __init__(self, model_path: str, classes: list[str], ocr_dims: int, ocr_dropout: float, ocr_gru_layers: int, device: torch.device):
        """Конструктор. Класс-интерфейс для работы с OCR."""
        self.device = device

        # Инициализация модели
        self.model = OCRModel(
            classes = classes,
            dims = ocr_dims,
            use_attention = True,
            grayscale = True,
            dropout_prob = ocr_dropout,
            gru_layers = ocr_gru_layers
        ).to(self.device)

        # Загрузка весов
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"\nЗагружена модель {model_path}\n")

        self.model.eval()

        # Предобработка изображения
        self.transform = transforms.Compose([
            transforms.Grayscale(), # Преобразование в 1 канал
            transforms.Resize((50, 180), interpolation=Image.BILINEAR),
            transforms.ToTensor(), # Преобразование в тензор
            transforms.Normalize((0.5,), (0.5,)) # Нормализация до [-1, 1]
        ])

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Предобработка изображения. Работает токлько с grayscale."""
        image = Image.open(image_path).convert("RGB") # исходное RGB изображение
        image = self.transform(image) # применение трансформаций: (1, 50, 180)
        image = image.unsqueeze(0).float()  # добавление размерности батча: (1, 1, 50, 180)

        return image.to(self.device) # на gpu

    def predict(self, image_path: str) -> str:
        """Предсказание текста с изображения."""
        image = self.preprocess_image(image_path)

        # Предсказание
        with torch.no_grad():
            preds, _ = self.model(image)

        return self.model.decode_predictions(preds)[0] # т.к. один элемент в батче
    
    
    def predict_from_image(self, pil_image: Image.Image) -> str:
        """Предсказание текста с изображения PIL."""
        # Преобразование изображения
        image = self.transform(pil_image.convert("RGB")) # RGB → Grayscale + Resize + Normalize
        image = image.unsqueeze(0).float().to(self.device) # (1, 1, 50, 180)

        # Предсказание
        with torch.no_grad():
            preds, _ = self.model(image)

        return self.model.decode_predictions(preds)[0]  # один элемент в батче


if __name__ == "__main__":
    ocr = TextDetector(
        model_path = MODEL_PATH,
        classes = CLASSES,
        ocr_dims = OCR_DIMS,
        ocr_dropout = OCR_DROPOUT,
        ocr_gru_layers = OCR_GRU_LAYERS,
        device = DEVICE
    )

    for filename in os.listdir(TEST_PATH):

        if filename.lower().endswith((".jpg", ".png")):
            filepath = os.path.join(TEST_PATH, filename)

            try:
                text = ocr.predict(filepath)
                print(f"{filename}: {text}")
            except Exception as e:
                print(f"Ошибка обработки {filename}: {e}")


