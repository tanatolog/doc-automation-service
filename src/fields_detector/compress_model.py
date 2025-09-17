# Сжатие модели, чтобы ее можно было загрузить на github
import torch
from fields_detector.model import ObjectDetector
from torch.serialization import add_safe_globals

MODEL_PATH = "models/fields_det_11_112.pth"
COMPRESSED_PATH = "models/fields_det_11_112_compressed.pth"
DEVICE = "cpu"

# Загружаем полный чекпоинт
add_safe_globals([ObjectDetector])
checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
model = checkpoint["model"].to(DEVICE)
model.eval()

# Сохраняем только state_dict
torch.save(model.state_dict(), COMPRESSED_PATH)
print(f"Сжатая модель сохранена в {COMPRESSED_PATH}")