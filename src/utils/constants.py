"""Модуль содержащий константы"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Список классов которые предсказывает OCR, ∅ - это blank
CLASSES  = ["∅", " ", ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "А", "Б", "В",
            "Г", "Д", "Е", "Ж", "З", "И", "Й", "К", "Л", "М", "Н", "О", "П", "Р",
            "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Э", "Ю", "Я"]

BB_CLASS_LABELS = ["код", "датавыдачи", "фамилия", "имя", "отчество", "пол", "датарождения", "месторождения", "серия", "номер"]

LABEL_MAP = {k: v + 1 for v, k in enumerate(BB_CLASS_LABELS)} # словарь (название: метка) начиная с 1
LABEL_MAP["background"] = 0 # добавление класса background с меткой 0
REV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()} # инвертированный словарь (метка: название)

# Хорошо различимые, контрастные цвета https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
DISTINCT_COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6","#d2f53c", "#fabebe", "#008080", 
                   "#000080", "#aa6e28", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#e6beff", "#808080", "#ffffff"]
LABEL_COLOR_MAP = {k: DISTINCT_COLORS[i] for i, k in enumerate(LABEL_MAP.keys())} # словарь цветов для рамок

# Конфигурация OCR
OCR_DIMS = 512
OCR_DROPOUT = 0.5
OCR_GRU_LAYERS = 4

# Конфигурация AngleRegressionModel
ANGLE_IMAGE_SIZE = (224, 224)
ANGLE_DROPOUT = 0.3