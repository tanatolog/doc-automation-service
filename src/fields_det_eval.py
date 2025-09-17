"""Модуль оценки точности модели детекции объектов"""
from tqdm import tqdm
from pprint import PrettyPrinter
import torch
from torch.serialization import add_safe_globals

from fields_detector.bb_dataset import BbDataset
from utils.constants import LABEL_MAP, REV_LABEL_MAP, DEVICE
from fields_detector.jaccard import find_jaccard_indx
from fields_detector.model import ObjectDetector

SPLIT = "test" #"train"
DATASET_FOLDER = "data/fields_dataset/"
USE_DIFFICULT = True # использовать трудные объекты
BATCH_SIZE = 16
WORKERS_AMOUNT = 4 # колличество потоков, кот будут загружать данные из датасета, исп. в DataLoader
MODEL_PATH = "models/fields_det_11_112_compressed.pth" # путь к модели

def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Вычислениe Mean Average Precision (mAP) — основной метрики для задачи обнаружения объектов.

    :param det_boxes: список тензоров, один тензор для каждого изображения, содержит предсказанные рамки
    :param det_labels: список тензоров, один тензор для каждого изображения, содержит предсказанные названия объектов
    :param det_scores: список тензоров, один тензор для каждого изображения, содержит предсказанные оценки классов объектов
    :param true_boxes: список тензоров, один тензор для каждого изображения, содержит истинные рамки
    :param true_labels: список тензоров, один тензор для каждого изображения, содержит истинные названия объектов
    :param true_difficulties: список тензоров, один тензор для каждого изображения, содержит истинные сложности объектов (0 или 1)
    :return: список средних значений точности для всех классов, среднее значение точности (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels) == len(true_difficulties) # все должно быть равно количеству изображений
    n_classes = len(LABEL_MAP)

    # Объединение и сохранение истинных данных в виде единых тензоров
    true_images = list() # список, в котором каждому объекту сопоставляется индекс изображения

    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))

    true_images = torch.LongTensor(true_images).to(DEVICE) # (n_objects), n_objects количесво всех объектов на всех изображениях
    true_boxes = torch.cat(true_boxes, dim=0) # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0) # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0) # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0) # один объект — один индекс, рамка, метка

    # Объединение и сохранение предсказанных данных в виде единых тензоров
    det_images = list()

    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))

    det_images = torch.LongTensor(det_images).to(DEVICE) # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0) # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0) # (n_detections)
    det_scores = torch.cat(det_scores, dim=0) # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0) # один объект — один индекс, рамка, метка

    # Расчет средней точности для всех классов кроме фона
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float) # (n_classes - 1)

    for c in range(1, n_classes): # цикл по всем классам кроме фона
        # Извлечение истинных данных только текущего класса
        true_class_images = true_images[true_labels == c] # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c] # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c] # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item() # количество легких объектов

        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(DEVICE) # создание тензора для хранения истинных объектов, кот были найдены (n_class_objects)

        # Извлечение предсказанных данных только текущего класса
        det_class_images = det_images[det_labels == c] # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c] # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c] # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)

        if n_class_detections == 0:
            continue

        # Сортировка предсказанных данных в порядке убывания оценки достоверности
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True) # (n_class_detections)
        det_class_images = det_class_images[sort_ind] # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind] # (n_class_detections, 4)

        # Проверка, является результат истинным или ложноположительным
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(DEVICE) # тензор бинарных меток для правильно найденных объектов (верный класс, достаточное перекрытие IoU) (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(DEVICE) # тензор бинарных меток для ошибочных объектов (нет соответствующего объекта или низкое перекрытие) (n_class_detections)
        
        for obj in range(n_class_detections): # цикл по предсказанным объектам
            this_detection_box = det_class_boxes[obj].unsqueeze(0) # (1, 4)
            this_image = det_class_images[obj] # () скаляр

            # Получение истинных объектов этого класса на текущем изображении
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)

            if object_boxes.size(0) == 0: # если такого объекта на этом изображении нет, то обнаружение является ложноположительным
                false_positives[obj] = 1
                continue

            # Найти максимальное перекрытие этой предсказанной рамки с объектами на этом изображении этого класса
            overlaps = find_jaccard_indx(this_detection_box, object_boxes) # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0) # (), () скаляры
            # ind это индекс объекта из object_boxes, object_difficulties

            # в true_class_boxes ind соответсвует объекту с этим индексом
            original_ind = torch.arange(true_class_boxes.size(0), dtype=torch.long, device=DEVICE)[true_class_images == this_image][ind] # нужен для обновления вектора true_class_boxes_detected

            if max_overlap.item() > 0.5: # если верно, то потенциальное совпадение

                if object_difficulties[ind] == 0: # сложные объекты игнорируются для честного сравнения моделей

                    if true_class_boxes_detected[original_ind] == 0: # если рамка еще не была отмечена
                        true_positives[obj] = 1 # помечаем как истинно позитивыный
                        true_class_boxes_detected[original_ind] = 1  # теперь этот объект учтен
                    else: # иначе это ложноположительный результат, поскольку этот объект уже учтен
                        false_positives[obj] = 1
    
            else: # иначе обнаружение происходит в местоположении, отличном от реального объекта, и является ложноположительным
                false_positives[obj] = 1

        # Кумулятивные TP и FP
        cumul_true_positives = torch.cumsum(true_positives, dim=0) # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0) # (n_class_detections)
        # Precision и Recall
        cumul_precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + 1e-10) # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects # (n_class_detections)

        # Реализация интерполированного усреднения точности (interpolated Average Precision) на 11 точках, что соответствует стандарту PASCAL VOC
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist() # 11 равномерных значений от 0.0 до 1.0 включительно (шаг 0.1) (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(DEVICE) # максимальная достигнутая точность (precision) при уровне полноты (11)

        for i, t in enumerate(recall_thresholds): # цикл по точкам
            recalls_above_t = cumul_recall >= t

            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max() # максимальная precision среди всех детекций с recall >= t
            else:
                precisions[i] = 0.

        # Среднее значение из 11 точек даёт AP для одного класса
        average_precisions[c - 1] = precisions.mean()

    # Подсчёт mAP
    mean_average_precision = average_precisions.mean().item() # среднее значение AP по всем классам, кроме фона
    average_precisions = {REV_LABEL_MAP[c + 1]: v for c, v in enumerate(average_precisions.tolist())} # преобразование в словарь

    return average_precisions, mean_average_precision


def evaluate(test_loader, model):
    """Тестирование модели"""

    model.eval() # перевод в режим предсказания

    # списки для предсказаных и истинных данных
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()

    with torch.no_grad(): # отключение расчета градиентов

        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc="Evaluating")): # цикл по батчам, tqdm добавляет прогресс-бар
            images = images.to(DEVICE) # на устройсво (N, 3, 300, 300)

            # Прямой проход
            predicted_locs, predicted_scores = model(images)

            # Расшифровка предсказаний
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs, predicted_scores,
                min_score=0.01, 
                max_overlap=0.45,
                top_k=200
            )

            # Пересылка на устройсво текущего батча
            boxes = [b.to(DEVICE) for b in boxes]
            labels = [l.to(DEVICE) for l in labels]
            difficulties = [d.to(DEVICE) for d in difficulties]

            # Сохранение данных
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Вычислениe Mean Average Precision (mAP) — основная метрика для задачи обнаружения объектов
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Вывод результатов
    pp = PrettyPrinter() # красивое и читаемое форматирование сложных структур при выводе в консоль
    pp.pprint(APs)
    print("\nMean Average Precision (mAP): %.3f" % mAP)


if __name__ == "__main__":
    # Загрузка модели
    add_safe_globals([ObjectDetector])
    model = ObjectDetector(n_classes = len(LABEL_MAP))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    print(f"\nЗагружена сжатая модель\n")

    model.eval() # переключение в режим тестирования

    # загрузка данных
    test_dataset = BbDataset(
        DATASET_FOLDER,
        split = SPLIT,
        keep_difficult = USE_DIFFICULT
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size = BATCH_SIZE, 
        shuffle = False,
        collate_fn = test_dataset.collate_fn, 
        num_workers = WORKERS_AMOUNT, 
        pin_memory = True
    )
    
    evaluate(test_loader, model)
