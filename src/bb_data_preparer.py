"""Модуль для формирования обучающей и тестовой выборок из датасета bounding boxes"""
import os
import json
import xml.etree.ElementTree as ElementTree
import re

from utils.constants import LABEL_MAP

BB_DATASET_PATH = "data/fields_dataset/"
OUTPUT_FOLDER = "data/fields_dataset/"

class BbDataPreparer:
    """Класс для подготовки обучающей и тестовой выборок из датасета bounding boxes"""

    def __init__(self):
        """Конструктор. Класс для подготовки обучающей и тестовой выборок из датасета bounding boxes"""
        print("\nВыполняется подготовка датасета bounding boxes")


    def parse_annotation(self, annotation_path: str):
        """Извлечение информации об объектах на изображении из аннотации VOC"""

        tree = ElementTree.parse(annotation_path) # чтение xml, возврат дерева xml
        root = tree.getroot() # возврат корневого эл дерева (annotation)

        boxes = list() # список рамок
        labels = list() # список меток классов объектов
        difficulties = list() # список сложностей объектов

        for object in root.iter("object"): # цикл по всем объектам в дереве
            difficult = int(object.find("difficult").text == "1") # поиск difficult, преобразование его в bool, а потом в int

            label = object.find("name").text.lower().strip()
            if label not in LABEL_MAP:
                continue

            bndbox = object.find("bndbox")
            xmin = int(bndbox.find("xmin").text) - 1 # координаты рамки
            ymin = int(bndbox.find("ymin").text) - 1 # вычитание 1 — приведение к индексации с нуля (VOC использует 1-based)
            xmax = int(bndbox.find("xmax").text) - 1
            ymax = int(bndbox.find("ymax").text) - 1

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(LABEL_MAP[label]) # сохранение метки класса
            difficulties.append(difficult)

        return {"boxes": boxes, "labels": labels, "difficulties": difficulties} # возврат словаря
    

    def remove_duplicates_from_train(self, dataset_path: str):
        """Удаляет из папки train_annotations все XML-файлы, которые уже есть в test_annotations"""

        test_files = set(
            f for f in os.listdir(os.path.join(dataset_path, "test_annotations"))
            if f.endswith(".xml")
        )

        removed_count = 0
        for f in os.listdir(os.path.join(dataset_path, "train_annotations")):
            if f.endswith(".xml") and f in test_files:
                os.remove(os.path.join(os.path.join(dataset_path, "train_annotations"), f))
                removed_count += 1
                print(f"[+] Удалён файл: {f}")

        print(f"\nВсего удалено дубликатов: {removed_count}")


    def update_difficult_flag(self, voc_annotation: str):
        """
        Обновляет флаг <difficult> в аннотации VOC: заменяет 0 на 1.
        
        :param voc_annotation: путь к XML-файлу аннотации
        """
        if not voc_annotation.endswith(".xml"):
            return

        tree = ElementTree.parse(voc_annotation)
        root = tree.getroot()

        updated = False
        for obj in root.findall("object"):
            difficult_tag = obj.find("difficult")
            if difficult_tag is not None and difficult_tag.text == "1":
                difficult_tag.text = "0"
                updated = True

        if updated:
            tree.write(voc_annotation, encoding="utf-8", xml_declaration=True)


    def annotations_update_difficult_flag(self, directory: str):
        """
        Обход всех XML-файлов в директории и применение update_difficult_flag к файлам, чьи имена не соответствуют шаблону "число"_"число".xml.
        
        :param directory: путь к директории с аннотациями
        """
        print("Установка сложных объектов в файлах с именем 'число'_'число'.xml")
        pattern = re.compile(r"^\d+_\d+\.xml$")

        for filename in os.listdir(directory):
            if not filename.endswith(".xml"):
                continue
            if pattern.match(filename):
                continue  # Пропустить "число_число.xml"
            
            file_path = os.path.join(directory, filename)
            self.update_difficult_flag(file_path)



    def create_data_lists(self, dataset_path, output_folder):
        """Создание списков изображений, ограничивающих рамок и названий объектов на этих изображениях и сохранение их в файл"""

        dataset_path = os.path.abspath(dataset_path) # преобразование относительных путей в абсолютные гарантирует, что пути не зависят от текущей рабочей директории

        # обучающая выборка
        train_images = list() # список путей к изображениям обучающей выборки
        train_objects = list() # список списков объектов на каджом изображении
        n_objects = 0 # счетчик колличества объектов

        annotations_path = os.path.join(dataset_path, "train_annotations")
        #self.annotations_update_difficult_flag(annotations_path) # устанавливаем флаг difficult всем сложным объектам

        for annotation in os.listdir(annotations_path): # цикл по аннотациям обучающей выборки
            annotation_path = os.path.join(annotations_path, annotation)
            objects = self.parse_annotation(annotation_path)# получение объектов на изображении

            if len(objects["boxes"]) == 0:
                continue

            name = os.path.splitext(os.path.basename(annotation))[0]
            image_path = os.path.join(dataset_path, "images", name + ".jpg")

            if not os.path.exists(image_path):
                print(f"[!] Изображение не найдено для аннотации: {annotation}")
                continue

            n_objects += len(objects["boxes"])
            train_objects.append(objects)
            train_images.append(os.path.join(dataset_path, "images", name + ".jpg"))

        assert len(train_objects) == len(train_images) # проверка, размер списков должен быть одинаков

        # сохранение выборки в файл json
        with open(os.path.join(output_folder, "TRAIN_images.json"), "w") as j:
            json.dump(train_images, j)
        with open(os.path.join(output_folder, "TRAIN_objects.json"), "w") as j:
            json.dump(train_objects, j)
        with open(os.path.join(output_folder, "label_map.json"), "w") as j:
            json.dump(LABEL_MAP, j)

        print(f"\nОбучающая выборка подготовлена: {os.path.abspath(output_folder)}\n" +
            f"Всего изображений: {len(train_images)}\n" +
            f"Всего объектов: {n_objects}")

        # тестовая выборка
        test_images = list() # список путей к изображениям обучающей выборки
        test_objects = list() # список списков объектов на каджом изображении
        n_objects = 0 # счетчик колличества объектов

        annotations_path = os.path.join(dataset_path, "test_annotations")
        #self.annotations_update_difficult_flag(annotations_path) # устанавливаем флаг difficult всем сложным объектам

        for annotation in os.listdir(annotations_path): # цикл по аннотациям изображений тестовой выборки
            annotation_path = os.path.join(annotations_path, annotation)
            objects = self.parse_annotation(annotation_path) # получение объектов на изображении

            if len(objects) == 0:
                continue

            name = os.path.splitext(os.path.basename(annotation))[0]
            image_path = os.path.join(dataset_path, "images", name + ".jpg")

            if not os.path.exists(image_path):
                print(f"[!] Изображение не найдено для аннотации: {annotation}")
                continue

            test_objects.append(objects)
            n_objects += len(objects["boxes"])
            test_images.append(os.path.join(dataset_path, "images", name + ".jpg"))

        assert len(test_objects) == len(test_images) # проверка, размер списков должен быть одинаков

        # сохранение выборки в файл json
        with open(os.path.join(output_folder, "TEST_images.json"), "w") as j:
            json.dump(test_images, j)
        with open(os.path.join(output_folder, "TEST_objects.json"), "w") as j:
            json.dump(test_objects, j)

        print(f"\nТестовая выборка подготовлена: {os.path.abspath(output_folder)}\n" +
            f"Всего изображений: {len(test_images)}\n" +
            f"Всего объектов: {n_objects}")


"""Запуск функции формирования выборок из датасета bounding boxes"""
if __name__ == "__main__":
    preparer = BbDataPreparer()

    #preparer.remove_duplicates_from_train(BB_DATASET_PATH)

    preparer.create_data_lists(
        dataset_path = BB_DATASET_PATH, 
        output_folder = OUTPUT_FOLDER
    )