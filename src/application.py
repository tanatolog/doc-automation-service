"""Модуль в котором располагается приложение"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import string
from PIL import Image
from datetime import datetime
import win32file # type: ignore
import win32api  # type: ignore
import hashlib
import subprocess

from utils.constants import DEVICE, LABEL_COLOR_MAP, REV_LABEL_MAP, CLASSES, OCR_DIMS, OCR_DROPOUT, OCR_GRU_LAYERS, ANGLE_DROPOUT, ANGLE_IMAGE_SIZE
from fields_detector.documents_fields_detector import DocumentFieldsDetector
from ocr.text_detector import TextDetector
from utils.line_splitter import LineSplitter 
from database.document_base import DocumentBase
from angle_regression.angle_detector import AngleDetector

# Пути к моделям
FIELD_MODEL_PATH = "models/fields_det_11_112_compressed.pth"
OCR_MODEL_PATH = "models/ocr_3.pthcheckpoint-91.26.pth"
ANGLE_REG_MODEL_PATH = "models/angle_det_4.pthcheckpoint-86.84.pth"

# База данных и ключ-флешка
DATABASE_PATH = "src/database/document_base.db"
KEY_FILENAME = "db_key.txt"

# параметры распознавания
MIN_SCORE = 0.25 # минимальный порог для того, чтобы предсказанная рамка считалась соответствующей определенному классу
MAX_OVERLAP = 0.25 # максимальное перекрытие, которое могут иметь две рамки, чтобы рамка с более низким баллом не удалялась с помощью Non-Maximum Suppression (NMS)
TOP = 200

# параметры приложения
USE_ROTATE = True #True # использовать предсказание угла поворота
SAFE_FIELDS = False # сохранять обрезанные поля

class Form:
    """Класс, описывающий работу приложения"""

    def __init__(self, master):
        """Конструктор. Класс, описывающий работу приложения"""
        try: # считывание пароля к базе данных
            db_password = self.find_usb_key(key_filename = KEY_FILENAME)
        except FileNotFoundError as e:
            tk.Tk().withdraw()  # скрытие основного окна
            messagebox.showerror("Ошибка безопасности", f"{e}\n\nПриложение завершено.")
            sys.exit(1)


        self.master = master
        master.title("Система выделения и распознавания полей паспорта РФ")
        master.geometry("1310x700")
        master.resizable(False, False)

        self.angle_detector = AngleDetector( # детектор угла поворота
            model_path = ANGLE_REG_MODEL_PATH,
            angle_dropout = ANGLE_DROPOUT,
            image_size = ANGLE_IMAGE_SIZE,
            device = DEVICE
        )

        self.field_detector = DocumentFieldsDetector( # детектор полей документа
            model_path = FIELD_MODEL_PATH,
            device = DEVICE,
            color_map = LABEL_COLOR_MAP,
            rev_label_map = REV_LABEL_MAP
        )

        self.text_detector = TextDetector( # детектор текста - OCR
            model_path = OCR_MODEL_PATH,
            classes = CLASSES,
            ocr_dims = OCR_DIMS,
            ocr_dropout = OCR_DROPOUT,
            ocr_gru_layers = OCR_GRU_LAYERS,
            device = DEVICE
        )

        self.db = DocumentBase(DATABASE_PATH, db_password) # база данных

        self.line_splitter = LineSplitter(
            threshold_1 = 20,
            threshold_2 = 30
        )

        self.style = ttk.Style()
        self.style.configure("TLabel", font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10))
        self.style.configure("TEntry", font=("Segoe UI", 10))

        # Основной контейнер
        self.frame = ttk.Frame(master, padding="15 15 15 15")
        self.frame.pack(fill="both", expand=True)

        # Группа: Выбор пути
        path_frame = ttk.LabelFrame(self.frame, text="Путь к изображениям", padding="10")
        path_frame.pack(fill="x", pady=5)
        ttk.Button(path_frame, text="Выбрать", command=self.select_folder).pack(side="left", padx=(0, 10))

        self.entry_path = ttk.Entry(path_frame, width=50)
        self.entry_path.pack(side="left", fill="x", expand=True)

        # Группа: Вывод информации
        output_frame = ttk.LabelFrame(self.frame, text="Результаты обработки", padding="10")
        output_frame.pack(fill="x", pady=5)

        self.label_status = ttk.Label(output_frame, text="Статус: Готовность к работе")
        self.label_status.pack(anchor="w", pady=2)
        self.label_found = ttk.Label(output_frame, text="Найдено изображений: —")
        self.label_found.pack(anchor="w", pady=2)
        self.label_recognized = ttk.Label(output_frame, text="Распознано полей: —")
        self.label_recognized.pack(anchor="w", pady=2)

        # Контейнер для кнопки обработки
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill="x", pady=(5, 0))
        ttk.Button(button_frame, text="Обработать изображения", command=self.process_images).pack(anchor="w", padx=10)

        # Контейнер для таблиц базы данных
        table_frame = ttk.LabelFrame(self.frame, text="Содержимое базы данных", padding="10")
        table_frame.pack(fill="both", expand=True, pady=10)

        # Контейнер для таблицы паспартов и скролбара
        self.columns = ["№", "фамилия", "имя", "отчество", "пол", "дата рождения", "место рождения", "серия", "номер", "код", "дата выдачи", "имя файла"]

        column_widths = {
            "№": 15,
            "фамилия": 120,
            "имя": 120,
            "отчество": 120,
            "пол": 20,
            "дата рождения": 70,
            "место рождения": 300,
            "серия": 30,
            "номер": 30,
            "код": 30,
            "дата выдачи": 50,
            "имя файла": 120,
        }

        tree_container = ttk.Frame(table_frame)
        tree_container.pack(fill="both", expand=True)

        # Скролбар
        tree_scrollbar = ttk.Scrollbar(tree_container, orient="vertical")
        tree_scrollbar.pack(side="right", fill="y")

        # Таблица паспартов
        self.tree = ttk.Treeview(
            tree_container,
            columns = self.columns,
            show = "headings",
            height = 5,
            yscrollcommand=tree_scrollbar.set
        )
        tree_scrollbar.config(command=self.tree.yview)

        for col in self.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths[col], anchor="center")

        self.tree.pack(side="left", fill="both", expand=True)

        self.refresh_table()

        # Привязка двойного клика по ячейке таблицы
        self.tree.bind("<Double-1>", self.edit_cell)


    def get_flash_serial(self, letter):
        """Возвращает серийный номер флешки"""
        if win32file.GetDriveType(letter.upper() + ":\\") == win32file.DRIVE_REMOVABLE:  # Тип означает съемный диск (флешка)
            serial = win32api.GetVolumeInformation(letter + ":\\")[1]                    # Серийный номер тома
            return serial
        

    def get_machine_uuid(self):
        try:
            output = subprocess.check_output(
                ["wmic", "csproduct", "get", "UUID"],
                text=True
            ).splitlines()
            return output[1].strip()
        except:
            return ""


    def find_usb_key(self, key_filename: str):
        """
        Ищет USB-устройство и возвращает хешированный ключ.

        :param key_filename: имя файла с паролем
        :return: SHA-256 хеш ключа
        :raises FileNotFoundError: если ключ не найден
        """
        for letter in string.ascii_uppercase:
            drive = f"{letter}:/"

            if os.path.exists(drive):
                full_path = os.path.join(drive, key_filename)

                if os.path.isfile(full_path):

                    with open(full_path, "r", encoding="utf-8") as f:
                        password = f.read().strip()

                    machine_is = self.get_machine_uuid()
                    flash_id = self.get_flash_serial(letter)

                    key_material = f"{password}{machine_is}{flash_id}"
                    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()

        raise FileNotFoundError("USB-ключ с паролем не найден.")


    def edit_cell(self, event):
        """Редактирование ячейки таблицы"""
        region = self.tree.identify("region", event.x, event.y)

        if region != "cell":
            return

        row_id = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)

        x, y, width, height = self.tree.bbox(row_id, column)
        column_index = int(column.replace("#", "")) - 1

        non_editable_columns = {"№", "имя файла"}
        if self.columns[column_index] in non_editable_columns: # Нельзя редактировать
            return

        # Получение текущего значения
        item = self.tree.item(row_id)
        value = item["values"][column_index]

        # Entry поверх ячейки
        entry = tk.Entry(self.tree)
        entry.place(x=x, y=y, width=width, height=height)
        entry.insert(0, value)
        entry.focus()

        def save_edit(event):
            new_value = entry.get()
            entry.destroy()

            item_values = list(item["values"])
            item_values[column_index] = new_value
            self.tree.item(row_id, values=item_values)

            # обновление записи в базе данных
            try:
                passport_id = item_values[0]  # поле "№"
                field_name = self.columns[column_index]
                new_value = item_values[column_index]
                self.db.update_passport_field(passport_id, field_name, new_value)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось обновить базу данных:\n{e}")

        entry.bind("<Return>", save_edit)
        entry.bind("<FocusOut>", lambda e: entry.destroy())


    def refresh_table(self):
        """Обновление таблицы в приложении из базы данных"""
        for row in self.tree.get_children():
            self.tree.delete(row)

        for row in self.db.output_passports():
            self.tree.insert("", "end", values=row)


    def select_folder(self):
        """Метод для выбора директории"""
        folder = filedialog.askdirectory()
        if folder:
            self.entry_path.delete(0, tk.END)
            self.entry_path.insert(0, folder)


    def process_field(self, label: str, cropped: Image.Image, crop_folder: str, filename_prefix: str, index: int):
        """
        Обработка конкретного текстового поля по метке.
        Поворот, сегментация, OCR.
        
        :param label: метка поля
        :param cropped: обрезанное изображение поля
        :return: словарь (название класса, распознанный текст)
        """
        texts = {}

        if SAFE_FIELDS == True: # сохранение строки
            os.makedirs(crop_folder, exist_ok=True) # папка для сохранения полей

        if label in ["серия", "номер"]:
            cropped = cropped.rotate(90, expand=True)

        elif label == "месторождения":

            # Разделение на строки
            lines = self.line_splitter.combo_split_into_lines(cropped)

            for i, line in enumerate(lines):

                if SAFE_FIELDS == True: # сохранение строки
                    line_filename = f"{filename_prefix}_{index}_{label}_line_{i}.png"
                    line_save_path = os.path.join(crop_folder, line_filename)
                    line.save(line_save_path)

                line_text = self.text_detector.predict_from_image(line)

                if "месторождения" not in texts:
                    texts["месторождения"] = line_text
                else:
                    texts["месторождения"] = texts["месторождения"] + " " + line_text

            return texts

        # Обычное поле
        text = self.text_detector.predict_from_image(cropped)

        texts[label] = text

        if SAFE_FIELDS == True: # сохранение строки# Сохраняем изображение поля
            crop_filename = f"{filename_prefix}_{index}_{label}.png"
            crop_path = os.path.join(crop_folder, crop_filename)
            cropped.save(crop_path)

        return texts
    

    def error_handling(self, texts: dict[str, str]):
        """Исправление ошибок распознавания текста"""
        required_keys = ["серия", "номер", "код", "датавыдачи", "фамилия", "имя", "отчество", "пол", "датарождения", "месторождения"]

        # Добавление недостающих ключей
        for key in required_keys:
            if key not in texts:
                texts[key] = ""

        # Коррекция значений
        if ("Е" in texts["пол"]) or ("Н" in texts["пол"]) or (texts["пол"] == ""):
            texts["пол"] = "ЖЕН"
        else:
            texts["пол"] = "МУЖ"

        location = texts["месторождения"]
        substrings = ["Г.", "ГОР.", "ГОРОД.", "С.", "П.", "ПОС."]
        for sub in substrings:
            index = location.find(sub)
            if index != -1:
                texts["месторождения"] = texts["месторождения"][index:].strip()
                break

        if texts["датавыдачи"] != "":
            datavydachi = texts["датавыдачи"]
            dot_index = datavydachi.find(".")
            if dot_index == 1:
                texts["датавыдачи"] = "2" + datavydachi

            # Обрезка лишнего после второй точки в дате выдачи
            if texts["датавыдачи"].count(".") >= 2:
                parts = texts["датавыдачи"].split(".")
                if len(parts[2]) > 4:
                    parts[2] = parts[2][:4]
                    texts["датавыдачи"] = ".".join(parts)

        if texts["датарождения"] != "":
            birth_data = texts["датарождения"]
            dot_index = birth_data.find(".")
            if dot_index == 1:
                texts["датарождения"] = "1" + birth_data

            # Обрезка лишнего после второй точки в дате выдачи
            if texts["датарождения"].count(".") >= 2:
                parts = texts["датарождения"].split(".")
                if len(parts[2]) > 4:
                    parts[2] = parts[2][:4]
                    texts["датарождения"] = ".".join(parts)
        

        return texts


    def add_passport_in_base(self, texts: dict[str, str], filename: str):
        """
        Добавление данных о паспорте в базу данных и исправление ошибок распознавания

        :param texts: словарь (название класса, текст)
        :param filename: имя изображения с паспартом
        """

        passport_data = {}

        passport_data["serie"] = texts["серия"]
        passport_data["number"] = texts["номер"]
        passport_data["kod"] = texts["код"]
        passport_data["issuance"] = texts["датавыдачи"]
        passport_data["surname"] = texts["фамилия"]
        passport_data["name"] = texts["имя"]
        passport_data["midname"] = texts["отчество"]
        passport_data["sex"] = texts["пол"]
        passport_data["birthdate"] = texts["датарождения"]
        passport_data["birthplace"] = texts["месторождения"]

        passport_data["added"] = datetime.now()
        passport_data["filename"] = filename

        try:
            self.db.insert_passport(passport_data)
        except ValueError as e:
            print(f"Ошибка вставки: {e}")


    def process_images(self):
        """Обработать изображения"""
        folder = self.entry_path.get()

        if not os.path.isdir(folder):
            messagebox.showerror("Ошибка", "Указанная папка не существует.")
            return

        self.label_status.config(text="Статус: Идет обработка изображений...")
        self.master.update_idletasks()  # Обновление интерфейса

        # Создание выходной папки рядом с исходной
        output_folder = os.path.join(os.path.dirname(folder), os.path.basename(folder) + "_processed")
        os.makedirs(output_folder, exist_ok=True)

        crop_folder = os.path.join(os.path.dirname(folder), os.path.basename(folder) + "_croped")

        if SAFE_FIELDS == True:
            os.makedirs(crop_folder, exist_ok=True)

        image_count = 0
        recognized_fields = 0

        for filename in os.listdir(folder): # цикл по изображениям

            if filename.lower().endswith(".jpg"):
                image_count += 1
                image_path = os.path.join(folder, filename)
                original = Image.open(image_path).convert("RGB")

                # Выравнивание
                if USE_ROTATE == True:
                    angle = self.angle_detector.predict_from_image(original)[0]

                    if abs(angle) > 10:
                        original = original.rotate(-angle, expand = True)
                #print(f"{filename}: угол = {angle:.2f}°")

                # Детекция полей
                boxes, labels = self.field_detector.detect_objects_on_image(original, min_score=MIN_SCORE, max_overlap=MAX_OVERLAP, top_k=TOP)

                if not labels: # если не нашли ни одного объекта
                    continue

                recognized_fields += len(labels)

                result_img = self.field_detector.draw_boxes_on_image(original, boxes, labels)

                # Сохранение в выходную папку
                save_path = os.path.join(output_folder, filename)
                result_img.save(save_path)

                # OCR по каждому полю
                texts = {}
                for idx, (label, box) in enumerate(zip(labels, boxes)):
                    left, top, right, bottom = map(int, box)
                    cropped = original.crop((left, top, right, bottom))

                    field_texts = self.process_field(label, cropped, crop_folder, os.path.splitext(filename)[0], idx)

                    texts.update(field_texts)

                texts = self.error_handling(texts) # обработка ошибок

                self.add_passport_in_base(texts, filename)

        self.refresh_table()
        self.label_found.config(text=f"Найдено изображений: {image_count}")
        self.label_recognized.config(text=f"Распознано полей: {recognized_fields}")
        self.label_status.config(text="Статус: Готовность к работе")
        messagebox.showinfo("Обработка завершена", "Обработка завершена")


if __name__ == "__main__":
    root = tk.Tk()
    app = Form(root)
    root.mainloop()