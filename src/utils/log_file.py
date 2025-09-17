"""Модуль класса логгирования"""
import os
from rich.console import Console
from rich.table import Table

class LogFile:
    """Класс файла логгирования"""
    def __init__(self, file_path: str):
        """Конструктор. Класс файла логгирования. Файл пересоздается!"""
        self.path = file_path

        if os.path.exists(file_path): # удаление файла, если он существует
            os.remove(file_path)

    def open(self):
        """Открыть файл"""
        self.file = open(self.path, "a", encoding="utf-8")  # режим добавления

    def write(self, message: str):
        """Записать сообщение в файл"""
        self.file.write(message + "\n")

    def write_table(self, table: Table):
        """Записать таблицу rich в файл без ANSI-кодов"""
        temp_console = Console(record=True)
        temp_console.print(table)
        text = temp_console.export_text(clear=True)
        self.file.write(text + "\n")

    def close(self):
        """Закрыть файл"""
        if not self.file.closed:
            self.file.close()