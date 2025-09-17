"""Модуль для описания класса базы данных документов"""
import sqlcipher3

DATABASE_PATH = "src/database/document_base.db"
DB_PASSWORD_PATH = "src/database/db_key.txt"

class DocumentBase:
    """Класс базы данных документов"""

    def __init__(self, database_path: str, password: str):
        """Конструктор. Класс базы данных документов."""
        self.database_path = database_path

        self.password = password

        self.create_passport_table()
        print(f"\nУстановлено соединение с базой {database_path}\n")

    
    def __execute_sql_command(self, command: str, parameters: tuple = None):
        """Выполнение sql комманды (с параметрами или без)"""
        # соединение с базой sqlcipher (или создание, если не сущ), потом закрытие соединения
        with sqlcipher3.connect(self.database_path) as connection:
            cursor = connection.cursor() # создание курсора
            cursor.execute(f"PRAGMA key = '{self.password}';") # ввод пароля

            # выполнение SQL команды
            if parameters:
                cursor.execute(command, parameters)
            else:
                cursor.execute(command)

            connection.commit() # принять изменения


    def __fetchall_sql_query(self, query: str) -> list[tuple]:
        """Выполнение SELECT-запроса и возврат всех строк результата"""
        # соединение с базой sqlcipher (или создание, если не сущ), потом закрытие соединения
        with sqlcipher3.connect(self.database_path) as connection:
            cursor = connection.cursor()
            cursor.execute(f"PRAGMA key = '{self.password}';") # ввод пароля
            cursor.execute(query)
            return cursor.fetchall()


    def create_passport_table(self):
        """Создание таблицы паспортов"""

        # SQL команда для создания таблицы
        create_passport_comm = """
        CREATE TABLE IF NOT EXISTS Passport (
            id INTEGER PRIMARY KEY,
            kod TEXT,
            issuance TEXT,
            surname TEXT,
            name TEXT,
            midname TEXT,
            sex TEXT,
            birthdate TEXT,
            birthplace TEXT,
            serie TEXT,
            number TEXT,
            added TEXT,
            filename TEXT
        );
        """
        self.__execute_sql_command(create_passport_comm)


    def _delete_table(self, table_name: str):
        """Удаление таблицы"""

        if not table_name.isidentifier(): # безопасное форматирование названия таблицы
            raise ValueError("Недопустимое имя таблицы")
        
        self.__execute_sql_command(f"DROP TABLE IF EXISTS {table_name};")
        print(f"Таблица {table_name} удалена")

    
    def insert_passport(self, passport_data: dict):
        """
        Добавление записи в таблицу Passport.

        :param passport_data: словарь с ключами: kod, issuance, surname, name, midname, sex, birthdate, birthplace, serie, number, added, filename
        """

        required_keys = ["kod", "issuance", "surname", "name", "midname", "sex", "birthdate", "birthplace", "serie", "number", "added", "filename"]

        # Проверка наличия всех обязательных полей
        for key in required_keys:
            if key not in passport_data:
                raise ValueError(f"Отсутствует обязательное поле: {key}")

        insert_query = f"""
        INSERT INTO Passport (kod, issuance, surname, name, midname, sex, birthdate, birthplace, serie, number, added, filename) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        values = tuple(passport_data[key] for key in required_keys)

        self.__execute_sql_command(insert_query, values)


    def output_table(self, table_name: str) -> list[tuple]:
        """Возвращение всех строк таблицы по имени"""

        if not table_name.isidentifier():
            raise ValueError("Недопустимое имя таблицы")

        query = f"SELECT * FROM {table_name};"
        return self.__fetchall_sql_query(query)
    

    def output_passports(self):
        """Возвращение информации о паспортах из таблицы Passport"""
        query = """
        SELECT id, surname, name, midname, sex, birthdate, birthplace, serie, number, kod, issuance, filename
        FROM Passport;
        """
        return self.__fetchall_sql_query(query)
    
    def update_passport_field(self, passport_id: int, field_name: str, new_value: str):
        """Обновление одного поля записи паспорта по ID"""
        field_map = { # Преобразование имени столбца из пользовательского в SQL-имя
            "фамилия": "surname",
            "имя": "name",
            "отчество": "midname",
            "пол": "sex",
            "дата рождения": "birthdate",
            "место рождения": "birthplace",
            "серия": "serie",
            "номер": "number",
            "код": "kod",
            "дата выдачи": "issuance",
        }

        if field_name not in field_map:
            raise ValueError(f"Поле {field_name} не может быть обновлено.")

        column_name = field_map[field_name]

        command = f"UPDATE Passport SET {column_name} = ? WHERE id = ?"
        self.__execute_sql_command(command, (new_value, passport_id))
    
        
if __name__ == "__main__":

    doc_base = DocumentBase(DATABASE_PATH, DB_PASSWORD_PATH)

    sample_passport = {
        "kod": "123-456",
        "issuance": "12.12.2000",
        "surname": "Иванов",
        "name": "Иван",
        "midname": "Иванович",
        "sex": "МУЖ",
        "birthdate": "55.12.1900",
        "birthplace": "ГМОСКВА",
        "serie": "1234",
        "number": "567890",
        "added": "02.06.2025 18:40",
        "filename": "ivanov_passport.jpg"
    }

    doc_base.insert_passport(sample_passport)
    doc_base.insert_passport(sample_passport)
    doc_base.insert_passport(sample_passport)

    rows = doc_base.output_table("Passport")
    print(rows)

    doc_base._delete_table("Passport")