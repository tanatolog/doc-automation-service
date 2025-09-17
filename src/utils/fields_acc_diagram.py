"""Модуль для построения диаграммы точности распознавания полей"""
import matplotlib.pyplot as plt

# Данные
fields = {  'датавыдачи': 0.875,
            'датарождения': 0.8785, 
            'имя': 0.9655,
            'код': 0.9645,
            'месторождения': 0.8620,
            'номер': 0.9765,
            'отчество': 0.7886,     
            'пол': 0.9533,
            'серия': 1.0,        
            'фамилия': 0.9065
 }  

# Преобразуем значения в проценты
labels = list(fields.keys())
values = [v * 100 for v in fields.values()]

# Построение диаграммы
plt.figure(figsize=(12, 6))
bars = plt.bar(labels, values, color='skyblue')

# Подписи над столбцами
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{value:.1f}%', ha='center', va='bottom')

plt.ylim(0, 110)
plt.ylabel('Точность распознавания (%)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()