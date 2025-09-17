"""Модуль таблиц для красивого вывода информации"""
from rich.table import Table

def general_table():
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Epoch", style="aquamarine1", width=12)
    table.add_column("Train Loss", style="bright_green")
    table.add_column("Test Loss", style="bright_green")
    table.add_column("Accuracy", style="bright_yellow")
    table.add_column("Best Accuracy", style="gold1")
    table.columns[0].header_style = "aquamarine1"
    table.columns[1].header_style = "bright_green"
    table.columns[2].header_style = "bright_green"
    table.columns[3].header_style = "bright_yellow"
    table.columns[4].header_style = "bright_yellow"
    return table


def predictions_table():
    table = Table(show_header=True, header_style="hot_pink")
    table.add_column("Ground Truth", width=12)
    table.add_column("Predicted")
    table.border_style = "bright_yellow"
    table.columns[0].style = "violet"
    table.columns[1].style = "grey93"
    return table
