"""Модуль для проверки доступности CUDA"""
import torch
print("CUDA доступен:", torch.cuda.is_available())
print("Имя устройства:", torch.cuda.get_device_name(0))
print("Текущее устройство:", torch.cuda.current_device())

#nvidia-smi