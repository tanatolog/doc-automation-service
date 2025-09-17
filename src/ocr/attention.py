import torch
from torch import nn


class Attention(nn.Module):
    """Класс реализующий механизм аддитивного внимания Bahdanau attention"""

    def __init__(self, dims):
        """
        Конструктор. Класс реализующий механизм аддитивного внимания Bahdanau attention.

        :param dims: размер линейных слоев
        """
        super().__init__()

        self.linear_in = nn.Linear(in_features=dims, out_features=dims, bias=False) # линейный слой, 256 -> 256
        self.linear_out = nn.Linear(in_features=dims * 2, out_features=dims, bias=False) # линейный слой, 512 -> 256

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Прямой проход.
        
        :param query: последовательность запросов (выход GRU на каждом временном шаге) (batch size, output length, dimensions)
        :param context: Данные к которым применяется механизм распознавания, контекстом является выход CNN по изображению (batch size, query length, dimensions)

        :return output: контекстуально-усиленные признаки (batch size, output length, dimensions)
        :return weights: веса внимания (можно визуализировать) (batch size, output length, query length)
        """

        batch_size, output_len, dims = query.size() # batch_size, 23, 256
        query_len = context.size(1) # 23

        query = query.reshape(batch_size * output_len, dims) # (batch_size * 23, 256)
        query = self.linear_in(query) # (batch_size * 23, 256)
        query = query.reshape(batch_size, output_len, dims) # (batch_size, 23, 256)

        # батчевое умножение матриц, показывает, насколько каждый выходной токен должен внимательно смотреть на каждый входной признак
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous()) # (batch_size, 23, 256) * (batch_size, 256, 23) = (batch_size, 23, 23)

        # Преобразование attention_scores в вероятности, сумма = 1
        attention_scores = attention_scores.view(batch_size * output_len, query_len) # (batch_size * 23, 23)
        attention_weights = self.softmax(attention_scores) # (batch_size * 23, 23)
        attention_weights = attention_weights.view(batch_size, output_len, query_len) # (batch_size, 23, 23)

        # итоговое внимание, где каждый выходной токен имеет взвешенную сумму признаков изображения
        mix = torch.bmm(attention_weights, context) # (batch_size, 23, 256)

        # Склеиваем query и mix
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dims) # (batch_size * 23, 512)

        output = self.linear_out(combined) # (batch_size * 23, 256)
        output = output.view(batch_size, output_len, dims) # (batch_size, 23, 256)
        output = self.tanh(output) # гиперболический тангенс

        return output, attention_weights # (batch_size, 23, 256) (batch_size, 23, 256)
