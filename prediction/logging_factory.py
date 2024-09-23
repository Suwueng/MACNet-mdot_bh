import json
import sys
import logging
import warnings
from torchinfo import summary


class Logger:
    def __init__(self, name=None, log_file='app.log', level=logging.DEBUG):
        """
        初始化日志记录器。

        参数:
        - name (str): 记录器的名称（通常为模块名）。默认为None，使用root logger。
        - log_file (str): 日志文件路径。
        - level (int): 日志记录级别，默认为DEBUG。
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # 创建文件处理器
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(level)

    def get_logger(self):
        """获取配置好的 logger 对象。"""
        return self.logger


class LoggerDebug(Logger):
    def __init__(self, name=__name__, log_file='debug.log', level=logging.DEBUG):
        super().__init__(name, log_file, level)

        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)


class LoggerTraining(Logger):
    def __init__(self, name='TrainingLogger', log_file='training.log', level=logging.INFO):
        super().__init__(name, log_file, level)

        # 创建控制台处理器
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(level)

        # 设置日志格式
        formatter_file_handler = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(formatter_file_handler)
        formatter_console_handler = logging.Formatter('%(message)s')
        self.console_handler.setFormatter(formatter_console_handler)

        # 添加处理器到 logger
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)


def log_print(logger=None, message=None):
    if message is None:
        return
    elif logger is not None:
        logger.info(message)
    else:
        print(message)


def log_summary(logger=None, model=None, input_size=None, input_data=None, device=None, verbose=0):
    summary_message = f'The summary of {model.__class__.__name__} network\n'
    summary_message += str(summary(model, input_size=input_size, input_data=input_data, device=device, verbose=verbose))

    log_print(logger, summary_message)


def log_loss(log_data, save_path):
    with open(save_path, 'w') as f:
        json.dump(log_data, f)


class TableFormat:
    def __init__(self, width: int = 90, line: str = '=', alignment: str = '<'):
        self._table = None
        self._header = None
        self._header_str = None
        self._data = None
        self._data_str = None

        self._column_width = None

        self._width = width
        self._alignment = alignment
        self._line = line * (self._width // len(line)) + line[:(self._width % len(line))] + '\n'

    def get_table(self):
        return self._table

    def _adjust_column_width(self):
        grid_width = [[len(col) for col in self._header_str]]
        for i, row in enumerate(self._data_str):
            if len(row) != len(self._header_str):
                raise ValueError("The number of columns in the header and the data does match.")
            grid_width.append([len(col) for col in self._data_str[i]])

        column_num = len(self._header_str)
        row_num = len(self._data_str) + 1
        self._column_width = [max([grid_width[i][j] for i in range(row_num)]) for j in range(column_num)]

    def _adjust_width(self, width):
        if width is not None:
            self._width = width

        self._adjust_column_width()

        max_width = sum(self._column_width) + 2 * (len(self._header_str) - 1)
        if self._width < max_width:
            warnings.warn(f'The current width {self._width} is insufficient, and it should be at least {max_width}. '
                          f'Please increase the width or shorten the content length.')
            self._width = max_width

        extra_width = self._width - max_width
        self._column_width = [i + extra_width // len(self._header_str) for i in self._column_width]
        for i in range(extra_width % len(self._header_str)):
            self._column_width[i] += 1

    def generate(self, header: list = None, data: list = None,
                 width: int = None, line: str = None, alignment: str = '<'):

        self._header = header
        self._header_str = [str(col) for col in self._header]
        self._data = data
        self._data_str = [[str(col) for col in self._data[row]] for row in range(len(self._data))]

        # Format setting
        self._alignment = alignment     # Alignment
        self._adjust_width(width)       # Width
        if line is not None:            # Line style
            self._line = line * (self._width // len(line)) + line[:(self._width % len(line))] + '\n'

        # Fill in the table
            # Header
        self._table = self._line + ''.join(
            [f'{col:{self._alignment}{w}}' for col, w in zip(self._header_str, self._column_width)]
        ) + '\n' + self._line

            # Data
        for row in self._data_str:
            self._table += ''.join(
                [f'{col:{self._alignment}{w}}' for col, w in zip(row, self._column_width)]
            ) + '\n'
        self._table += self._line

        return self

    def add_note(self, note):
        self._table += note + '\n' + self._line

        return self

