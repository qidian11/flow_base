import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any


class ExperimentVisualizer:
    """
    一个标准化的科学绘图轮子，支持多数据载入、多子图并排渲染与局部切片。
    """

    def __init__(self, default_figsize: Tuple[int, int] = (16, 6)):
        """
        初始化可视化器。
        :param default_figsize: 默认的画布长宽比 (宽, 高)
        """
        self.default_figsize = default_figsize
        self.series_data: Dict[str, Dict[str, Any]] = {}
        self.fig = None
        self.axes = None

    def add_data(self, label: str, data: Union[str, Path, np.ndarray], **plot_kwargs):
        """
        载入一组数据。支持直接传入 numpy array，或传入 .npy 文件路径。

        :param label: 图例中显示的名称
        :param data: 数据本身 (numpy 数组) 或 .npy 文件的路径
        :param plot_kwargs: 透传给 matplotlib.plot 的参数 (如 color='blue', linestyle='--')
        """
        if isinstance(data, (str, Path)):
            y_values = np.load(data)
        elif isinstance(data, np.ndarray):
            y_values = data
        else:
            raise ValueError("传入的数据必须是 numpy 数组或 .npy 文件路径。")

        self.series_data[label] = {
            'y': y_values,
            'kwargs': plot_kwargs
        }
        return self  # 支持链式调用

    def create_subplots(self, n_rows: int = 1, n_cols: int = 2, figsize: Optional[Tuple[int, int]] = None):
        """
        生成指定排列方式的画布。
        """
        size = figsize if figsize else self.default_figsize
        self.fig, self.axes = plt.subplots(n_rows, n_cols, figsize=size)

        # 统一将 axes 转换为一维列表，方便后续按索引操作
        if n_rows * n_cols == 1:
            self.axes = [self.axes]
        elif isinstance(self.axes, np.ndarray):
            self.axes = self.axes.flatten()

        return self.fig, self.axes

    def plot_on_ax(self,
                   ax_index: int,
                   title: str,
                   x_label: str = 'Steps',
                   y_label: str = 'Loss Value',
                   x_start: int = 0,
                   y_lim: Optional[Tuple[float, float]] = None,
                   show_grid: bool = True):
        """
        将目前载入的所有数据，绘制到指定的子图 (Axis) 上。

        :param ax_index: 子图的索引 (例如 0 代表左图，1 代表右图)
        :param x_start: 数据的起始截断点 (用于放大后半部分)
        :param y_lim: Y 轴的上下限，例如 (-65, 0)
        """
        if self.axes is None or ax_index >= len(self.axes):
            raise RuntimeError(f"子图索引 {ax_index} 越界或画布未初始化，请先调用 create_subplots()。")

        ax = self.axes[ax_index]

        for label, item in self.series_data.items():
            y = item['y']
            kwargs = item['kwargs']

            # 安全处理截断：如果数据长度大于起始点才绘制
            if len(y) > x_start:
                x_axis = np.arange(x_start, len(y))
                y_sliced = y[x_start:]
                ax.plot(x_axis, y_sliced, label=label, **kwargs)

        # 样式设置
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if y_lim:
            ax.set_ylim(y_lim)
        if show_grid:
            ax.grid(True, linestyle=':', alpha=0.6)

        ax.legend()
        return ax

    def show(self):
        """自动调整布局并展示"""
        if self.fig:
            self.fig.tight_layout()
        plt.show()

    def close(self):
        """关闭当前画布，释放内存并重置内部指针"""
        if self.fig:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
            self.axes = None

    def save(self, filename: str, dpi: int = 300, auto_close: bool = False):
        """保存高质量图片"""
        if self.fig:
            self.fig.tight_layout()
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"图表已保存至: {filename}")

            if auto_close:
                self.close()