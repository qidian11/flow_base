import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any, List


class ExperimentVisualizer:
    def __init__(self, default_figsize: Tuple[int, int] = (8, 6)):
        self.default_figsize = default_figsize
        self.series_data: List[Dict[str, Any]] = []
        self.group_colors: Dict[str, str] = {}  # 存储组名到颜色的映射
        self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.fig = None
        self.axes = None

    def _smooth_ema(self, data: np.ndarray, weight: float) -> np.ndarray:
        """内部私有方法：指数移动平均平滑"""
        if weight <= 0: return data
        last = data[0]
        smoothed = np.zeros_like(data)
        for i, point in enumerate(data):
            smoothed_val = last * weight + (1 - weight) * point
            smoothed[i] = smoothed_val
            last = smoothed_val
        return smoothed

    def add_data(self, label: str, data: Union[str, Path, np.ndarray],
                 group: str = "default", color: Optional[str] = None, **plot_kwargs):
        """
        载入数据。
        :param group: 组名。相同组名的线条会自动分配相同（或相近）的颜色。
        :param color: 手动指定颜色。若指定，则覆盖自动分配。
        """
        if isinstance(data, (str, Path)):
            y_values = np.load(data)
        else:
            y_values = data

        # 颜色分配逻辑
        if color:
            self.group_colors[group] = color
        elif group not in self.group_colors:
            # 如果是新组，从默认色循环中取下一个
            color_idx = len(self.group_colors) % len(self.color_cycle)
            self.group_colors[group] = self.color_cycle[color_idx]

        self.series_data.append({
            'label': label,
            'y': y_values,
            'group': group,
            'color': self.group_colors[group],
            'kwargs': plot_kwargs
        })
        return self

    def create_subplots(self, n_rows: int = 1, n_cols: int = 1, figsize: Optional[Tuple[int, int]] = None):
        size = figsize if figsize else self.default_figsize
        self.fig, self.axes = plt.subplots(n_rows, n_cols, figsize=size)
        if n_rows * n_cols == 1:
            self.axes = [self.axes]
        else:
            self.axes = self.axes.flatten()
        return self.fig, self.axes

    def plot_on_ax(self,
                   ax_index: int,
                   title: str,
                   smooth: bool = True,
                   smooth_weight: float = 0.85,
                   show_raw: bool = True,
                   y_lim: Optional[Tuple[float, float]] = None,
                   x_start: int = 0,
                   legend_kwargs: Optional[Dict] = None):
        """
        在画布上绘图。
        :param smooth: 是否开启平滑。
        :param smooth_weight: 平滑权重 (0.0 - 1.0)。
        :param show_raw: 是否在平滑线下方叠加半透明的原始数据阴影。
        """
        ax = self.axes[ax_index]

        for item in self.series_data:
            y_raw = item['y']
            if len(y_raw) <= x_start: continue

            y_to_plot = y_raw[x_start:]
            x_axis = np.arange(x_start, len(y_raw))
            color = item['color']

            # 如果开启平滑
            if smooth:
                y_smoothed = self._smooth_ema(y_raw, smooth_weight)[x_start:]

                # 叠加原始图像（作为背景阴影）
                if show_raw:
                    ax.plot(x_axis, y_to_plot, color=color, alpha=0.15, linewidth=1, label='_nolegend_')

                # 绘制平滑主线
                ax.plot(x_axis, y_smoothed, label=item['label'], color=color, **item['kwargs'])
            else:
                # 仅绘制原始线
                ax.plot(x_axis, y_to_plot, label=item['label'], color=color, **item['kwargs'])

        ax.set_title(title)
        if y_lim: ax.set_ylim(y_lim)
        ax.grid(True, linestyle=':', alpha=0.6)
        # --- 重点修改这里 ---
        if legend_kwargs is None:
            # handlelength 默认是 2.0，我们放大到 3.5 就能完美展示长虚线了
            legend_kwargs = {'handlelength': 3.5}

        ax.legend(**legend_kwargs)
        return ax

    def show(self):
        if self.fig:
            self.fig.tight_layout()
            plt.show()

    def save(self, filename: str, dpi: int = 300, auto_close: bool = False):
        if self.fig:
            self.fig.tight_layout()
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            if auto_close:
                import matplotlib.pyplot as plt
                plt.close(self.fig)