from enum import Enum, auto


class PlotOptions:
    def __init__(
        self,
        title: str = "",
        legend: tuple[str, ...] = None,
        legend_title: str = None,
        supplementary_legend: tuple[str, ...] = None,
        supplementary_legend_title: str = None,
        options: tuple[dict, ...] = None,
        save_name: str = None,
        xlim: tuple[float, float] = None,
        ylim: tuple[float, float] = None,
        keep_frame: bool = True,
    ):
        self.title = title
        self.legend = legend
        self.legend_title = legend_title
        self.options = options
        self.save_name = save_name
        self.supplementary_legend = supplementary_legend
        self.supplementary_legend_title = supplementary_legend_title
        self.xlim = xlim
        self.ylim = ylim
        self.keep_frame = keep_frame


class CustomAnalysis:
    def __init__(self, name, fun):
        self.name = name
        self.fun = fun


class Integrator(Enum):
    RK4 = auto()
    RK45 = auto()
