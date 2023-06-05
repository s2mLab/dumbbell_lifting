from typing import Union

from .ocp import OcpConfiguration, PlotOptions


class StudyConfiguration:
    def __init__(
        self,
        studies: tuple[OcpConfiguration, ...],
        rmse_index: Union[tuple[int, ...], None],
        plot_options: PlotOptions,
    ):
        self.studies = studies
        self.rmse_index = rmse_index

        if isinstance(plot_options, tuple):
            plot_options = plot_options[0]
        self.plot_options = plot_options if plot_options is None else plot_options
        if len(self.plot_options.options) < len(self.studies):
            raise ValueError("len(plot_options.options) must be >= len(studies)")
