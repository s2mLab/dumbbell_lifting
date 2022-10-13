from enum import Enum

from studies import StudyConfiguration, PlotOptions, StudySetup, get_nmpc, Program


class Conditions(Enum):
    DEBUG_FAST = StudyConfiguration(
        studies=(get_nmpc(Program.TORQUE_DRIVEN_XIA, StudySetup(split_controls=False)),),
        rmse_index=None,
        plot_options=PlotOptions(
            title="%s pour les conditions $C/\\tau\\varnothing$  et $C/\\alpha\\varnothing$",
            legend_indices=None,
            options=({"linestyle": "-"}, {"linestyle": "--"}),
            to_degrees=True,
        ),
    )
