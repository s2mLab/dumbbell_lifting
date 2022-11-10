from enum import Enum

from studies import StudyConfiguration, PlotOptions, StudySetup, get_nmpc, Program


class Conditions(Enum):
    DEBUG_FAST = StudyConfiguration(
        studies=(
            get_nmpc(Program.TORQUE_DRIVEN_XIA, StudySetup(n_total_round_trips=5, split_controls=False)),
        ),
        rmse_index=None,
        plot_options=PlotOptions(
            title="%s pour les conditions $C/\\tau\\varnothing$  et $C/\\alpha\\varnothing$",
            legend_indices=None,
            options=({"linestyle": "-"}, {"linestyle": "--"}),
            to_degrees=True,
        ),
    )

    DEBUG_ALL_CONDITIONS = StudyConfiguration(
        studies=(
            # get_nmpc(Program.TORQUE_DRIVEN_NO_FATIGUE, StudySetup(n_total_round_trips=5)),
            # get_nmpc(Program.MUSCLE_DRIVEN_NO_FATIGUE, StudySetup(n_total_round_trips=5)),
            get_nmpc(Program.TORQUE_DRIVEN_XIA, StudySetup(n_total_round_trips=20, split_controls=True)),
            # get_nmpc(Program.TORQUE_DRIVEN_XIA, StudySetup(n_total_round_trips=5, split_controls=False)),
            # get_nmpc(Program.MUSCLE_DRIVEN_XIA, StudySetup(n_total_round_trips=5)),
        ),
        rmse_index=None,
        plot_options=PlotOptions(
            title="Fast debugger",
            legend_indices=None,
            options=(
                {"linestyle": "-"},
                {"linestyle": "--"},
                {"linestyle": "-"},
                {"linestyle": "--"},
                {"linestyle": "-"},
            ),
            to_degrees=True,
        ),
    )
