from enum import Enum

from matplotlib import colors as mcolors

from studies import StudyConfiguration, PlotOptions, StudySetup, get_ocp, Program


class Conditions(Enum):
    DEBUG_FAST = StudyConfiguration(
        studies=(get_ocp(Program.TORQUE_DRIVEN_XIA, StudySetup(split_controls=False)),),
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
            get_ocp(Program.TORQUE_DRIVEN_NO_FATIGUE, StudySetup()),
            get_ocp(Program.MUSCLE_DRIVEN_NO_FATIGUE, StudySetup()),
            get_ocp(Program.TORQUE_DRIVEN_XIA, StudySetup(split_controls=True)),
            get_ocp(Program.TORQUE_DRIVEN_XIA, StudySetup(split_controls=False)),
            get_ocp(Program.MUSCLE_DRIVEN_XIA, StudySetup()),
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

    STUDY1 = StudyConfiguration(
        studies=(
            # get_ocp(Program.TORQUE_DRIVEN_NO_FATIGUE, StudySetup()),
            # get_ocp(Program.TORQUE_DRIVEN_XIA, StudySetup(split_controls=True)),
            # get_ocp(Program.TORQUE_DRIVEN_XIA, StudySetup(split_controls=False)),
            # get_ocp(Program.MUSCLE_DRIVEN_NO_FATIGUE, StudySetup()),
            get_ocp(Program.MUSCLE_DRIVEN_XIA, StudySetup()),
        ),
        rmse_index=(0, 0, 0, 5, 5),
        plot_options=PlotOptions(
            title="",
            legend_indices=(
                True,
                True,
                True,
                True,
                True,
            ),
            options=(
                {"linestyle": "-", "color": mcolors.CSS4_COLORS["black"], "linewidth": 5},
                {"linestyle": "-", "color": mcolors.CSS4_COLORS["lightcoral"]},
                {"linestyle": "-", "color": mcolors.CSS4_COLORS["red"]},
                {"linestyle": "--", "color": mcolors.CSS4_COLORS["black"]},
                {"linestyle": "--", "color": mcolors.CSS4_COLORS["red"]},
            ),
            to_degrees=True,
            maximize=False,
            save_path=("feasibility_q0", "feasibility_q1"),
        ),
    )

