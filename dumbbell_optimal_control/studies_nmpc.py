from enum import Enum

from studies import StudyConfiguration, PlotOptions, StudySetup, get_nmpc, Program, get_ocp


def get_nmpc_n_round_trip(n_trips):
    return get_nmpc(Program.TORQUE_DRIVEN_XIA_FATIGUE_ONLY,
                    StudySetup(
                        round_trip_time=1,
                        n_shoot_per_round_trip=50,
                        n_round_trips_to_advance=1,
                        n_round_trips=n_trips,
                        n_total_round_trips=n_trips,
                        split_controls=True,
                        fatigue_stabilization_factor=0,
                    ), )


def get_nmpc_n_round_trip_torque_only(n_trips):
    return get_nmpc(Program.TORQUE_DRIVEN_XIA_TORQUE_ONLY,
                    StudySetup(
                        round_trip_time=1,
                        n_shoot_per_round_trip=50,
                        n_round_trips_to_advance=1,
                        n_round_trips=n_trips,
                        n_total_round_trips=n_trips,
                        split_controls=True,
                        fatigue_stabilization_factor=0,
                    ), )


def get_nmpc_n_round_trip_fatigue_torque(n_trips):
    return get_nmpc(Program.TORQUE_DRIVEN_XIA,
                    StudySetup(
                        round_trip_time=1,
                        n_shoot_per_round_trip=50,
                        n_round_trips_to_advance=1,
                        n_round_trips=n_trips,
                        n_total_round_trips=n_trips,
                        split_controls=True,
                        fatigue_stabilization_factor=0,
                    ), )


def plot_options():
    return PlotOptions(
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


class Conditions(Enum):
    DEBUG_FAST = StudyConfiguration(
        studies=(
            get_nmpc(Program.TORQUE_DRIVEN_XIA,
                     StudySetup(
                         n_round_trips_to_advance=1,
                         n_round_trips=2,
                         n_total_round_trips=3,
                         split_controls=True,
                     )),
        ),
        rmse_index=None,
        plot_options=PlotOptions(
            title="%s pour les conditions $C/\\tau\\varnothing$  et $C/\\alpha\\varnothing$",
            legend_indices=None,
            options=({"linestyle": "-"}, {"linestyle": "--"}),
            to_degrees=True,
        ),
    )

    CONDITIONS_FATIGUE_TORQUE = StudyConfiguration(
        studies=(
            get_nmpc(Program.TORQUE_DRIVEN_XIA,
                     StudySetup(
                         round_trip_time=1,
                         n_shoot_per_round_trip=50,
                         n_round_trips_to_advance=1,
                         n_round_trips=3,
                         n_total_round_trips=60,
                         split_controls=True,
                     ), ),
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
    CONDITIONS_ONLY_FATIGUE = StudyConfiguration(
        studies=(
            get_nmpc(Program.TORQUE_DRIVEN_XIA_FATIGUE_ONLY,
                     StudySetup(
                         round_trip_time=1,
                         n_shoot_per_round_trip=50,
                         n_round_trips_to_advance=1,
                         n_round_trips=3,
                         n_total_round_trips=60,
                         split_controls=True,
                     ), ),
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
    CONDITIONS_ONLY_TORQUE = StudyConfiguration(
        studies=(
            get_nmpc(Program.TORQUE_DRIVEN_XIA_TORQUE_ONLY,
                     StudySetup(
                         round_trip_time=1,
                         n_shoot_per_round_trip=50,
                         n_round_trips_to_advance=1,
                         n_round_trips=3,
                         n_total_round_trips=60,
                         split_controls=True,
                     ), ),
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

    FULL_WINDOW_FATIGUE_TORQUE = StudyConfiguration(
        studies=(
            get_nmpc(Program.TORQUE_DRIVEN_XIA,
                     StudySetup(
                         round_trip_time=1,
                         n_shoot_per_round_trip=50,
                         n_round_trips_to_advance=1,
                         n_round_trips=32,
                         n_total_round_trips=32,
                         split_controls=True,
                     ), ),
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

    FULL_WINDOW_ONLY_TORQUE = StudyConfiguration(
        studies=(
            get_nmpc(Program.TORQUE_DRIVEN_XIA_TORQUE_ONLY,
                     StudySetup(
                         round_trip_time=1,
                         n_shoot_per_round_trip=50,
                         n_round_trips_to_advance=1,
                         n_round_trips=32,
                         n_total_round_trips=32,
                         split_controls=True,
                     ), ),
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

    FULL_WINDOW_ONLY_FATIGUE = StudyConfiguration(
        studies=(
            get_nmpc(Program.TORQUE_DRIVEN_XIA_FATIGUE_ONLY,
                     StudySetup(
                         round_trip_time=1,
                         n_shoot_per_round_trip=50,
                         n_round_trips_to_advance=1,
                         n_round_trips=32,
                         n_total_round_trips=32,
                         split_controls=True,
                     ), ),
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

    CONDITIONS_FATIGUE_TORQUE_WITH_STABILIZATION_21 = StudyConfiguration(
        studies=(
            get_nmpc(Program.TORQUE_DRIVEN_XIA,
                     StudySetup(
                         round_trip_time=1,
                         n_shoot_per_round_trip=50,
                         n_round_trips_to_advance=1,
                         n_round_trips=3,
                         n_total_round_trips=21,
                         split_controls=True,
                     ), ),
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
    CONDITIONS_ONLY_FATIGUE_WITH_STABILIZATION_20 = StudyConfiguration(
        studies=(
            get_nmpc(Program.TORQUE_DRIVEN_XIA_FATIGUE_ONLY,
                     StudySetup(
                         round_trip_time=1,
                         n_shoot_per_round_trip=50,
                         n_round_trips_to_advance=1,
                         n_round_trips=3,
                         n_total_round_trips=20,
                         split_controls=True,
                     ), ),
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
    CONDITIONS_ONLY_TORQUE_WITH_STABILIZATION_9 = StudyConfiguration(
        studies=(
            get_nmpc(Program.TORQUE_DRIVEN_XIA_TORQUE_ONLY,
                     StudySetup(
                         round_trip_time=1,
                         n_shoot_per_round_trip=50,
                         n_round_trips_to_advance=1,
                         n_round_trips=3,
                         n_total_round_trips=9,
                         split_controls=True,
                     ), ),
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

    FULL_WINDOW_FATIGUE_TORQUE_WITH_STABILIZATION_20 = StudyConfiguration(
        studies=(
            get_nmpc(Program.TORQUE_DRIVEN_XIA,
                     StudySetup(
                         round_trip_time=1,
                         n_shoot_per_round_trip=50,
                         n_round_trips_to_advance=1,
                         n_round_trips=20,
                         n_total_round_trips=20,
                         split_controls=True,
                     ), ),
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

    FULL_WINDOW_ONLY_TORQUE_WITH_STABILIZATION_14 = StudyConfiguration(
        studies=(
            get_nmpc(Program.TORQUE_DRIVEN_XIA_TORQUE_ONLY,
                     StudySetup(
                         round_trip_time=1,
                         n_shoot_per_round_trip=50,
                         n_round_trips_to_advance=1,
                         n_round_trips=14,
                         n_total_round_trips=14,
                         split_controls=True,
                     ), ),
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

    FULL_WINDOW_ONLY_FATIGUE_WITH_STABILIZATION_17 = StudyConfiguration(
        studies=(
            get_nmpc(Program.TORQUE_DRIVEN_XIA_FATIGUE_ONLY,
                     StudySetup(
                         round_trip_time=1,
                         n_shoot_per_round_trip=50,
                         n_round_trips_to_advance=1,
                         n_round_trips=17,
                         n_total_round_trips=17,
                         split_controls=True,
                     ), ),
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

    FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_3 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip(3),),
        rmse_index=None, plot_options=PlotOptions(
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
    FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_4 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip(4),),
        rmse_index=None, plot_options=PlotOptions(
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
    FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_5 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip(5),),
        rmse_index=None, plot_options=PlotOptions(
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
    FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_6 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip(6),),
        rmse_index=None, plot_options=PlotOptions(
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
    FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_7 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip(7),),
        rmse_index=None, plot_options=PlotOptions(
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
    FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_8 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip(8),),
        rmse_index=None, plot_options=PlotOptions(
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
    FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_9 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip(9),),
        rmse_index=None, plot_options=PlotOptions(
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
    FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_10 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip(10),),
        rmse_index=None, plot_options=PlotOptions(
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
    FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_11 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip(11),),
        rmse_index=None, plot_options=PlotOptions(
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
    FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_12 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip(12),),
        rmse_index=None, plot_options=PlotOptions(
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
    FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_13 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip(13),),
        rmse_index=None, plot_options=PlotOptions(
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
    FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_14 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip(14),),
        rmse_index=None, plot_options=PlotOptions(
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
    FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_15 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip(15),),
        rmse_index=None, plot_options=PlotOptions(
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
    FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_16 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip(16),),
        rmse_index=None, plot_options=PlotOptions(
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
    FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_17 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip(17),),
        rmse_index=None, plot_options=PlotOptions(
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
    FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_18 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip(18),),
        rmse_index=None, plot_options=PlotOptions(
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

    FULL_WINDOW_ONLY_TORQUE_NO_STABILIZATION_9 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip_torque_only(9),),
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
    FULL_WINDOW_ONLY_TORQUE_NO_STABILIZATION_10 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip_torque_only(10),),
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
    FULL_WINDOW_ONLY_TORQUE_NO_STABILIZATION_11 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip_torque_only(11),),
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

    FULL_WINDOW_ONLY_TORQUE_NO_STABILIZATION_12 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip_torque_only(12),),
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
    FULL_WINDOW_ONLY_TORQUE_NO_STABILIZATION_13 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip_torque_only(13),),
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
    FULL_WINDOW_ONLY_TORQUE_NO_STABILIZATION_14 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip_torque_only(14),),
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

    # FATIGUE_TORQUE
    FULL_WINDOW_FATIGUE_TORQUE_NO_STABILIZATION_15 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip_fatigue_torque(15),),
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
    FULL_WINDOW_FATIGUE_TORQUE_NO_STABILIZATION_16 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip_fatigue_torque(16),),
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
    FULL_WINDOW_FATIGUE_TORQUE_NO_STABILIZATION_17 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip_fatigue_torque(17),),
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

    FULL_WINDOW_FATIGUE_TORQUE_NO_STABILIZATION_18 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip_fatigue_torque(18),),
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
    FULL_WINDOW_FATIGUE_TORQUE_NO_STABILIZATION_19 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip_fatigue_torque(19),),
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
    FULL_WINDOW_FATIGUE_TORQUE_NO_STABILIZATION_20 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip_fatigue_torque(20),),
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
    FULL_WINDOW_FATIGUE_TORQUE_NO_STABILIZATION_21 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip_fatigue_torque(21),),
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
    FULL_WINDOW_FATIGUE_TORQUE_NO_STABILIZATION_22 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip_fatigue_torque(22),),
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

    FULL_WINDOW_FATIGUE_TORQUE_NO_STABILIZATION_23 = StudyConfiguration(
        studies=(get_nmpc_n_round_trip_fatigue_torque(23),),
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
