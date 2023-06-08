from enum import Enum

import numpy as np

from feasibility_studies import (
    StudyConfiguration,
    FatigueModels,
    TargetFunctions,
    FatigueParameters,
    Integrator,
    CustomAnalysis,
    PlotOptions,
)


def get_time_at_precision(result, precision: float):
    """
    Get the time at which the result is within the target precision.

    Parameters
    ----------
    result :
        The result to check.
    precision : float
        The target precision.

    Returns
    -------
    float
        The time at which the result is within the target precision.
    """

    boolean = np.abs(1 - np.sum(result.y, axis=0)) <= precision
    idx = np.where(boolean)[0]

    # if one wants to plot the precision
    # import matplotlib.pyplot as plt
    # plt.plot(result.t, np.abs(1 - np.sum(result.y, axis=0)))
    # # below precision in g dots
    # plt.plot(result.t[boolean], np.abs(1 - np.sum(result.y, axis=0))[boolean], 'go', ms=5)
    # # over precision in red dots
    # inv_boolean = np.logical_not(boolean)
    # plt.plot(result.t[inv_boolean], np.abs(1 - np.sum(result.y, axis=0))[inv_boolean], 'ro', ms=1)
    # # log y scale
    # plt.yscale('log')
    # plt.show()

    if idx.shape[0] == 0:
        return None
    # find the first index where the boolean is True for each value after the first index
    diff_idx = np.where(np.diff(idx) > 1)[0] + 1
    if diff_idx.shape[0] == 0:
        final_idx = idx[0]
    else:
        final_idx = result.y.shape[1] - 1 - (not np.where(np.flip(boolean))[0][0])

    if final_idx == result.y.shape[1] - 1:
        return None

    return result.t[final_idx]


class CustomColor:
    Green = "#00cc96"
    Yellow = "#ffa15a"
    Red = "#ef553b"
    Gray = "tab:gray"


mr_initial_value = 0.6

class Study(Enum):
    STUDY1_STABILIZER_EFFECT_SHORT_TIME = StudyConfiguration(
        name="STUDY1_STABILIZER_EFFECT_SHORT",
        repeat=1,
        fatigue_models=(
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=200),
                integrator=Integrator.RK45,
                x0=(0, mr_initial_value, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=100),
                integrator=Integrator.RK45,
                x0=(0, mr_initial_value, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=50),
                integrator=Integrator.RK45,
                x0=(0, mr_initial_value, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=10),
                integrator=Integrator.RK45,
                x0=(0, mr_initial_value, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=0),
                integrator=Integrator.RK45,
                x0=(0, mr_initial_value, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
        ),
        t_end=60,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=100000,
        plot_options=PlotOptions(
            title="",
            legend=(
                "$m_a$",
                "$m_r$",
                "$m_f$",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "$TL$",
            ),
            supplementary_legend=("$S = 200$", "$S = 100$", "$S = 50$", "$S = 10$", "$S = 0$"),
            supplementary_legend_title="$m_a + m_r + m_f$",
            options=(
                {"linestyle": "-"},
                {"linestyle": "--"},
                {"linestyle": "-."},
                {"linestyle": ":"},
                {
                    "linestyle": "-",
                    "linewidth": 3,
                },
            ),
            save_name="STUDY1_STABILIZER_EFFECT",
            xlim=(0, 1.01),
            ylim=(0, 101),
            keep_frame=False,
        ),
    )

    STUDY1_STABILIZER_EFFECT_LONG_TIME = StudyConfiguration(
        name="STUDY1_STABILIZER_EFFECT_LONG",
        repeat=1,
        fatigue_models=(
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=200),
                integrator=Integrator.RK45,
                x0=(0, mr_initial_value, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=100),
                integrator=Integrator.RK45,
                x0=(0, mr_initial_value, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=50),
                integrator=Integrator.RK45,
                x0=(0, mr_initial_value, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=10),
                integrator=Integrator.RK45,
                x0=(0, mr_initial_value, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=0),
                integrator=Integrator.RK45,
                x0=(0, mr_initial_value, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
        ),
        t_end=60,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=100000,
        common_custom_analyses=(
            CustomAnalysis("First time with sum at 99%", lambda result: get_time_at_precision(result, 1e-2)),
            CustomAnalysis("First time with sum at 99.9%", lambda result: get_time_at_precision(result, 1e-3)),
            CustomAnalysis("First time with sum at 99.99%", lambda result: get_time_at_precision(result, 1e-4)),
            CustomAnalysis("First time with sum at 99.999%", lambda result: get_time_at_precision(result, 1e-5)),
            CustomAnalysis("First time with sum at 99.9999%", lambda result: get_time_at_precision(result, 1e-6)),
            CustomAnalysis("First time with sum at 99.99999%", lambda result: get_time_at_precision(result, 1e-7)),
            CustomAnalysis("First time with sum at 99.999999%", lambda result: get_time_at_precision(result, 1e-8)),
            CustomAnalysis("First time with sum at 99.9999999%", lambda result: get_time_at_precision(result, 1e-9)),
        ),
        plot_options=PlotOptions(
            title="",
            legend=(
            "$m_a$", "$m_r$", "$m_f$", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_",
            "_", "$TL$"),
            supplementary_legend=("$S = 200$", "$S = 100$", "$S = 50$", "$S = 10$", "$S = 0$"),
            supplementary_legend_title="$m_a + m_r + m_f$",
            options=({"linestyle": "-"}, {"linestyle": "--"}, {"linestyle": "-."}, {"linestyle": ":"},
                     {"linestyle": "-", "linewidth": 3, }),
            # save_path="STUDY1_STABILIZER_EFFECT.png",
            xlim=(0, 1.01),
            ylim=(0, 101),
            keep_frame=False,
        ),
    )

    STUDY2_STABILIZER_EFFECT_SAME_START = StudyConfiguration(
        name="STUDY2_STABILIZER_EFFECT_SAME_START",
        repeat=1,
        fatigue_models=(
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=10),
                integrator=Integrator.RK45,
                x0=(0, 1, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
                rms_indices=(0, 1, 2),
            ),
            FatigueModels.XIA(
                FatigueParameters(stabilization_factor=10),
                integrator=Integrator.RK45,
                x0=(0, 1, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
                rms_indices=(0, 1, 2),
            ),
        ),
        t_end=60,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=100000,
        common_custom_analyses=(
            CustomAnalysis("Sum of components at the final index", lambda results: np.sum(results.y[:, -1], axis=0)),
            CustomAnalysis("ma at final node", lambda result: result.y[0, -1]),
            CustomAnalysis("mr at final node", lambda result: result.y[1, -1]),
            CustomAnalysis("mf at final node", lambda result: result.y[2, -1]),
        ),
        plot_options=PlotOptions(
            title="",
            legend=(
                "$m_a$",
                "$m_r$",
                "$m_f$",
                "$m_a+m_r+m_f$",
                "_",
                "_",
                "_",
                "_",
                "$TL$",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
                "$TL$",
            ),
            options=({"linestyle": "-"}, {"linestyle": "--"}),
            save_name="STUDY2_STABILIZER_EFFECT_SAME_START",
            xlim=(0, 60.1),
            ylim=(0, 101),
            keep_frame=False,
        ),
    )

    STUDY3_STABILIZER_EFFECT_SLIGHTLYBAD_START = StudyConfiguration(
        name="STUDY3_STABILIZER_EFFECT_SLIGHTLYBAD_START",
        repeat=1,
        fatigue_models=(
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=10),
                integrator=Integrator.RK45,
                x0=(0, 1 - 1e-4, 0),
                rms_indices=(0, 1, 2),
                custom_analyses=(
                    CustomAnalysis(
                        "First time with sum at 99.999%", lambda result: get_time_at_precision(result, 1e-5)
                    ),
                    CustomAnalysis(
                        "First time with sum at 99.9999%", lambda result: get_time_at_precision(result, 1e-6)
                    ),
                    CustomAnalysis(
                        "First time with sum at 99.99999%", lambda result: get_time_at_precision(result, 1e-7)
                    ),
                    CustomAnalysis(
                        "First time with sum at 99.999999%", lambda result: get_time_at_precision(result, 1e-8)
                    ),
                    CustomAnalysis(
                        "First time with sum at 99.9999999%", lambda result: get_time_at_precision(result, 1e-9)
                    ),
                ),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=10),
                integrator=Integrator.RK45,
                x0=(0, 1 + 1e-4, 0),
                rms_indices=(0, 1, 2),
                custom_analyses=(
                    CustomAnalysis(
                        "First time with sum at 100.001%", lambda result: get_time_at_precision(result, 1e-5)
                    ),
                    CustomAnalysis(
                        "First time with sum at 100.0001%", lambda result: get_time_at_precision(result, 1e-6)
                    ),
                    CustomAnalysis(
                        "First time with sum at 100.00001%", lambda result: get_time_at_precision(result, 1e-7)
                    ),
                    CustomAnalysis(
                        "First time with sum at 100.000001%", lambda result: get_time_at_precision(result, 1e-8)
                    ),
                    CustomAnalysis(
                        "First time with sum at 100.0000001%", lambda result: get_time_at_precision(result, 1e-9)
                    ),
                ),
            ),
            FatigueModels.XIA(
                FatigueParameters(),
                integrator=Integrator.RK45,
                x0=(0, 1, 0),
                rms_indices=(0, 1, 2),
            ),
        ),
        t_end=60,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=100000,
        common_custom_analyses=(
            CustomAnalysis("Sum of components at the final index", lambda results: np.sum(results.y[:, -1], axis=0)),
            CustomAnalysis("ma at final node", lambda result: result.y[0, -1]),
            CustomAnalysis("mr at final node", lambda result: result.y[1, -1]),
            CustomAnalysis("mf at final node", lambda result: result.y[2, -1]),
        ),
    )

    STUDY4_STABILIZER_EFFECT_SLIGHTLYBAD_START = StudyConfiguration(
        name="STUDY4_STABILIZER_EFFECT_SLIGHTLYBAD_START",
        repeat=1,
        fatigue_models=(
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=200),
                integrator=Integrator.RK45,
                x0=(0, 1 - 1e-4, 0),
                rms_indices=(0, 1, 2),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=100),
                integrator=Integrator.RK45,
                x0=(0, 1 - 1e-4, 0),
                rms_indices=(0, 1, 2),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=50),
                integrator=Integrator.RK45,
                x0=(0, 1 - 1e-4, 0),
                rms_indices=(0, 1, 2),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=20),
                integrator=Integrator.RK45,
                x0=(0, 1 - 1e-4, 0),
                rms_indices=(0, 1, 2),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=10),
                integrator=Integrator.RK45,
                x0=(0, 1 - 1e-4, 0),
                rms_indices=(0, 1, 2),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=5),
                integrator=Integrator.RK45,
                x0=(0, 1 - 1e-4, 0),
                rms_indices=(0, 1, 2),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=0),
                integrator=Integrator.RK45,
                x0=(0, 1 - 1e-4, 0),
                rms_indices=(0, 1, 2),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=200),
                integrator=Integrator.RK45,
                x0=(0, 1 + 1e-4, 0),
                rms_indices=(0, 1, 2),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=100),
                integrator=Integrator.RK45,
                x0=(0, 1 + 1e-4, 0),
                rms_indices=(0, 1, 2),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=50),
                integrator=Integrator.RK45,
                x0=(0, 1 + 1e-4, 0),
                rms_indices=(0, 1, 2),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=20),
                integrator=Integrator.RK45,
                x0=(0, 1 + 1e-4, 0),
                rms_indices=(0, 1, 2),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=10),
                integrator=Integrator.RK45,
                x0=(0, 1 + 1e-4, 0),
                rms_indices=(0, 1, 2),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=5),
                integrator=Integrator.RK45,
                x0=(0, 1 + 1e-4, 0),
                rms_indices=(0, 1, 2),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=0),
                integrator=Integrator.RK45,
                x0=(0, 1 + 1e-4, 0),
                rms_indices=(0, 1, 2),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
        ),
        t_end=60,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=100000,
        common_custom_analyses=(
            CustomAnalysis("First time with sum at 99.99%", lambda result: get_time_at_precision(result, 1e-4)),
            CustomAnalysis("First time with sum at 99.999%", lambda result: get_time_at_precision(result, 1e-5)),
            CustomAnalysis("First time with sum at 99.9999%", lambda result: get_time_at_precision(result, 1e-6)),
            CustomAnalysis("First time with sum at 99.99999%", lambda result: get_time_at_precision(result, 1e-7)),
            CustomAnalysis("First time with sum at 99.999999%", lambda result: get_time_at_precision(result, 1e-8)),
            CustomAnalysis("First time with sum at 99.9999999%", lambda result: get_time_at_precision(result, 1e-9)),
            CustomAnalysis("ma at final node", lambda result: result.y[0, -1]),
            CustomAnalysis("mr at final node", lambda result: result.y[1, -1]),
            CustomAnalysis("mf at final node", lambda result: result.y[2, -1]),
        ),
        plot_options=PlotOptions(
            title="",
            legend=(
                "$m_a$", "$m_r$", "$m_f$", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_",
                "_",
                "_", "$TL$"),
            supplementary_legend=("$S = 200$", "$S = 100$", "$S = 50$", "$S = 20$", "$S = 10$", "$S = 5$", "$S = 0$",
                                "$S = 200$", "$S = 100$", "$S = 50$", "$S = 20$", "$S = 10$", "$S = 5$", "$S = 0$"),
            supplementary_legend_title="$m_a + m_r + m_f$",
            options=({"linestyle": "-"}, {"linestyle": "--"}, {"linestyle": "-."}, {"linestyle": ":"},
                     {"linestyle": "-"}, {"linestyle": "-"},
                     {"linestyle": "-", "linewidth": 3, },
                     {"linestyle": "-"}, {"linestyle": "--"}, {"linestyle": "-."}, {"linestyle": ":"},
                     {"linestyle": "-"}, {"linestyle": "-"},
                     {"linestyle": "-", "linewidth": 3, }),
            # save_path="STUDY1_STABILIZER_EFFECT.png",
            xlim=(0, 60),
            ylim=(99.99, 100.01),
            keep_frame=False,
        ),
    )
