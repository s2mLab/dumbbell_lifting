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


def get_time_at_precision(result, target: float):
    if target < 1:
        idx = np.where(np.sum(result.y, axis=0) > target)[0]
    else:
        idx = np.where(np.sum(result.y, axis=0) < target)[0]

    if idx.shape[0] == 0:
        return None
    return result.t[idx[0]]


class CustomColor:
    Green = "#00cc96"
    Yellow = "#ffa15a"
    Red = "#ef553b"
    Gray = "tab:gray"


class Study(Enum):
    STUDY1_STABILIZER_EFFECT_SHORT_TIME = StudyConfiguration(
        name="STUDY1_STABILIZER_EFFECT",
        repeat=1,
        fatigue_models=(
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=200),
                integrator=Integrator.RK45,
                x0=(0, 0.6, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=100),
                integrator=Integrator.RK45,
                x0=(0, 0.6, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=50),
                integrator=Integrator.RK45,
                x0=(0, 0.6, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=10),
                integrator=Integrator.RK45,
                x0=(0, 0.6, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=0),
                integrator=Integrator.RK45,
                x0=(0, 0.6, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
        ),
        t_end=1,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=1000,
        plot_options=PlotOptions(
            title="",
            legend=("$m_a$", "$m_r$", "$m_f$", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "$TL$"),
            supplementary_legend=("$S = 200$", "$S = 100$", "$S = 50$", "$S = 10$", "$S = 0$"),
            supplementary_legend_title="$m_a + m_r + m_f$",
            options=({"linestyle": "-"}, {"linestyle": "--"}, {"linestyle": "-."}, {"linestyle": ":"}, {"linestyle": "-", "linewidth": 3,}),
            save_path="STUDY1_STABILIZER_EFFECT.png",
            xlim=(0, 1.01),
            ylim=(0, 101),
            keep_frame=False,
        ),
        common_custom_analyses=(
            CustomAnalysis("First time with sum at 99.999%", lambda result: get_time_at_precision(result, 0.99999)),
            CustomAnalysis("First time with sum at 99.9999%", lambda result: get_time_at_precision(result, 0.999999)),
            CustomAnalysis("First time with sum at 99.99999%", lambda result: get_time_at_precision(result, 0.9999999)),
            CustomAnalysis("First time with sum at 99.999999%", lambda result: get_time_at_precision(result, 0.99999999)),
            CustomAnalysis("First time with sum at 99.9999999%", lambda result: get_time_at_precision(result, 0.999999999)),
        ),
    )

    STUDY1_STABILIZER_EFFECT_LONG_TIME = StudyConfiguration(
        name="STUDY1_STABILIZER_EFFECT",
        repeat=1,
        fatigue_models=(
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=200),
                integrator=Integrator.RK45,
                x0=(0, 0.6, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=100),
                integrator=Integrator.RK45,
                x0=(0, 0.6, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=50),
                integrator=Integrator.RK45,
                x0=(0, 0.6, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=10),
                integrator=Integrator.RK45,
                x0=(0, 0.6, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=0),
                integrator=Integrator.RK45,
                x0=(0, 0.6, 0),
                colors=(CustomColor.Green, CustomColor.Yellow, CustomColor.Red),
            ),
        ),
        t_end=60,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=100000,
        common_custom_analyses=(
            CustomAnalysis("First time with sum at 99.999%", lambda result: get_time_at_precision(result, 0.99999)),
            CustomAnalysis("First time with sum at 99.9999%", lambda result: get_time_at_precision(result, 0.999999)),
            CustomAnalysis("First time with sum at 99.99999%", lambda result: get_time_at_precision(result, 0.9999999)),
            CustomAnalysis("First time with sum at 99.999999%", lambda result: get_time_at_precision(result, 0.99999999)),
            CustomAnalysis("First time with sum at 99.9999999%", lambda result: get_time_at_precision(result, 0.999999999)),
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
            CustomAnalysis("Fatigue at final node", lambda result: result.y[2, -1]),
        ),
        plot_options=PlotOptions(
            title="",
            legend=(
            "$m_a$", "$m_r$", "$m_f$", "$m_a+m_r+m_f$", "_", "_", "_", "_", "$TL$", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_",
            "_", "$TL$"),
            # supplementary_legend=("$S = 200$", "$S = 100$", "$S = 50$", "$S = 10$", "$S = 0$"),
            # supplementary_legend_title="$m_a + m_r + m_f$",
            options=({"linestyle": "-"}, {"linestyle": "--"}),
            save_path="STUDY2_STABILIZER_EFFECT_SAME_START.png",
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
                x0=(0, 1-1e-4, 0),
                rms_indices=(0, 1, 2),
                custom_analyses=(
            CustomAnalysis("First time with sum at 99.999%", lambda result: get_time_at_precision(result, 0.99999)),
            CustomAnalysis("First time with sum at 99.9999%", lambda result: get_time_at_precision(result, 0.999999)),
            CustomAnalysis("First time with sum at 99.99999%", lambda result: get_time_at_precision(result, 0.9999999)),
            CustomAnalysis("First time with sum at 99.999999%", lambda result: get_time_at_precision(result, 0.99999999)),
            CustomAnalysis("First time with sum at 99.9999999%", lambda result: get_time_at_precision(result, 0.999999999)),
                )
            ),
            FatigueModels.XIA_STABILIZED(
                FatigueParameters(stabilization_factor=10),
                integrator=Integrator.RK45,
                x0=(0, 1+1e-4, 0),
                rms_indices=(0, 1, 2),
                custom_analyses=(
                    CustomAnalysis("First time with sum at 100.001%", lambda result: get_time_at_precision(result, 1.00001)),
                    CustomAnalysis("First time with sum at 100.0001%", lambda result: get_time_at_precision(result, 1.000001)),
                    CustomAnalysis("First time with sum at 100.00001%", lambda result: get_time_at_precision(result, 1.0000001)),
                    CustomAnalysis("First time with sum at 100.000001%", lambda result: get_time_at_precision(result, 1.00000001)),
                    CustomAnalysis("First time with sum at 100.0000001%", lambda result: get_time_at_precision(result, 1.000000001)),
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
            CustomAnalysis("Fatigue at final node", lambda result: result.y[2, -1]),
        ),
    )
