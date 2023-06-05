from bioptim import XiaTauFatigue, XiaFatigueStabilized
from bioptim.dynamics.fatigue.muscle_fatigue import MuscleFatigue

from .enums import FatigableStructure


class FatigueParameters:
    # according to https://doi.org/10.1016/j.jbiomech.2018.06.005
    # Elbow Parameters
    def __init__(
        self,
        LD: float = 10,
        LR: float = 10,
        # F: float = 0.00912,
        F: float = 0.456,
        # F: float = 0.912,
        # R: float = 0.00094,
        R: float = 0.0094,
        scaling: float = 1,
        # stabilization_factor: float = 10,
        stabilization_factor: float = 0,
        effort_factor: float = 0.0075,
        effort_threshold: float = 0.2,
        split_controls: bool = True,
        apply_on_joint: bool = False,
    ):
        self.LD = LD
        self.LR = LR
        self.F = F
        self.R = R
        self.scaling = scaling
        self.effort_factor = effort_factor
        self.stabilization_factor = stabilization_factor
        self.effort_threshold = effort_threshold
        self.split_controls = split_controls
        self.apply_on_joint = apply_on_joint


class FatigueModel:
    def __init__(
        self,
        model: MuscleFatigue,
    ):
        self.model = model


class Xia(FatigueModel):
    def __init__(self, fatigable_structure: FatigableStructure, fatigue_params: FatigueParameters):
        if fatigable_structure == FatigableStructure.JOINTS:
            model = XiaTauFatigue(
                XiaFatigueStabilized(
                    LD=fatigue_params.LD,
                    LR=fatigue_params.LR,
                    F=fatigue_params.F,
                    R=fatigue_params.R,
                    stabilization_factor=fatigue_params.stabilization_factor,
                    scaling=-fatigue_params.scaling,
                    apply_to_joint_dynamics=fatigue_params.apply_on_joint,
                ),
                XiaFatigueStabilized(
                    LD=fatigue_params.LD,
                    LR=fatigue_params.LR,
                    F=fatigue_params.F,
                    R=fatigue_params.R,
                    stabilization_factor=fatigue_params.stabilization_factor,
                    scaling=fatigue_params.scaling,
                    apply_to_joint_dynamics=fatigue_params.apply_on_joint,
                ),
                split_controls=fatigue_params.split_controls,
            )
        elif fatigable_structure == FatigableStructure.MUSCLES:
            model = XiaFatigueStabilized(
                LD=fatigue_params.LD,
                LR=fatigue_params.LR,
                F=fatigue_params.F,
                R=fatigue_params.R,
                stabilization_factor=fatigue_params.stabilization_factor,
                scaling=fatigue_params.scaling,
                apply_to_joint_dynamics=fatigue_params.apply_on_joint,
            )
        else:
            raise NotImplementedError("Fatigue structure model not implemented")

        super(Xia, self).__init__(model)


class FatigueModels:
    NONE = None
    XIA_STABILIZED = Xia
