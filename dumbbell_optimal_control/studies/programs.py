from enum import Enum

from bioptim import ObjectiveList, ConstraintList, ObjectiveFcn

from .ocp import (
    DynamicsFcn,
    FatigableStructure,
    FatigueModels,
    FatigueParameters,
)
from .study_setup import StudySetup


class ProgramsFcn:
    @staticmethod
    def torque_driven_no_fatigue(_: StudySetup):
        fatigue_model = FatigueModels.NONE

        objectives = ObjectiveList()
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)

        # Keep arm pointing down as much as possible
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=0, weight=10_000)

        return r"$TauNf$", DynamicsFcn.TORQUE_DRIVEN, fatigue_model, objectives, ConstraintList()

    @staticmethod
    def muscles_driven_no_fatigue(_: StudySetup):
        fatigue_model = FatigueModels.NONE

        objectives = ObjectiveList()
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1)

        # Keep arm pointing down as much as possible
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=0, weight=10_000)

        return r"$MusNf$", DynamicsFcn.MUSCLE_DRIVEN, fatigue_model, objectives, ConstraintList()

    @staticmethod
    def torque_driven_xia(study_setup: StudySetup):
        fatigue_model = FatigueModels.XIA_STABILIZED(
            FatigableStructure.JOINTS,
            FatigueParameters(
                scaling=study_setup.tau_limits_no_muscles[1],
                split_controls=study_setup.split_controls,
                apply_on_joint=False,
            ),
        )

        objectives = ObjectiveList()
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau_minus_mf", weight=study_setup.weight_fatigue)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau_plus_mf", weight=study_setup.weight_fatigue)

        # Keep arm pointing down as much as possible
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=0, weight=10_000)

        return r"$TauXia$", DynamicsFcn.TORQUE_DRIVEN, fatigue_model, objectives, ConstraintList()

    @staticmethod
    def muscle_driven_xia(study_setup: StudySetup):
        fatigue_model = FatigueModels.XIA_STABILIZED(
            FatigableStructure.MUSCLES, FatigueParameters(apply_on_joint=False)
        )

        objectives = ObjectiveList()
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="muscles_mf", weight=study_setup.weight_fatigue)

        # Keep arm pointing down as much as possible
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=0, weight=10_000)

        return r"$MusXia$", DynamicsFcn.MUSCLE_DRIVEN, fatigue_model, objectives, ConstraintList()


class Program(Enum):
    TORQUE_DRIVEN_NO_FATIGUE = (ProgramsFcn.torque_driven_no_fatigue, )
    MUSCLE_DRIVEN_NO_FATIGUE = (ProgramsFcn.muscles_driven_no_fatigue, )
    TORQUE_DRIVEN_XIA = (ProgramsFcn.torque_driven_xia, )
    MUSCLE_DRIVEN_XIA = (ProgramsFcn.muscle_driven_xia, )
