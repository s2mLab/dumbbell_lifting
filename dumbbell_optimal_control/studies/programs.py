from enum import Enum

from bioptim import ObjectiveList, ConstraintList, ObjectiveFcn, ConstraintFcn, Node
from casadi import MX, if_else

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

        # fatigue constraints
        constraints = ConstraintList()

        # torque max
        tau_max = 50

        def TL_plus_mf_inf_one_plus(all_pn) -> MX:
            """

            Parameters
            ----------
            all_pn: PenaltyNodeList
                The penalty node elements
           """

            if all_pn.nlp.u_bounds.max[0, 1] != 0:
                return MX(0)
                # otherwise, this one is not used...
            else:
                return all_pn.nlp.controls["tau_plus"].cx / tau_max + all_pn.nlp.states["tau_plus_mf"].cx

        def TL_plus_mf_inf_one_minus(all_pn) -> MX:
            """

            Parameters
            ----------
            all_pn: PenaltyNodeList
                The penalty node elements
           """
            if all_pn.nlp.u_bounds.min[0, 1] != 0:
                return all_pn.nlp.controls["tau_minus"].cx / - tau_max + all_pn.nlp.states["tau_minus_mf"].cx
            else:
                return MX(0)

        constraints.add(TL_plus_mf_inf_one_plus, min_bound=0,  max_bound=1, node=Node.ALL)
        constraints.add(TL_plus_mf_inf_one_minus, min_bound=0, max_bound=1, node=Node.ALL)

        return r"$TauXia$", DynamicsFcn.TORQUE_DRIVEN, fatigue_model, objectives, constraints

    @staticmethod
    def torque_driven_xia_fatigue_only(study_setup: StudySetup):
        fatigue_model = FatigueModels.XIA_STABILIZED(
            FatigableStructure.JOINTS,
            FatigueParameters(
                scaling=study_setup.tau_limits_no_muscles[1],
                split_controls=study_setup.split_controls,
                apply_on_joint=False,
            ),
        )

        objectives = ObjectiveList()

        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau_minus_mf", weight=study_setup.weight_fatigue)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau_plus_mf", weight=study_setup.weight_fatigue)

        # Keep arm pointing down as much as possible
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=0, weight=10_000)

        # fatigue constraints
        constraints = ConstraintList()

        # torque max
        tau_max = 50

        def TL_plus_mf_inf_one_plus(all_pn) -> MX:
            """

            Parameters
            ----------
            all_pn: PenaltyNodeList
                The penalty node elements
           """

            if all_pn.nlp.u_bounds.max[0, 1] != 0:
                return MX(0)
                # otherwise, this one is not used...
            else:
                return all_pn.nlp.controls["tau_plus"].cx / tau_max + all_pn.nlp.states["tau_plus_mf"].cx

        def TL_plus_mf_inf_one_minus(all_pn) -> MX:
            """

            Parameters
            ----------
            all_pn: PenaltyNodeList
                The penalty node elements
           """
            if all_pn.nlp.u_bounds.min[0, 1] != 0:
                return all_pn.nlp.controls["tau_minus"].cx / - tau_max + all_pn.nlp.states["tau_minus_mf"].cx
            else:
                return MX(0)

        constraints.add(TL_plus_mf_inf_one_plus, min_bound=0, max_bound=1, node=Node.ALL)
        constraints.add(TL_plus_mf_inf_one_minus, min_bound=0, max_bound=1, node=Node.ALL)

        return r"$TauXia_mf$", DynamicsFcn.TORQUE_DRIVEN, fatigue_model, objectives, constraints

    @staticmethod
    def torque_driven_xia_torque_only(study_setup: StudySetup):
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

        # Keep arm pointing down as much as possible
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=0, weight=10_000)

        # fatigue constraints
        constraints = ConstraintList()

        # torque max
        tau_max = 50

        def TL_plus_mf_inf_one_plus(all_pn) -> MX:
            """

            Parameters
            ----------
            all_pn: PenaltyNodeList
                The penalty node elements
           """

            if all_pn.nlp.u_bounds.max[0, 1] != 0:
                return MX(0)
                # otherwise, this one is not used...
            else:
                return all_pn.nlp.controls["tau_plus"].cx / tau_max + all_pn.nlp.states["tau_plus_mf"].cx

        def TL_plus_mf_inf_one_minus(all_pn) -> MX:
            """

            Parameters
            ----------
            all_pn: PenaltyNodeList
                The penalty node elements
           """
            if all_pn.nlp.u_bounds.min[0, 1] != 0:
                return all_pn.nlp.controls["tau_minus"].cx / - tau_max + all_pn.nlp.states["tau_minus_mf"].cx
            else:
                return MX(0)

        constraints.add(TL_plus_mf_inf_one_plus, min_bound=0, max_bound=1, node=Node.ALL)
        constraints.add(TL_plus_mf_inf_one_minus, min_bound=0, max_bound=1, node=Node.ALL)

        return r"$TauXia_mf$", DynamicsFcn.TORQUE_DRIVEN, fatigue_model, objectives, constraints

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
    TORQUE_DRIVEN_XIA_FATIGUE_ONLY = (ProgramsFcn.torque_driven_xia_fatigue_only,)
    TORQUE_DRIVEN_XIA_TORQUE_ONLY = (ProgramsFcn.torque_driven_xia_torque_only,)
    MUSCLE_DRIVEN_XIA = (ProgramsFcn.muscle_driven_xia, )
