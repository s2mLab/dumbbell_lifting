import pickle
import os

from studies_nmpc import Conditions
from studies import Study


def main():

    all_studies = (
        Study(Conditions.CONDITIONS_FATIGUE_TORQUE),
        Study(Conditions.CONDITIONS_ONLY_FATIGUE),
        Study(Conditions.CONDITIONS_ONLY_TORQUE),
        Study(Conditions.FULL_WINDOW_FATIGUE_TORQUE),
        Study(Conditions.FULL_WINDOW_ONLY_FATIGUE),
        Study(Conditions.FULL_WINDOW_ONLY_TORQUE),
        # NO STABILIZATION
        Study(Conditions.CONDITIONS_FATIGUE_TORQUE_NO_STABILIZATION),
        Study(Conditions.CONDITIONS_ONLY_FATIGUE_NO_STABILIZATION),
        Study(Conditions.CONDITIONS_ONLY_TORQUE_NO_STABILIZATION),
        # NO STABILIZATION
        Study(Conditions.FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_17),
        # Study(Conditions.FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_18), # Did not converge
        # NO STABILIZATION
        Study(Conditions.FULL_WINDOW_ONLY_TORQUE_NO_STABILIZATION_14),
        # Study(Conditions.FULL_WINDOW_ONLY_TORQUE_NO_STABILIZATION_15), # Did not converge
        # NO STABILIZATION
        Study(Conditions.FULL_WINDOW_FATIGUE_TORQUE_NO_STABILIZATION_20),
        # Study(Conditions.FULL_WINDOW_FATIGUE_TORQUE_NO_STABILIZATION_21), # Did not converge
        # EXTRA SIMULATIONS
        Study(Conditions.FULL_WINDOW_FATIGUE_TORQUE_WITH_STABILIZATION_20),
        Study(Conditions.FULL_WINDOW_ONLY_FATIGUE_WITH_STABILIZATION_17),
        Study(Conditions.FULL_WINDOW_ONLY_TORQUE_WITH_STABILIZATION_14),
        Study(Conditions.CONDITIONS_FATIGUE_TORQUE_WITH_STABILIZATION_21),
        Study(Conditions.CONDITIONS_ONLY_FATIGUE_WITH_STABILIZATION_20),
        Study(Conditions.CONDITIONS_ONLY_TORQUE_WITH_STABILIZATION_9),


    )

    # --- Solve the program --- #
    for study in all_studies:
        # Perform the study (or reload)
        print("----------------")
        print(study.name)
        print("----------------")
        # result folders
        # study.result_folder = "results"
        study.result_folder = "results_again"
        # study.result_folder = "results_no_stabilization"
        # study.result_folder = "results_extra_simulations"
        study.run()
        study.save()

        print(study.solution_costs)
        for i, sol in enumerate(study.solution[0][1]):
            print(f"OCP {i}")
            print(sol.status)

        print("----------------")


if __name__ == "__main__":
    main()
