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
        # Study(Conditions.CONDITIONS_FATIGUE_TORQUE_NO_STABILIZATION),
        # Study(Conditions.CONDITIONS_ONLY_FATIGUE_NO_STABILIZATION),
        # Study(Conditions.CONDITIONS_ONLY_TORQUE_NO_STABILIZATION),
        # DONE
        # Study(Conditions.FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_17),
        # Study(Conditions.FULL_WINDOW_ONLY_FATIGUE_NO_STABILIZATION_18), # Did not converge
        # DONE
        # Study(Conditions.FULL_WINDOW_ONLY_TORQUE_NO_STABILIZATION_14),
        # Study(Conditions.FULL_WINDOW_ONLY_TORQUE_NO_STABILIZATION_15), # Did not converge
        # DONE
        # Study(Conditions.FULL_WINDOW_FATIGUE_TORQUE_NO_STABILIZATION_19),
        # Study(Conditions.FULL_WINDOW_FATIGUE_TORQUE_NO_STABILIZATION_20), # it converge until there.
        # Study(Conditions.FULL_WINDOW_FATIGUE_TORQUE_NO_STABILIZATION_21), # Did not converge
        # EXTRA SIMULATIONS
        # Study(Conditions.FULL_WINDOW_FATIGUE_TORQUE_WITH_STABILIZATION_20),
        # Study(Conditions.FULL_WINDOW_ONLY_FATIGUE_WITH_STABILIZATION_17),
        # Study(Conditions.FULL_WINDOW_ONLY_TORQUE_WITH_STABILIZATION_14),
        # Study(Conditions.CONDITIONS_FATIGUE_TORQUE_WITH_STABILIZATION_21),
        # Study(Conditions.CONDITIONS_ONLY_FATIGUE_WITH_STABILIZATION_20),
        # Study(Conditions.CONDITIONS_ONLY_TORQUE_WITH_STABILIZATION_9),


    )
    # all_studies = (
    # Study(Conditions.DEBUG_FAST),
    #                Study(Conditions.NO_STABILIZATION),
    #                )

    # --- Solve the program --- #
    for study in all_studies:
        # Perform the study (or reload)
        print("----------------")
        print(study.name)
        print("----------------")
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

        # if sol.status == 1:
        #     print(study.name)
        #     print("OCP did not converge")
        #     return


        # study.solution[x][y]
        # Condition x, if y is 0: then full solution, if y is 1: then a tuple of all intermediate OCP
        # study.plot_data_stacked_per_window()
        # study.plot_cpu_time()
        # study.plot_cost()
        # study.plot_cycle_cost()
        # study.plot_torques()
        # study.plot_pools(show=False, export=True)
        # study.plot_data_stacked_per_cycle()
        # study.plot_first_and_last_cycles()
        # study.solution[0][0].graphs()
        # study.solution[0][0].animate(
        #     show_floor=False,
        #     show_muscles=False,
        #     show_global_ref_frame=False,
        #     show_local_ref_frame=False,
        #     show_markers=False,
        #     show_gravity_vector=False,
        # )

        print("----------------")


if __name__ == "__main__":
    main()
