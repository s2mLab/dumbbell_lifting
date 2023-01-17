from studies_nmpc import Conditions
from studies import Study


def main():
    # all_studies = (Study(Conditions.CONDITIONS),)
    # all_studies = (Study(Conditions.CONDITIONS_2),)
    all_studies = (Study(Conditions.CONDITIONS_3),)
    # all_studies = (Study(Conditions.DEBUG_FAST),)

    # --- Solve the program --- #
    for study in all_studies:
        # Perform the study (or reload)
        study.run()

        # study.solution[x][y]
        # Condition x, if y is 0: then full solution, if y is 1: then a tuple of all intermediate OCP
        # study.plot_data_stacked_per_window()
        study.plot_cost()
        study.plot_torques()
        study.plot_pools()
        study.plot_data_stacked_per_cycle()
        study.plot_first_and_last_cycles()
        # study.solution[0][0].graphs()
        study.solution[0][0].animate(
            show_floor=False,
            show_muscles=False,
            show_global_ref_frame=False,
            show_local_ref_frame=False,
            show_markers=False,
            show_gravity_vector=False,
        )

        print("----------------")


if __name__ == "__main__":
    main()
