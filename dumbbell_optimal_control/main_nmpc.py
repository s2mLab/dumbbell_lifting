from studies_nmpc import Conditions
from studies import Study


def main():
    all_studies = (Study(Conditions.DEBUG_FAST),)

    # --- Solve the program --- #
    for study in all_studies:
        # Perform the study (or reload)
        study.run()

        # study.solution[x][y]
        # Condition x, if y is 0: then full solution, if y is 1: then a tuple of all intermediate OCP
        study.solution[0][0].graphs()
        print("----------------")


if __name__ == "__main__":
    main()
