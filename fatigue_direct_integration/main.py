from feasibility_studies import FatigueIntegrator
from studies import Study


def main():
    # Define the studies to perform
    all_studies = (
        Study.STUDY1_STABILIZER_EFFECT_SHORT_TIME,
        Study.STUDY1_STABILIZER_EFFECT_LONG_TIME,
        Study.STUDY2_STABILIZER_EFFECT_SAME_START,
        Study.STUDY3_STABILIZER_EFFECT_SLIGHTLYBAD_START,
    )

    # Prepare and run the studies
    for study in all_studies:
        print(f"Performing study: {study.name}")
        runner = FatigueIntegrator(study.value)
        runner.perform()

        # Print some results
        runner.print_integration_time()
        if len(runner.study.fatigue_models) == 2:
            runner.print_rmse()
        runner.print_custom_analyses()
        runner.plot_results()
        print("----------------")


if __name__ == "__main__":
    main()
