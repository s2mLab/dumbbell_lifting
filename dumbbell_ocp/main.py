from studies import Study, Conditions, DataType


def main():
    all_studies = (Study(Conditions.STUDY1),)

    # --- Solve the program --- #
    for study in all_studies:
        study.run()

        # Show (for debug purposes)
        export_model = True
        background_color = (1, 1, 1) if export_model else (0.5, 0.5, 0.5)
        show_gravity_vector = False if export_model else True
        show_floor = False if export_model else True
        show_local_ref_frame = False if export_model else True
        show_global_ref_frame = False if export_model else True
        show_markers = False if export_model else True
        show_mass_center = False if export_model else True
        show_global_center_of_mass = False if export_model else True
        show_segments_center_of_mass = False if export_model else True
        study.solution[-1].animate(
            show_gravity_vector=show_gravity_vector,
            show_floor=show_floor,
            show_local_ref_frame=show_local_ref_frame,
            show_global_ref_frame=show_global_ref_frame,
            show_markers=show_markers,
            show_mass_center=show_mass_center,
            show_global_center_of_mass=show_global_center_of_mass,
            show_segments_center_of_mass=show_segments_center_of_mass,
            mesh_opacity=1,
            background_color=background_color,
        )

        study.generate_latex_table()
        study.save_solutions()
        study.prepare_plot_data(DataType.STATES, "q")
        # study.plot()
        print("----------------")


if __name__ == "__main__":
    main()
