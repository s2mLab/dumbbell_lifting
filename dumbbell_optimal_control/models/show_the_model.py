"""
This file is to display the human model into bioviz
"""
import bioviz


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

# model_name = "arm26.bioMod"
model_name = "arm26_viz.bioMod" # this model was only used to display the results because it included more meshfiles.
b = bioviz.Viz(
    model_name,
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
b.set_camera_position(-0.5, 3.5922578781963685, 0.1)
b.resize(1000, 1000)
if export_model:
    b.snapshot("doc/model.png")
b.exec()

print("roll")
print(b.get_camera_roll())
print("zoom")
print(b.get_camera_zoom())
print("position")
print(b.get_camera_position())
print("get_camera_focus_point")
print(b.get_camera_focus_point())