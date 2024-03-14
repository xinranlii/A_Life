import mujoco
import mujoco_viewer
import numpy as np
import dm_control.mujoco

model = dm_control.mujoco.MjModel.from_xml_path('door.xml')
data = dm_control.mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)

door_angle = 0
handle_angle = 0

# Example indices, replace with actual indices from your XML file
door_hinge_idx = 0  # Example index for door_hinge
handle_hinge_idx = 1  # Example index for handle_hinge

# Constants for angle adjustments
door_angle_step = 180  # Amount to adjust door angle by each step
handle_angle_step = 30  # Amount to adjust handle angle by each step

# Simulation loop
for i in range(10000):
    if viewer.is_alive:
        if i % 100 == 0:
            # door_angle = -door_angle
            # handle_angle = (handle_angle + 30) % 60
            # Oscillate door angle between -90 and 90
            door_angle += door_angle_step
            if door_angle > 90:
                door_angle_step = -180
            elif door_angle < -90:
                door_angle_step = 180
            
            # Rotate handle angle within 0 to 60 degrees
            handle_angle += handle_angle_step
            if handle_angle > 60:
                handle_angle = 0  # Reset to 0 if it exceeds 60

        data.ctrl[door_hinge_idx] = door_angle
        data.ctrl[handle_hinge_idx] = handle_angle

        dm_control.mujoco.mj_step(model, data)

        viewer.render()
    else:
        break

# Close the viewer
viewer.close()