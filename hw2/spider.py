import mujoco
import mujoco_viewer
import numpy as np
import dm_control.mujoco

model = dm_control.mujoco.MjModel.from_xml_path('spider.xml')
data = dm_control.mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)

# Simulation loop
for i in range(10000):
    if viewer.is_alive:
        dm_control.mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# Close the viewer
viewer.close()