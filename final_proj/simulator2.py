import xml.etree.ElementTree as ET
import random
import mujoco_viewer
import numpy as np
import dm_control.mujoco
import mujoco
import csv
import pandas as pd


class ParentNode:
    def __init__(self, element, position=0, upper_depth=0, lower_depth=0, type='root', actuators=None):
        self.element = element  # XML element
        self.position = position  # Position identifier for upper legs (1, 2, 3, 4)
        self.upper_depth = upper_depth  # Depth of upper leg in the hierarchy
        self.lower_depth = lower_depth  # Depth of lower leg in the hierarchy
        self.type = type  # Type of the parent (root, upper_leg, lower_leg)
        self.actuators = actuators

    def create_actuator(self, joint_name, gear):
        ET.SubElement(self.actuators, 'motor', name=f"{joint_name}_motor", joint=joint_name, gear=gear)

    def create_upper_leg(self, name, rgba, mass, gear, joint_range):
        """Creates an upper leg based on the position identifier."""
        # Map position identifier to specific coordinates
        range_str = f"{joint_range[0]} {joint_range[1]}" 
        positions = {
            1: "0.4 0 0",
            2: "0 0.4 0",
            3: "-0.4 0 0",
            4: "0 -0.4 0",
        }

        joint_pos = {
            1: "-0.2 0 0",
            2: "0 -0.2 0",
            3: "0.2 0 0",
            4: "0 0.2 0",
        }
        
        axis = {
            1: "0 1 0",
            2: "1 0 0",
            3: "0 1 0",
            4: "1 0 0",
        }

        euler_rotations = {
            1: "0 90 0",
            2: "90 0 0",
            3: "0 -90 0",
            4: "-90 0 0",
        }

        pos_0 = joint_range[0]
        pos_1 = joint_range[1]
        
        pos_1_3 = f"{pos_0} {pos_1}" 
        pos_2_4 = f"{pos_1} {-pos_0}"
        range = {
            1: pos_1_3,
            2: pos_2_4,
            3: pos_1_3,
            4: pos_2_4,
        }
        # print(self.type)

        if self.position in positions:
            pos = positions[self.position]
            jp = joint_pos[self.position]
            ax = axis[self.position]
            eu = euler_rotations[self.position]
            rg = range[self.position]
            upper_leg = ET.SubElement(self.element, 'body', name=f"{name}_upper_leg", pos=pos)
            ET.SubElement(upper_leg, 'joint', name=f"{name}_upper_joint", type="hinge", pos=jp ,axis=ax, range=rg, damping="0.1")
            ET.SubElement(upper_leg, 'geom', type="cylinder", size="0.05 0.2", rgba=rgba, euler=eu, mass=str(mass))
            self.create_actuator(f"{name}_upper_joint", str(gear))

            return ParentNode(upper_leg, self.position, self.upper_depth + 1, self.lower_depth, 'upper_leg', self.actuators)
        else:
            print("Invalid upper leg position")
            return None

    def create_lower_leg(self, name, rgba, mass, gear, joint_range):
        """Creates a lower leg attached to its upper leg."""
        # print(self.upper_depth, " is current upper depth")
        if(self.type == 'upper_leg'):
            pos_extend = 0.2
            positions = {
                1: str(pos_extend) +" 0 -0.2",
                2: "0 " + str(pos_extend) + " -0.2",
                3: str(-pos_extend) +" 0 -0.2",
                4: "0 " + str(-pos_extend) + " -0.2",
            }
        else:
            if self.upper_depth == 0:
                pos_extend = -0.2
            else:
                pos_extend = -0.4
            positions = {
                1: "0 0 " + str(pos_extend),
                2: "0 0 " + str(pos_extend),
                3: "0 0 " + str(pos_extend),
                4: "0 0 " + str(pos_extend),
            }
        axis = {
            1: "0 1 0",
            2: "1 0 0",
            3: "0 1 0",
            4: "1 0 0",
        }

        pos_0 = joint_range[0]
        pos_1 = joint_range[1]
        pos_1_3 = f"{pos_1} {-pos_0}"
        pos_2_4 = f"{pos_0} {pos_1}" 

        range = {
            1: pos_1_3,
            2: pos_2_4,
            3: pos_1_3,
            4: pos_2_4,
        }

        # range = {
        #     1: "0 30",
        #     2: "-30 0",
        #     3: "-30 0",
        #     4: "0 30",
        # }

        if self.position in positions:
            pos = positions[self.position]
            ax = axis[self.position]
            rg = range[self.position]
            lower_leg = ET.SubElement(self.element, 'body', name=f"{name}_lower_leg", pos=pos)
            ET.SubElement(lower_leg, 'joint', name=f"{name}_lower_joint", type="hinge", pos="0 0 0.2",axis=ax, range=rg, damping="0.1")
            ET.SubElement(lower_leg, 'geom', type="cylinder", pos="0 0 0", size="0.04 0.2", rgba=rgba, mass=str(mass))
            self.create_actuator(f"{name}_lower_joint", str(gear))

            return ParentNode(lower_leg, self.position, self.upper_depth, self.lower_depth + 1, 'lower_leg', self.actuators)

        else:
            print("Invalid lower leg position")
            return None


def create_spider_model(model_file, random_num_upper_legs, random_num_lower_legs, u_mass, l_mass, u_gear, l_gear, u_joint_range_min, l_joint_range_min):
    mujoco = ET.Element('mujoco')
    mujoco.set('model', 'spider_robot')

    actuators = ET.SubElement(mujoco, 'actuator')

    # Global options
    option = ET.SubElement(mujoco, 'option')
    option.set('gravity', '0 0 -9.81')

    # World body
    worldbody = ET.SubElement(mujoco, 'worldbody')
    ET.SubElement(worldbody, 'light', diffuse=".5 .5 .5", pos="0 0 6", dir="0 0 -2")
    ET.SubElement(worldbody, 'geom', type="plane", size="10 10 0.1", rgba=".9 .9 .9 1", mass="1")

    # Main body setup
    main_body = ET.SubElement(worldbody, 'body', name="main", pos="0 0 1.5")
    ET.SubElement(main_body, 'joint', type="free")
    ET.SubElement(main_body, 'geom', type="sphere", size="0.2", rgba="0.486 0.917 0.97 0.75")
    root_node = ParentNode(main_body, 0, 0, 0, 'root', actuators)

    # Creating upper legs with different positions
    leg_colors = ["1 0 0 1", "0 1 0 1", "0 0 1 1", "0.5 0.5 0.5 1"]

    for i in range(1, 5):  # Positions 1 to 4
        upper_leg_node = ParentNode(main_body, i, 0, 0, 'upper_leg', actuators).create_upper_leg(f"leg{i}", leg_colors[i-1], u_mass, u_gear, u_joint_range_min)
        # ET.SubElement(actuator, 'motor', name=f"leg{i}_upper_leg_moter", gear=100, joint=f"leg{i}_upper_joint")
        parent = upper_leg_node
        for q in range(int(random_num_upper_legs)):
            child = parent.create_upper_leg(f"leg{4*(q+1)+i}", leg_colors[i-1], u_mass, u_gear, u_joint_range_min)
            parent = child
        for j in range(int(random_num_lower_legs)):
            child = parent.create_lower_leg(f"leg{4*j+i}", "0.486 0.917 0.97 0.75",l_mass, l_gear, l_joint_range_min)
            parent = child

    # Save the model
    tree = ET.ElementTree(mujoco)
    tree.write(model_file)
    # print(f"Model saved to spider_model.xml")


def animation_4_display(filename, random_num_upper_legs, random_num_lower_legs):
    create_spider_model(filename, random_num_upper_legs, random_num_lower_legs)
    model = dm_control.mujoco.MjModel.from_xml_path("spider_model.xml")
    data = dm_control.mujoco.MjData(model)

    viewer = mujoco_viewer.MujocoViewer(model, data)
    print("Number of control inputs (actuators):", model.nu)
    total_actuators = model.nu

    # Simulation loop
    for i in range(2000):
        if viewer.is_alive:
            t = 2*np.sin(0.1 * i)
            for j in range(model.nu // 4):
                data.ctrl[j] = t  
                data.ctrl[j + model.nu // 4] = -t  
                data.ctrl[j + model.nu // 2] = t 
                data.ctrl[j + model.nu // 4 * 3] = -t 

            dm_control.mujoco.mj_step(model, data)
            viewer.render()
        else:
            break

    viewer.close()

