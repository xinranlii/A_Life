import xml.etree.ElementTree as ET
import random
import mujoco_viewer
import numpy as np
import dm_control.mujoco
import mujoco
import csv

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


def create_spider_model(model_file, random_num_upper_legs, random_num_lower_legs):
    mujoco = ET.Element('mujoco')
    mujoco.set('model', 'spider_robot')

    actuators = ET.SubElement(mujoco, 'actuator')

    # Global options
    option = ET.SubElement(mujoco, 'option')
    option.set('gravity', '0 0 -9.81')

    # World body
    worldbody = ET.SubElement(mujoco, 'worldbody')
    ET.SubElement(worldbody, 'light', diffuse=".5 .5 .5", pos="0 0 6", dir="0 0 -2")
    ET.SubElement(worldbody, 'geom', type="plane", size="5 5 0.1", rgba=".9 .9 .9 1", mass="1")

    # Main body setup
    main_body = ET.SubElement(worldbody, 'body', name="main", pos="0 0 1.5")
    ET.SubElement(main_body, 'joint', type="free")
    ET.SubElement(main_body, 'geom', type="sphere", size="0.2", rgba="0.486 0.917 0.97 0.75")
    root_node = ParentNode(main_body, 0, 0, 0, 'root', actuators)

    # Creating upper legs with different positions
    leg_colors = ["1 0 0 1", "0 1 0 1", "0 0 1 1", "0.5 0.5 0.5 1"]

    # [u_mass, l_mass, gear, u_joint_range_min, u_joint_range_max, l_joint_range_min, l_joint_range_max] = mutation()
    mutated_variables = mutation()

    u_mass = mutated_variables['new_upper_mass']
    l_mass = mutated_variables['new_lower_mass']
    gear = mutated_variables['new_gear']
    u_joint_range_min = mutated_variables['new_upper_joint_range_min']
    u_joint_range_max = mutated_variables['new_upper_joint_range_max']
    l_joint_range_min = mutated_variables['new_lower_joint_range_min']
    l_joint_range_max = mutated_variables['new_lower_joint_range_max']

    for i in range(1, 5):  # Positions 1 to 4
        upper_leg_node = ParentNode(main_body, i, 0, 0, 'upper_leg', actuators).create_upper_leg(f"leg{i}", leg_colors[i-1], u_mass, gear, (u_joint_range_min, u_joint_range_max))
        # ET.SubElement(actuator, 'motor', name=f"leg{i}_upper_leg_moter", gear=100, joint=f"leg{i}_upper_joint")
        parent = upper_leg_node
        for q in range(random_num_upper_legs):
            child = parent.create_upper_leg(f"leg{4*(q+1)+i}", leg_colors[i-1], u_mass, gear, (u_joint_range_min, u_joint_range_max))
            parent = child
        for j in range(random_num_lower_legs):
            child = parent.create_lower_leg(f"leg{4*j+i}", "0.486 0.917 0.97 0.75",l_mass, gear, (l_joint_range_min, l_joint_range_max))
            parent = child

    # Save the model
    tree = ET.ElementTree(mujoco)
    tree.write(model_file)
    # print(f"Model saved to spider_model.xml")
    return random_num_upper_legs, random_num_lower_legs
    
def fitness_function(model_file, control_strategy, randomness, simulation_steps=1000):
    random_num_upper_legs, random_num_lower_legs = randomness()
    create_spider_model(model_file, random_num_upper_legs, random_num_lower_legs)
    model = dm_control.mujoco.MjModel.from_xml_path(model_file)
    data = dm_control.mujoco.MjData(model)

    # Get the initial position of the robot's center of mass
    initial_pos = np.copy(data.qpos[:3])  # Assumes qpos[0:3] represents the main body's position

    # Simulation loop
    for i in range(simulation_steps):
        control_strategy(model, data, i)

        dm_control.mujoco.mj_step(model, data)

    # Calculate the final position and compute the distance moved
    final_pos = np.copy(data.qpos[:3])
    distance_moved = np.linalg.norm(final_pos - initial_pos)

    # The fitness value could be the distance moved
    return distance_moved

def animation(model, data, current_step):
    t = np.sin(0.01 * current_step)
    for j in range(model.nu // 4):
        data.ctrl[j] = t  
        data.ctrl[j + model.nu // 4] = -t 
        data.ctrl[j + model.nu // 2] = t  
        data.ctrl[j + model.nu // 4 * 3] = -t 

def animation_4_display(filename, random_num_upper_legs, random_num_lower_legs):
    # random_num_upper_legs = random.randint(0, 3)
    # random_num_lower_legs = random.randint(1, 4)
    create_spider_model(filename, random_num_upper_legs, random_num_lower_legs)
    model = dm_control.mujoco.MjModel.from_xml_path("spider_model.xml")
    data = dm_control.mujoco.MjData(model)

    viewer = mujoco_viewer.MujocoViewer(model, data)
    print("Number of control inputs (actuators):", model.nu)
    total_actuators = model.nu
    # eg upper_layer = 3
    # lower_layer = 2
    # total = (3 + 2) * 4
    # upper = [1, 2, 3, 6, 7, 8, 11, 12, 13 ,16 ,17, 18]
    #          1  1  1  2  2  2   3   3   3   4   4   4
    # lower = [4, 5, 9, 10, 14, 15, 19, 20]
    #          1  1  2   2   3   3   4   4

    # Simulation loop
    for i in range(2000):
        if viewer.is_alive:
            t = 2*np.sin(0.1 * i)
            # data.ctrl[:] = np.sin(0.1 * i)
            # for j in range(model.nu // 4):
            #     data.ctrl[j * 4] = t  # Even indices
            #     data.ctrl[j * 4 + 2] = -t  # Odd indices, opposing movement
            # data.ctrl[0] = np.sin(0.1 * i)
            # data.ctrl[model.nu / 4 * 1] = - np.sin(0.1 * i)
            # data.ctrl[model.nu / 4 * 2] = np.sin(0.1 * i)
            # data.ctrl[model.nu / 4 * 3] = - np.sin(0.1 * i)
                
            for j in range(model.nu // 4):
                data.ctrl[j] = t  # Even indices
                data.ctrl[j + model.nu // 4] = -t  # Even indices
                data.ctrl[j + model.nu // 2] = t  # Odd indices, opposing movement
                data.ctrl[j + model.nu // 4 * 3] = -t  # Even indices
                

            dm_control.mujoco.mj_step(model, data)
            viewer.render()
        else:
            break

    # Close the viewer
    viewer.close()
        
# def randomness():
#     random_num_upper_legs = random.randint(0, 3)
#     random_num_lower_legs = random.randint(1, 4)
#     print(random_num_upper_legs + 1, "layers of upper legs")
#     print(random_num_lower_legs, "layers of lower legs")
#     return random_num_upper_legs, random_num_lower_legs
    
# Based on the fitness and observation in hw3, the number of layers of lower legs which is equal or larger than the number of layers of upper legs performs better
# Thus, we modify the randomness function
def randomness():
    random_num_upper_legs = random.randint(0, 3)
    random_num_lower_legs = random.randint(random_num_upper_legs, 4)
    print(random_num_upper_legs + 1, "layers of upper legs")
    print(random_num_lower_legs, "layers of lower legs")
    return random_num_upper_legs, random_num_lower_legs

# def mutation():
#     # Define mutation ranges for mass, gear ratio, and joint range
#     mass_upper_mutation_range = (1.0, 5.0) 
#     mass_lower_mutation_range = (1.0, 10.0) 
#     gear_mutation_range = (100, 300) 
#     joint_range_mutation = [(-30, 0), (-25, 0), (-20, 0), (-15, 0)]

#     # Randomly select new attribute values within the defined ranges
#     new_upper_mass = random.uniform(*mass_upper_mutation_range)
#     new_lower_mass = random.uniform(*mass_lower_mutation_range)
#     new_gear = random.uniform(*gear_mutation_range)
#     new_upper_joint_range = random.choice(joint_range_mutation) 
#     new_lower_joint_range = random.choice(joint_range_mutation) 
#     print("new_upper_mass", "new_lower_mass",  "new_gear" ,"new_upper_joint_range", "new_lower_joint_range")
#     print(str(new_upper_mass), str(new_lower_mass), str(new_gear), new_upper_joint_range, new_lower_joint_range)
    
#     return str(new_upper_mass), str(new_lower_mass), str(new_gear), new_upper_joint_range, new_lower_joint_range

def mutation():
    mass_upper_mutation_range = (1.0, 5.0)
    mass_lower_mutation_range = (1.0, 10.0)
    gear_mutation_range = (100, 300)
    joint_range_mutation = [(-30, 0), (-25, 0), (-20, 0), (-15, 0)]
    new_upper_joint_range = random.choice(joint_range_mutation) 
    new_lower_joint_range = random.choice(joint_range_mutation) 

    mutated_variables = {
        'new_upper_mass': random.uniform(*mass_upper_mutation_range),
        'new_lower_mass': random.uniform(*mass_lower_mutation_range),
        'new_gear': random.uniform(*gear_mutation_range),
        'new_upper_joint_range_min': new_upper_joint_range[0],
        'new_upper_joint_range_max': new_upper_joint_range[1],
        'new_lower_joint_range_min': new_lower_joint_range[0],
        'new_lower_joint_range_max': new_lower_joint_range[1]
    }
    print(mutated_variables)

    return mutated_variables

# fitness_function(model_file, animation, (random_num_upper_legs, random_num_lower_legs), simulation_steps=2000)

def generate_data(num_entries, file_name):
    data_list = []

    for i in range(num_entries):
        mutation_result = mutation()
        random_num_upper_legs, random_num_lower_legs = randomness()
        randomness_dict = {
            'random_num_upper_legs': random_num_upper_legs + 1,  # Adding 1 to match your print statement logic
            'random_num_lower_legs': random_num_lower_legs,
        }
        model_filename = f"{model_file}_{i}.xml"  # Create a unique filename for each iteration
        
        # Calculate the fitness score for this configuration
        fitness_score = fitness_function(model_filename, animation, lambda: (random_num_upper_legs, random_num_lower_legs), simulation_steps=2000)
        
        # Combine all the data including fitness score
        combined_data = {**mutation_result, **randomness_dict, 'fitness_score': fitness_score}
        data_list.append(combined_data)

    # Write data to CSV
    write_data_to_csv(file_name, data_list)

def write_data_to_csv(file_name, data):
    with open(file_name, mode='w', newline='') as file:
        if not data:  
            return
        writer = csv.DictWriter(file, fieldnames=data[0].keys()) 
        writer.writeheader() 
        writer.writerows(data)  

model_file = "spider_model.xml"
csv_file_name = 'fitness.csv'

simulation_time = 20
# for i in range(simulation_time):
#     fitness = fitness_function(model_file+str(i+1), animation, randomness, simulation_steps=2000)
#     print(f"Fitness (Distance Moved): {fitness:.2f}")
# data = generate_data(simulation_time, csv_file_name)

random_num_upper_legs, random_num_lower_legs = randomness()
animation_4_display(model_file, random_num_upper_legs, random_num_lower_legs)