import json, os

from python import die, model_control

selected_component = "Car Door Panel Stamping"
selected_process = "Cold Stamping"
selected_material = "Aluminium Aloy 5754"

def predictionmesh (die_dir, edge_dir, blank_dir, qml):
    # print(blank_dir)
    die.load(die_dir, edge_dir, blank_dir, qml)


def components ():
    return [ component for component in os.listdir("components") if component[0] != "." ]

def processes (component):
    return [ process for process in os.listdir(os.path.join("components", component)) if process[0] != "."  ]


def materials (component, process):
    return [ material for material in os.listdir(os.path.join("components", component, process)) if material[0] != "." ]


def indicators (component, process, material):
    return [ indicator for indicator in os.listdir(os.path.join("components", component, process, material)) if indicator[0] != "." ]






#-------------------------------------------------------------------------------------
# LEGACY
#-------------------------------------------------------------------------------------

def process_inputs ():
    global selected_component

    with open('info.json') as f:
        info = json.load(f)

        for process in info["processes"]:
            if process["name"] == selected_component:
                return [ input["name"] for input in process["inputs"] ]
        

def process_input_units (input_name):
    global selected_component

    with open('info.json') as f:
        info = json.load(f)

        for process in info["processes"]:
            if process["name"] == selected_component:
                for input in process["inputs"]:
                    if input["name"] == input_name:
                        return input["units"]
                    

def process_input_lowerbound (input_name):
    global selected_component

    with open('info.json') as f:
        info = json.load(f)

        for process in info["processes"]:
            if process["name"] == selected_component:
                for input in process["inputs"]:
                    if input["name"] == input_name:
                        return input["lower bound"]
                    
                    
def process_input_upperbound (input_name):
    global selected_component

    with open('info.json') as f:
        info = json.load(f)

        for process in info["processes"]:
            if process["name"] == selected_component:
                for input in process["inputs"]:
                    if input["name"] == input_name:
                        return input["upper bound"]
                    
                    
def process_input_decimals (input_name):
    global selected_component

    with open('info.json') as f:
        info = json.load(f)

        for process in info["processes"]:
            if process["name"] == selected_component:
                for input in process["inputs"]:
                    if input["name"] == input_name:
                        return input["decimals"]


def process_outputs (process_name):
    with open('info.json') as f:
        info = json.load(f)

        for process in info["processes"]:
            if process["name"] == process_name:
                return [ output["name"] for output in process["outputs"] ]
            

    # return [ model["outputs"] if model["name"] for model in info["models"] == model_type  ]


def select_materialandprocess (material, process):
    global selected_component, selected_material

    selected_material = material
    selected_component = process

    model_control.selectMaterialandProcess(material, process)


def get_selected_materialandprocess ():
    global selected_component, selected_material

    return selected_component, selected_material


def optivaropts ():
    return (["python","python1","python2"])


def independentvars ():
    return (["sensitivity","independent","variables"])


def dependentvars ():
    return (["sensitivity","dependent","variables"])


