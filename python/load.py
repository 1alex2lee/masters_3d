import json

from python import die, model_control

selected_process = "Car Door Panel Stamping"
selected_material = "Aluminium Aloy 5754"

def predictionmesh (die_dir, edge_dir, blank_dir, qml):
    # print(blank_dir)
    die.load(die_dir, edge_dir, blank_dir, qml)

def materials ():
    with open('info.json') as f:
        info = json.load(f)

        return info["materials"]


def processes ():
    with open('info.json') as f:
        info = json.load(f)

        return [ process["name"] for process in info["processes"] ]


def modeltypes ():
    global selected_process

    with open('info.json') as f:
        info = json.load(f)

        for process in info["processes"]:
            if process["name"] == selected_process:
                return process["models"]


def process_inputs ():
    global selected_process

    with open('info.json') as f:
        info = json.load(f)

        for process in info["processes"]:
            if process["name"] == selected_process:
                return [ input["name"] for input in process["inputs"] ]
        

def process_input_units (input_name):
    global selected_process

    with open('info.json') as f:
        info = json.load(f)

        for process in info["processes"]:
            if process["name"] == selected_process:
                for input in process["inputs"]:
                    if input["name"] == input_name:
                        return input["units"]
                    

def process_input_lowerbound (input_name):
    global selected_process

    with open('info.json') as f:
        info = json.load(f)

        for process in info["processes"]:
            if process["name"] == selected_process:
                for input in process["inputs"]:
                    if input["name"] == input_name:
                        return input["lower bound"]
                    
                    
def process_input_upperbound (input_name):
    global selected_process

    with open('info.json') as f:
        info = json.load(f)

        for process in info["processes"]:
            if process["name"] == selected_process:
                for input in process["inputs"]:
                    if input["name"] == input_name:
                        return input["upper bound"]
                    
                    
def process_input_decimals (input_name):
    global selected_process

    with open('info.json') as f:
        info = json.load(f)

        for process in info["processes"]:
            if process["name"] == selected_process:
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
    global selected_process, selected_material

    selected_material = material
    selected_process = process

    model_control.selectMaterialandProcess(material, process)


def get_selected_materialandprocess ():
    global selected_process, selected_material

    return selected_process, selected_material


def optivaropts ():
    return (["python","python1","python2"])


def independentvars ():
    return (["sensitivity","independent","variables"])


def dependentvars ():
    return (["sensitivity","dependent","variables"])


