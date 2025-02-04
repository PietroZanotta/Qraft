# errors evaluation and graph on random circuits

import statistics
import numpy as np  
from qiskit.providers.fake_provider import Fake5QV1
from circuit import CircuitGenerator
from state import generate_circuit_state
import joblib
import matplotlib.pyplot as plt 
import os

output_dir = "error_plots" 
os.makedirs(output_dir, exist_ok=True) 

def plot(x_lab, y_lab, x, y_baseline, y_qraft, name):
  fig = plt.subplots(figsize =(12, 8)) 

  br1 = np.arange(len(y_baseline)) 
  br2 = [x + 0.30 for x in br1] 

  plt.bar(br1, y_baseline, color ="y", width = 0.30, label ="base") 
  plt.bar(br2, y_qraft, color ="b", width = 0.30, label ="qraft")

  plt.xlabel(x_lab, fontweight ="bold", fontsize = 15) 
  plt.ylabel(y_lab, fontweight ="bold", fontsize = 15) 
  plt.xticks([r + 0.30 for r in range(len(y_baseline))], x)

  plt.legend()
  filename = f"{name}.png"  
  filepath = os.path.join(output_dir, filename) 
  plt.savefig(filepath) 
#   plt.show()


qraft = joblib.load("qraft.pkl")

backend = Fake5QV1()
circuit_generator = CircuitGenerator()

stats_data= {
  "names": [],
  "medians": {
    "base": [],
    "qraft": []
  },

  "dse": {
    "base": [],
    "qraft": []
  },

  "program_error": {
    "base": [],
    "qraft": []
  }
}

for circuit_name in range(10):
  circuit_width = 3
  circuit_depth = 4
  gates = circuit_generator.get_applicable_gates(num_qubits=circuit_width,depth=circuit_depth)
  fc_circuit = circuit_generator.get_fc_circuit(circuit_width, gates)
  frc_circuit = circuit_generator.get_frc_circuit(circuit_width, gates)
  states, extras = generate_circuit_state(fc_circuit, frc_circuit, circuit_width, circuit_depth, backend)

  dominant_state = extras["dominant_state"]

  testing_data = {}
  for state in states:
    name = state.pop()
    result = state.pop()
    testing_data[name] = {
      "result": result,
      "data": state
    }

  print(testing_data)

  qraft_extra = {
    "states_errors": {},
    "program_error": 0,
    "dominant_state_error": 0
  }

  for state in testing_data:
    input = np.array(testing_data[state]["data"])
    input = input.reshape(1, -1)

    y_pred = qraft.predict(input)
    y_pred = y_pred[0]
    qraft_extra["states_errors"][state] = abs(y_pred - testing_data[state]["result"]) /100
    if state == dominant_state:
      qraft_extra["dominant_state_error"] = qraft_extra["states_errors"][state]
    qraft_extra["program_error"] += qraft_extra["states_errors"][state]  

    print(f"Predicted:{y_pred} . Actual:{testing_data[state]["result"]}")

  qraft_extra["program_error"] = qraft_extra["program_error"] / 2

  medians = (
    statistics.median([error * 100 for error in extras["states_errors"].values()]),
    statistics.median([error * 100 for error in qraft_extra["states_errors"].values()])
  )
  dse = (
    extras["dominant_state_error"] * 100,
    qraft_extra["dominant_state_error"] * 100,
  )
  pe = (
    extras["program_error"] * 100,
    qraft_extra["program_error"] * 100
  )
  stats_data["names"].append(f"Circuit {circuit_name}")
  stats_data["medians"]["base"].append(medians[0])
  stats_data["medians"]["qraft"].append(medians[1])
  stats_data["dse"]["base"].append(dse[0])
  stats_data["dse"]["qraft"].append(dse[1])
  stats_data["program_error"]["base"].append(pe[0])
  stats_data["program_error"]["qraft"].append(pe[1])


plot(x_lab="Algorithms", y_lab="State Error %", x=stats_data["names"], y_baseline=stats_data["medians"]["base"], y_qraft=stats_data["medians"]["qraft"], name = "states")
plot(x_lab="Algorithms", y_lab="Dominant State Error %", x=stats_data["names"], y_baseline=stats_data["dse"]["base"], y_qraft=stats_data["dse"]["qraft"], name = "program")
plot(x_lab="Algorithms", y_lab="Program Error %",x=stats_data["names"], y_baseline=stats_data["program_error"]["base"], y_qraft=stats_data["program_error"]["qraft"], name = "dominant_states")

# empirical cdf of the state error evalutation and plots 

