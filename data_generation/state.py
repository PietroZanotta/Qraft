import numpy as np
import numpy as np
from circuit import CircuitSimulator
from qiskit_aer import Aer

class State:
    def __init__(self, name):
        self.name = name
        self.hw = self.get_hw()
        self.ideal_probablity = 0
        self.run_probablities = []
        self.errors = []
        self.run_prob_percentile = {
            25: 0,
            50: 0,
            75: 0
        }
        self.error_percentile = {
            25: 0,
            50: 0,
            75: 0
        }
        self.average_state_error = 0
    
    def __repr__(self) -> str:
        return f"<State name={self.name} ideal_prob={self.ideal_probablity}>"

    def get_hw(self):
        # return the hamming weight of the state
        return sum(int(char) for char in self.name)

    def set_ideal_probablity(self, ideal_prob):
        self.ideal_probablity = ideal_prob
    
    def add_run_probability(self, run_prob):
        self.run_probablities.append(run_prob)
    
    def calculate_run_probability_percentile(self):
        self.run_prob_percentile = {
            25: np.percentile(self.run_probablities, 25),
            50: np.percentile(self.run_probablities, 50),
            75: np.percentile(self.run_probablities, 75),
        }
    
    def calculate_run_probability_percentile(self):
        self.run_prob_percentile = {
            25: np.percentile(self.run_probablities, 25),
            50: np.percentile(self.run_probablities, 50),
            75: np.percentile(self.run_probablities, 75),
        }
    
    def calculate_errors(self):
        self.errors = [abs(run_prob - self.ideal_probablity)  for run_prob in self.run_probablities]
        self.average_error = sum(self.errors)/len(self.errors)

    def calculate_error_percentile(self):
        self.error_percentile = {
            25: np.percentile(self.errors, 25),
            50: np.percentile(self.errors, 50),
            75: np.percentile(self.errors, 75),
        }

    @classmethod
    def generate_states(cls, circuit_width) -> dict:
        
        def get_binary_combinations(n):
            # generates all possible binary strings of length n

            combinations = []
            for i in range(1 << n):
                # Convert the current number to a binary string of length n
                combinations.append(format(i, '0' + str(n) + 'b'))
            return combinations
    
        return {combination: State(combination) for combination in get_binary_combinations(circuit_width)}
        

n_sim = 100

def generate_circuit_state(fc_circuit, frc_circuit,  circuit_width, circuit_depth, backend):
  # forward circuits
  simulator = CircuitSimulator(fc_circuit)
  circuit_states = State.generate_states(circuit_width)

  # ideal backend simulation
  ideal_probablities = simulator.simulate()
  for prob in ideal_probablities:
      circuit_states[prob].set_ideal_probablity(ideal_probablities[prob])

  # simulation on fake backend
  for i in range(n_sim):
      results = simulator.simulate(backend)

      # computing simulated fake probabilities
      for state in circuit_states:
          circuit_states[state].add_run_probability(results.get(state, 0))

  # calculate the run probablity perc of fc_circuit
  for state in circuit_states:
      circuit_states[state].calculate_run_probability_percentile()

  # forward reverse circuit
  frc_simulator = CircuitSimulator(frc_circuit)
  frc_circuit_states = State.generate_states(circuit_width)

  # simulation on fake backend
  for i in range(n_sim):
      results = frc_simulator.simulate(backend)

      # computing simulated fake probabilities
      for state in frc_circuit_states:
          frc_circuit_states[state].add_run_probability(results.get(state, 0))

  # init the prob of 00000 to 1. all the other states are 0
  frc_circuit_states["0"*circuit_width].set_ideal_probablity(1)

  # computing error percs for forward reverse circuit states
  for state in frc_circuit_states:
      circuit_states[state].calculate_errors()
      frc_circuit_states[state].calculate_errors()
      frc_circuit_states[state].calculate_error_percentile()


    # computing perc program error for forward reverse circuit
  frc_program_error = []
  fc_program_error = []
  for i in range(n_sim):
      # print(i)
      frc_program_error.append(0)
      fc_program_error.append(0)
      for state in frc_circuit_states:
          frc_program_error[i] += frc_circuit_states[state].errors[i]
          fc_program_error[i] += circuit_states[state].errors[i]
      frc_program_error[i] = frc_program_error[i]/2 
      fc_program_error[i] = fc_program_error[i]/2 
  
  true_state_prob = sum([circuit_states[state].ideal_probablity for state in circuit_states])

  gates_count = fc_circuit.count_ops()
  

  columns = []
  extras = {
    "ideal_prob": {},
    "states_errors": {},
    "program_error": sum(fc_program_error)/len(fc_program_error)
  }

  domainant_state = None
  for state in frc_circuit_states:
      column = [
          circuit_width, circuit_depth,gates_count.get("u1", 0), gates_count.get("u2", 0), gates_count.get("u3", 0), gates_count.get("cx", 0), frc_circuit_states[state].hw, # meidan state prob
          circuit_states[state].run_prob_percentile[25]*100, circuit_states[state].run_prob_percentile[50]*100, circuit_states[state].run_prob_percentile[75]*100,
          frc_circuit_states[state].error_percentile[25]*100, frc_circuit_states[state].error_percentile[50]*100, frc_circuit_states[state].error_percentile[75]*100,
          np.percentile(frc_program_error, 25)*100, np.percentile(frc_program_error, 50)*100, np.percentile(frc_program_error, 75)*100,  (circuit_states[state].ideal_probablity/true_state_prob)*100, state
      ]
      extras["ideal_prob"][state] = circuit_states[state].ideal_probablity
      extras["states_errors"][state] = circuit_states[state].average_error
      if not domainant_state:
        domainant_state = circuit_states[state]
      else:
        domainant_state = circuit_states[state] if circuit_states[state].ideal_probablity > domainant_state.ideal_probablity else domainant_state

      columns.append(column)

  extras["dominant_state_error"] = domainant_state.average_error
  extras["dominant_state"] = domainant_state.name
  
  return columns, extras