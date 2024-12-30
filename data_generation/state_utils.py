import numpy as np
from circuit_utils import CircuitSimulator, State

n_sim = 100

def generate_circuit_state(fc, frc,  width, depth, backend):
  # Create the simulator
  simulator = CircuitSimulator(fc)
  circuit_states = State.generate_states(width)

  # Noiseless backend simulation
  ideal_probablities = simulator.simulate()
  for prob in ideal_probablities:
      circuit_states[prob].set_ideal_probablity(ideal_probablities[prob])

  # Simulate on fake backend and calculate observed probabilites
  for i in range(n_sim):
      results = simulator.simulate(backend)
      for state in circuit_states:
          circuit_states[state].add_run_probability(results.get(state, 0))

  # Compute the run probablity percentile of fc
  for state in circuit_states:
      circuit_states[state].calculate_run_probability_percentile()

  # Simulate for forward reverse circuit
  frc_simulator = CircuitSimulator(frc)
  frc_states = State.generate_states(width)

  for i in range(n_sim):
      results = frc_simulator.simulate(backend)
      for state in frc_states:
          frc_states[state].add_run_probability(
              results.get(state, 0)
          )

  # Set ideal probability of 00000 to 1 rest all states should be zero
  frc_states["0"*width].set_ideal_probablity(1)

  # Calculate the error percentiles for FRC States
  for state in frc_states:
      circuit_states[state].calculate_errors()
      frc_states[state].calculate_errors()
      frc_states[state].calculate_error_percentile()


  # Calculate percentile program error for FRC
  # 0000 - [1,2], 00001 - [3,4], 000010- [5,6]
  frc_program_error = [] # Sum of all the errors at each state per run
  fc_program_error = []
  for i in range(n_sim):
      frc_program_error.append(0)
      fc_program_error.append(0)
      for state in frc_states:
          frc_program_error[i] += frc_states[state].errors[i]
          fc_program_error[i] += circuit_states[state].errors[i]
      frc_program_error[i] = frc_program_error[i]/2 # Update program error formula
      fc_program_error[i] = fc_program_error[i]/2 # Update program error formula
  
  true_state_prob = sum([circuit_states[state].ideal_probablity for state in circuit_states])

  gates_count = fc.count_ops()
  

  columns = []
  extras = {
    "ideal_prob": {},
    "states_errors": {},
    "program_error": sum(fc_program_error)/len(fc_program_error)
  }

  domainant_state = None
  for state in frc_states:
      column = [
          width, depth,gates_count.get("u1", 0), gates_count.get("u2", 0), gates_count.get("u3", 0), gates_count.get("cx", 0), frc_states[state].hamming_weight, # meidan state prob
          circuit_states[state].run_prob_percentile[25]*100, circuit_states[state].run_prob_percentile[50]*100, circuit_states[state].run_prob_percentile[75]*100,
          frc_states[state].error_percentile[25]*100, frc_states[state].error_percentile[50]*100, frc_states[state].error_percentile[75]*100,
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