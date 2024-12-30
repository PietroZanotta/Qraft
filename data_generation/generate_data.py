import random
from qiskit.providers.fake_provider import Fake5QV1
from circuit_utils import CircuitGenerator
from state_utils import generate_circuit_state

csv_file = "circuit_simulations_data_inal.csv"
n_circ = 2
backend = Fake5QV1()


with open(csv_file, mode="w") as file:
    file.write("circuit_width,circuit_depth,u1,u2,u3,cx,hamming_weight,25_observed_state_prob,50_observed_state_prob,75_observed_state_prob,25_frc_state_error,50_frc_state_error,75_frc_state_error,25_frc_program_error,50_frc_program_error,75_frc_program_error,true_probability,state_name\n")
    circuit_generator = CircuitGenerator()
    
    # Simulation
    for _ in range(n_circ):
        # Define randomly width, depth and gates
        circuit_width = random.randint(1, 5)
        circuit_depth = random.randint(1, 5)
        gates = circuit_generator.get_applicable_gates(num_qubits=circuit_width,depth=circuit_depth)
        
        # Define the circuits
        fc_circuit = circuit_generator.get_fc_circuit(circuit_width, gates)
        frc_circuit = circuit_generator.get_frc_circuit(circuit_width, gates)
        
        # Run simulations
        columns, ext = generate_circuit_state(fc_circuit, frc_circuit, circuit_width, circuit_depth, backend)
        print(ext)
        for column in columns:
            file.write(",".join([str(i) for i in column]) + "\n")

        print(f"Generated {_+1}th circuit data")