# generates circuits used to train the model and saves them in a csv file
import random
from qiskit.providers.fake_provider import Fake5QV1
from circuit import CircuitGenerator
from state import generate_circuit_state

def gen_data(csv_file_name, n_circ):
    fake_backend = Fake5QV1()

    with open(csv_file_name, mode="w") as generation_file:
        # Write headers to file
        generation_file.write("circuit_width,circuit_depth,u1,u2,u3,cx,hamming_weight,25_observed_state_prob,50_observed_state_prob,75_observed_state_prob,25_frc_state_error,50_frc_state_error,75_frc_state_error,25_frc_program_error,50_frc_program_error,75_frc_program_error,true_probability,state_name\n")
        circuit_generator = CircuitGenerator()
        
        # simulation
        for i in range(n_circ):
            # define randomly width, depth and gates
            circuit_width = random.randint(1, 5)
            circuit_depth = random.randint(1, 5)
            gates = circuit_generator.get_applicable_gates(num_qubits=circuit_width,depth=circuit_depth)

            # define the circuits
            fc_circuit = circuit_generator.get_fc_circuit(circuit_width, gates)
            frc_circuit = circuit_generator.get_frc_circuit(circuit_width, gates)
            
            # run simulations
            columns, ext = generate_circuit_state(fc_circuit, frc_circuit, circuit_width, circuit_depth, fake_backend)
            print(ext)
            for column in columns:
                generation_file.write(",".join([str(token) for token in column]) + "\n")

            if i % 10 == 0:
                print(str(i) + "-th circuit")
