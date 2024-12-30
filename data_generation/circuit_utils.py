import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library.standard_gates import IGate, U1Gate, U2Gate, U3Gate, XGate, YGate, ZGate, HGate, SGate, SdgGate, TGate, TdgGate, RXGate, RYGate, RZGate, CXGate, CYGate, CZGate, CHGate, CRZGate, CU1Gate, CU3Gate, SwapGate, RZZGate, CCXGate
from qiskit import transpile
from qiskit_aer import Aer


class CircuitGenerator:
    def get_applicable_gates(self, num_qubits, depth, max_operands=2, seed=None):
        # Function generating a random circuit using randomly sampled gates
        
        # Set the seed
        if seed is None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)
        rng = np.random.default_rng(seed)
        
        # Classify the gates
        one_q_ops = [U1Gate, U2Gate, U3Gate]
        one_param = [U1Gate]
        two_param = [U2Gate]
        three_param = [U3Gate]
        two_q_ops = [CXGate]

        # Init the quantum circuit
        qr = QuantumRegister(num_qubits)

        # List containing the gates applied to the circuit
        gates_applied = []

        # apply arbitrary random operations at every depth
        for _ in range(depth):
    
            # choose either 1 or 2 qubits for the op
            remaining_qubits = list(range(num_qubits))
            
            while remaining_qubits:
                max_possible_operands = min(len(remaining_qubits), max_operands)
                num_operands = rng.choice(range(max_possible_operands)) + 1
                
                rng.shuffle(remaining_qubits)
                operands = remaining_qubits[:num_operands]
                remaining_qubits = [q for q in remaining_qubits if q not in operands]
                
                # Randomly choose between operations
                if num_operands == 1:
                    op = rng.choice(one_q_ops)
                
                elif num_operands == 2:
                    op = rng.choice(two_q_ops)
                
                # Based on the selected operations find the number of parameters needed
                if op in one_param:
                    num_angles = 1

                elif op in two_param:
                    num_angles = 2

                elif op in three_param:
                    num_angles = 3
                else:
                    num_angles = 0

                # Randomly define the rotation angle
                angles = [rng.uniform(0, 2 * np.pi) for x in range(num_angles)]
                register_operands = [qr[i] for i in operands]

                # Save information regarding the applied gates
                gates_applied.append((op, angles, register_operands))

        return gates_applied
    

    def get_fc_circuit(self, circuit_width, applicable_gates):
        # Return the fc applying the randomly selected gates

        qc = QuantumCircuit(circuit_width)
        for gate in applicable_gates:
            self.apply_gate(qc, gate)
        return qc


    def get_frc_circuit(self, circuit_width, applicable_gates:list):
        # Return the frc applying the randomly selected gates
        
        frc_circuit = self.get_fc_circuit(circuit_width=circuit_width, applicable_gates=applicable_gates)

        # Then reverse the circuit and append the reversed circuit to the forward one
        rc_circuit = frc_circuit.inverse()
        frc_circuit.append(rc_circuit, range(0, frc_circuit.num_qubits))

        return frc_circuit


    def apply_gate(self, qc: QuantumCircuit, gate):
        # Applies the gate op to the input circuit

        op = gate[0]
        angles = gate[1]
        register_operands = gate[2]
        qc.append(op(*angles), register_operands)


class CircuitSimulator:
    def __init__(self, circuit, n_sim=100):
        self.n_sim = n_sim
        self.circuit = circuit
        self.circuit.measure_active()
        

    def simulate(self, backend=None):
        # Simulates the provided circuit
        
        # Define the backend if no backend is given
        if not backend:
            backend = Aer.get_backend('statevector_simulator')
        
        # Transpile the circuit to the appropriate backend and run it n_sim times
        self.circuit = transpile(self.circuit, backend)
        results = backend.run(self.circuit, n_sim=self.n_sim)
        
        # Return the simulation output
        return {key: probability/self.n_sim for key, probability in results.result().get_counts().items()}


class State:
    def __init__(self, name):
        self.name = name
        self.hw = self.get_hw()
        self.ideal_prob = 0
        self.run_probs = []
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
    

    def __repr__(self):
        return f"<State name={self.name} ideal_prob={self.ideal_prob}>"


    def get_hw(self):
        # Function computing the Hamming Weight

        return sum(int(char) for char in self.name)


    def set_ideal_prob(self, ideal_prob):
        # Function setting ideal probabilities

        self.ideal_prob = ideal_prob
    

    def add_run_probability(self, run_prob):
        # Function recording the probabilities associated with different runs

        self.run_probs.append(run_prob)
    

    def calculate_run_probability_percentile(self):
        # Compute statistics of the run

        self.run_prob_percentile = {
            25: np.percentile(self.run_probs, 25),
            50: np.percentile(self.run_probs, 50),
            75: np.percentile(self.run_probs, 75),
        }
    
    
    def calculate_errors(self):
        # Compute statistics of the run

        self.errors = [abs(run_prob - self.ideal_prob)  for run_prob in self.run_probs]
        self.average_error = sum(self.errors)/len(self.errors)


    def calculate_error_percentile(self):
        # Compute statistics of the run

        self.error_percentile = {
            25: np.percentile(self.errors, 25),
            50: np.percentile(self.errors, 50),
            75: np.percentile(self.errors, 75),
        }


    @classmethod
    def generate_states(cls, circuit_width):
        
        def get_binary_combinations(n):
            combinations = []
            for i in range(1 << n):
                combinations.append(format(i, '0' + str(n) + 'b'))
            
            return combinations
        
        return {combination: State(combination) for combination in get_binary_combinations(circuit_width)}