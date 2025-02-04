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
        

        if max_operands < 1 or max_operands > 2:
            raise CircuitError("max_operands must be between 1 and 2")

        one_q_ops = [U1Gate, U2Gate, U3Gate] 
        one_param = [U1Gate] 
        two_param = [U2Gate]
        three_param = [U3Gate] 
        two_q_ops = [CXGate] 

        qr = QuantumRegister(num_qubits, 'q')

        gates_applied = []
        
        # apply arbitrary random operations at every depth
        for _ in range(depth):
            # choose either 1 or 2 qubits for the operation NOTE: Excluded 3 qubit operations on purpose
            remaining_qubits = list(range(num_qubits))

            while remaining_qubits:
                max_possible_operands = min(len(remaining_qubits), max_operands)
                num_operands = rng.choice(range(max_possible_operands)) + 1
                rng.shuffle(remaining_qubits)
                operands = remaining_qubits[:num_operands]
                remaining_qubits = [q for q in remaining_qubits if q not in operands]

                if num_operands == 1:
                    operation = rng.choice(one_q_ops)

                elif num_operands == 2:
                    operation = rng.choice(two_q_ops)

                if operation in one_param:
                    num_angles = 1

                elif operation in two_param:
                    num_angles = 2

                elif operation in three_param:
                    num_angles = 3

                else:
                    num_angles = 0

                angles = [rng.uniform(0, 2 * np.pi) for x in range(num_angles)]
                register_operands = [qr[i] for i in operands]

                gates_applied.append((operation, angles, register_operands))


        return gates_applied
    
    def get_fc_circuit(self, circuit_width, applicable_gates):
        # Return the fc applying the randomly selected gates

        qc = QuantumCircuit(circuit_width)
        for gate in applicable_gates:
            # print(gate)
            # print(qc)
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
    n_sim = 100
    def __init__(self, circuit) -> None:
        self.circuit = circuit
        self.circuit.measure_active()

    def simulate(self, backend=None):
        # Simulates the provided circuit
        if not backend:
            # define the backend if no backend is given
            backend = Aer.get_backend('statevector_simulator')
        
        # transpile the circuit to the appropriate backend and run it n_sim times
        self.circuit = transpile(self.circuit, backend)
        results = backend.run(self.circuit, shots=self.n_sim)
        
        return {key: probability/self.n_sim for key, probability in results.result().get_counts().items()}
        