from circuit_test import run_cpu
import circuit
circuit.RAM.MEM_SIZE = 5
with circuit.new_circuit() as c:
    run_cpu(c)