from circuit import Lines, new_circuit, RAM, Clock, cpu, bootloader_program, simulate
import parser
from pprint import pprint
import re


def tag_outputs(circuit, tags, name_filter=""):
    return Lines(
        l
        for l in sorted(set(circuit.lines.typed(*tags)), key=lambda x: x.name)
        if name_filter in l.name
    )


def simulation(program, frequency=12):
    if isinstance(program, str):
        program = parser.parse(program)
    RAM.MEM_SIZE = 6
    with new_circuit() as circuit:
        clock = Clock()
        output = clock.lines >> cpu
        iars = tag_outputs(circuit, ["BIT"], "Cpu/IAR")
        irs = tag_outputs(circuit, ["BIT"], "Cpu/IR")
        n_rows = 8
        n_cols = 8
        rams = {
            (row, col): tag_outputs(
                circuit, ["BIT"], f"RAMRegister-{row * n_cols + col}/"
            )
            for row in range(n_rows)
            for col in range(n_cols)
        }
        regs = {
            i: tag_outputs(circuit, ["BIT"], f"Cpu/Registers/{i}")
            for i in range(cpu.N_REGISTERS)
        }
        bus = tag_outputs(circuit, ["MAINBUS"])
        stepper = tag_outputs(circuit, ["STEPPER"])
        selectors = tag_outputs(circuit, ["SELECTOR", "CONTROL"])
        enablers = tag_outputs(circuit, ["ENABLER", "CONTROL"])
        mars = tag_outputs(circuit, ["MAROUTPUT"])
        acc = tag_outputs(circuit, ["BIT"], "ACC")
        flags = tag_outputs(circuit, ["FLAG"])
        bootloader_program_ = bootloader_program()
        simulation = {}

        def set_lines(lines, vals, prev=None):
            if prev is None:
                prev = {}
            prev.update({l: v for l, v in zip(lines, vals)})
            return prev

        bl_length = 6

        # Stores the Fibonacci sequence in RAM
        my_program = program
        print("MY PROGRAM")
        pprint(my_program)
        print("BL PROGRAM")
        pprint(bootloader_program_)

        def bootload(program):
            return {
                k: v
                for i, program_line in enumerate(program)
                for k, v in set_lines(
                    rams[(i // n_cols, i % n_cols)], program_line
                ).items()
            }

        fixed_dict = bootload(bootloader_program_)

        def vals(lines, keys=False, show_all=False, decimal=True):
            values = [int(simulation[line]) for line in lines]
            if show_all:
                return [(line.name, value) for line, value in zip(lines, values)]
            if keys:
                return {line.name for line, value in zip(lines, values) if value}
            if len(lines) != 8:
                decimal = True
            d_value = int("".join(map(str, values)), 2)
            if decimal:
                return d_value
            return "{0:02x}".format(d_value)

        io_lines = tag_outputs(circuit, ("IO", ("IO", "IN")))
        out_lines = tag_outputs(circuit, ("IO", ("IO", "OUT")))

        def input_dict(step):
            if step == 0:
                return {
                    line: v
                    for line, v in zip(
                        io_lines, list(map(int, "{0:010b}".format(len(my_program))[2:]))
                    )
                }
            step -= 4
            if step < 0:
                return {}
            step //= bl_length
            if step >= len(my_program):
                return {}
            return {line: v for line, v in zip(io_lines, my_program[step])}

        for i in range(100000000):
            feed_dict = clock.step()
            feed_dict.update(fixed_dict)
            feed_dict.update(input_dict(i // 24))
            simulation.update(simulate(feed_dict, circuit, simulation))
            if i % frequency:
                continue
            # pprint(feed_dict)
            out_enablers = set()
            for line in enablers:
                if simulation[line]:
                    for line_type in line.line_types:
                        if not isinstance(line_type, tuple):
                            continue
                        if line_type[0] == "E":
                            if line_type[1] == 'R':
                                continue
                            out_enablers.add(line_type[1])
                        if line_type[0] == "R":
                            out_enablers.add(f'R{line_type[1]}')
            out_selectors = set()
            for line in selectors:
                if simulation[line]:
                    for line_type in line.line_types:
                        if not isinstance(line_type, tuple):
                            continue
                        if line_type[0] == "S":
                            if line_type[1] == 'R':
                                continue
                            out_selectors.add(line_type[1])
                        if line_type[0] == "R":
                            out_selectors.add(f'R{line_type[1]}')
            yield {
                "iar": vals(iars),
                "ram": {'data':[
                    [vals(rams[(row, col)]) for col in range(n_cols)]
                    for row in range(n_rows)
                ]},
                "ir": {
                    "value": vals(irs),
                    "interpretation": parser.unparse(vals(irs)),
                },
                "registers": [vals(regs[i]) for i in range(len(regs))],
                "steps": {"steps":[bool(simulation[line]) for line in stepper]},
                "flags": {"values":{flag.name[-1]: bool(simulation[flag]) for flag in flags}},
                "input": {
                    "value": vals(io_lines),
                    "interpretation": parser.unparse(vals(io_lines)),
                },
                "bus": {
                    "value": vals(bus),
                    "interpretation": parser.unparse(vals(bus)),
                },
                "enablers": list(out_enablers),
                "selectors": list(out_selectors),
                "output": vals(out_lines),
            }

# for s in simulation(
#         """
#         DATA 0 43; # Starting register to store output in RAM
#         DATA 1 42; # Location of next register to write to
#         ST   1  0; # Store the next output register value in RAM
#         XOR  0  0; # Zero Register 0
#         DATA 1  1; # Store "1" in 1
#         XOR  2  2; # START LOOP(Loc 22) Zero Register 2
#         ADD  0  2; # Reg 2 <- 0 + (Reg 0)
#         ADD  1  2; # Reg 2 <- (Reg 0) + (Reg 1)  [r2 = r0 + r1]
#         XOR  0  0; # Zero Reg 0
#         ADD  1  0; # Reg 0 <- 0 + (Reg 1)   [r0 = r1]
#         XOR  1  1; # Zero Reg 1
#         ADD  2  1; # Reg 1 <- 0 + Reg 2   [r1 = (old r0) + (old r1)]
#         OUT  1   ; # Put Result on OUTPUT
#         DATA 3 42; # Load where to find next register address into 3
#         LD   3  3; # Load the actual next register address into 3
#         ST   3  1; # Store Reg 1 in the next output register
#         DATA 2  1; # Set Reg 2 to 1
#         ADD  2  3; # Add 1 to next register address
#         DATA 2 42; # Load Address where to store next output address into Reg 2
#         ST   2  3; # Replace next register address w/ same value + 1
#         JMP  22  ; # END LOOP(Loc 22)
#     """):
#     pprint(s)