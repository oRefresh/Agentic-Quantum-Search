"""
Infleqtion Superstaq Resource Estimator — Agentic Quantum Search Pipeline

Hardware projection layer for neutral-atom (trapped-atom) quantum computing
architectures. Ingests an OpenQASM 2.0 string and models the physical constraints
of Infleqtion's gate-zone based neutral-atom QPU array:

  - Physical atom count: logical qubits × atoms_per_logical + spare reservoir
  - Atom-shuttling serialization: 2-qubit gates batched by gate-zone capacity
  - Global laser pulses: single-qubit gates applied in parallel layers
  - Wall-clock: shuttle_batches × (shuttle_time + gate_time) + SQ_layers × pulse_overhead

Accepts output from qiskit_optimizer.py (--optimizer-output) or raw QASM directly.

Usage:
  python infleqtion.py --qasm "OPENQASM 2.0; ..."
  python infleqtion.py --qasm-file circuit.qasm
  python infleqtion.py --optimizer-output '{"qasm_string": "...", ...}'
"""

import argparse
import json
import sys

from qiskit import QuantumCircuit


# ─── Neutral-Atom Physical Parameters (Infleqtion / ColdQuanta Architecture) ─────
ATOMS_PER_LOGICAL          = 50    # Physical atoms per logical qubit (surface code on atoms)
SPARE_RESERVOIR_FRACTION   = 0.20  # Extra atom reservoir for loss replacement
GATE_ZONE_CAPACITY         = 10    # Max simultaneous 2-qubit operations per gate zone
ATOM_SHUTTLE_TIME_US       = 50.0  # µs to shuttle atoms between storage and gate zones
TWO_QUBIT_GATE_TIME_US     = 0.5   # µs per 2-qubit gate executed in the gate zone
SINGLE_QUBIT_GATE_TIME_US  = 0.05  # µs per single-qubit gate (global addressed laser)
LASER_PULSE_OVERHEAD_US    = 5.0   # µs overhead per global laser control zone activation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infleqtion Superstaq Resource Estimator — Agentic Quantum Search Pipeline"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--qasm", type=str,
                       help="Raw OpenQASM 2.0 string")
    group.add_argument("--qasm-file", type=str,
                       help="Path to a .qasm file")
    group.add_argument("--optimizer-output", type=str,
                       help="Full JSON blob emitted by qiskit_optimizer.py")
    group.add_argument("--optimizer-output-file", type=str,
                       help="Path to the JSON file emitted by qiskit_optimizer.py (avoids arg-list limits)")
    return parser.parse_args()


def load_circuit(args: argparse.Namespace) -> QuantumCircuit:
    if args.optimizer_output_file:
        with open(args.optimizer_output_file) as f:
            data = json.loads(f.read())
        return QuantumCircuit.from_qasm_str(data["qasm_string"])
    if args.optimizer_output:
        data = json.loads(args.optimizer_output)
        return QuantumCircuit.from_qasm_str(data["qasm_string"])
    if args.qasm_file:
        with open(args.qasm_file) as f:
            return QuantumCircuit.from_qasm_str(f.read())
    return QuantumCircuit.from_qasm_str(args.qasm)


def estimate_neutral_atom_resources(qc: QuantumCircuit) -> dict:
    """
    Model physical resource requirements for Infleqtion's gate-zone neutral-atom QPU.

    Gate execution model:
    - 2-qubit gates require atoms to be physically shuttled into a gate zone.
      Gate zones have limited capacity (GATE_ZONE_CAPACITY simultaneous operations).
      Gates are batched into shuttle rounds; each round incurs shuttle + gate time.
    - Single-qubit gates are applied via global laser pulses that address all atoms
      in parallel, grouped into layers by circuit depth analysis.
    - The two timing contributions add sequentially (shuttle phases ≠ laser phases).

    Physical qubit model:
    - Each logical qubit requires ATOMS_PER_LOGICAL physical atoms for error correction.
    - An additional spare reservoir compensates for atom loss during computation.
    """
    ops = qc.count_ops()
    n_logical    = qc.num_qubits
    circuit_depth = qc.depth()

    t_count = ops.get("t", 0) + ops.get("tdg", 0)

    two_qubit_ops = sum(
        v for k, v in ops.items()
        if k in {"cx", "cz", "swap", "ccx", "ecr", "rzz", "rxx"}
    )
    single_qubit_ops = sum(
        v for k, v in ops.items()
        if k in {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "u1", "u2", "u3", "p", "rx", "ry", "rz", "id"}
    )
    clifford_gates = single_qubit_ops + two_qubit_ops - t_count

    # Physical atom count
    physical_qubits = int(n_logical * ATOMS_PER_LOGICAL * (1.0 + SPARE_RESERVOIR_FRACTION))

    # Atom-shuttling model: 2-qubit gates are executed in batches of GATE_ZONE_CAPACITY
    shuttle_batches = max(1, -(-two_qubit_ops // GATE_ZONE_CAPACITY))  # ceiling division
    shuttle_time_us = shuttle_batches * (ATOM_SHUTTLE_TIME_US + TWO_QUBIT_GATE_TIME_US)

    # Single-qubit layer model: layers = circuit depth minus 2-qubit gate layers
    single_qubit_layers = max(1, circuit_depth - shuttle_batches)
    single_qubit_time_us = single_qubit_layers * (
        SINGLE_QUBIT_GATE_TIME_US + LASER_PULSE_OVERHEAD_US
    )

    wall_clock_us = shuttle_time_us + single_qubit_time_us
    wall_clock_s  = wall_clock_us * 1e-6
    space_time_volume = n_logical * circuit_depth

    return {
        "logical_qubits":      n_logical,
        "physical_qubits":     physical_qubits,
        "t_gates":             t_count,
        "clifford_gates":      clifford_gates,
        "two_qubit_ops":       two_qubit_ops,
        "single_qubit_ops":    single_qubit_ops,
        "shuttle_batches":     shuttle_batches,
        "shuttle_time_us":     shuttle_time_us,
        "single_qubit_time_us": single_qubit_time_us,
        "wall_clock_us":       wall_clock_us,
        "wall_clock_s":        wall_clock_s,
        "space_time_volume":   space_time_volume,
    }


def main() -> None:
    args = parse_args()

    try:
        qc = load_circuit(args)
    except Exception as e:
        print(json.dumps({"error": f"Circuit load failed: {e}"}), file=sys.stderr)
        sys.exit(1)

    r = estimate_neutral_atom_resources(qc)

    # When the optimizer JSON is the source, prefer its synthesis-corrected t_count over
    # the raw QASM re-count (which misses rotation→T-gate synthesis cost for Hamiltonian circuits).
    if args.optimizer_output_file or args.optimizer_output:
        try:
            if args.optimizer_output_file:
                with open(args.optimizer_output_file) as f:
                    opt = json.loads(f.read())
            else:
                opt = json.loads(args.optimizer_output)
            r["t_gates"] = opt.get("t_count", r["t_gates"])
        except Exception:
            pass

    output = {
        "estimator": "Infleqtion Superstaq (Neutral-Atom Gate-Zone Architecture)",
        "architecture": {
            "type": "Neutral-atom array with dedicated gate zones",
            "atoms_per_logical_qubit": ATOMS_PER_LOGICAL,
            "spare_reservoir_fraction": SPARE_RESERVOIR_FRACTION,
            "gate_zone_capacity": GATE_ZONE_CAPACITY,
            "atom_shuttle_time_us": ATOM_SHUTTLE_TIME_US,
            "two_qubit_gate_time_us": TWO_QUBIT_GATE_TIME_US,
            "single_qubit_gate_time_us": SINGLE_QUBIT_GATE_TIME_US,
            "laser_pulse_overhead_us": LASER_PULSE_OVERHEAD_US,
        },
        "logical_qubits":    r["logical_qubits"],
        "physical_qubits":   r["physical_qubits"],
        "t_gates":           r["t_gates"],
        "clifford_gates":    r["clifford_gates"],
        "space_time_volume": r["space_time_volume"],
        "wall_clock_time":   r["wall_clock_s"],
        "breakdown": {
            "two_qubit_ops":        r["two_qubit_ops"],
            "single_qubit_ops":     r["single_qubit_ops"],
            "shuttle_batches":      r["shuttle_batches"],
            "shuttle_time_us":      r["shuttle_time_us"],
            "single_qubit_time_us": r["single_qubit_time_us"],
            "wall_clock_us":        r["wall_clock_us"],
        },
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
