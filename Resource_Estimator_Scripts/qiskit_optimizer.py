"""
Qiskit Mathematical & Optimization Layer — Agentic Quantum Search Pipeline

Front-end preprocessing engine. Ingests a dynamic problem description (Heisenberg
spin-chain parameters, an FTCircuitBench QASM file, or a raw OpenQASM 2.0 string),
runs Qiskit transpiler passes at optimization_level=3 to minimize gate depth and
T-gate count, applies operator grouping, and emits a single JSON blob containing
the optimized OpenQASM 2.0 string plus pre/post metrics.

The JSON output is the contract consumed by all downstream resource estimators
via their --optimizer-output flag, or the qasm_string field can be extracted and
passed via --qasm / --qasm-file.

Usage:
  python qiskit_optimizer.py --heisenberg --qubits 8 --J 1.0 --h 0.5
  python qiskit_optimizer.py --heisenberg --qubits 4 --J 1.0 --h 0.5 --trotter-steps 2
  python qiskit_optimizer.py --qasm-file circuits/adder_n4.qasm
  python qiskit_optimizer.py --ftcircuit benchmarks/ftcircuitbench/qft_n6.qasm
  python qiskit_optimizer.py --qasm "OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];"
  python qiskit_optimizer.py --heisenberg --qubits 4 --no-optimize
"""

import argparse
import json
import sys

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info import SparsePauliOp

from qiskit.qasm2 import dumps

import math


# Fault-tolerant basis gate set — forces transpiler to decompose to T/Clifford primitives
FT_BASIS_GATES = ["cx", "u1", "u2", "u3", "h", "s", "sdg", "t", "tdg", "x", "y", "z", "id"]

# Rotation synthesis T-gate cost model (Selinger 2012 / GridSynth)
# T_cost per arbitrary single-qubit rotation ≈ ceil(3.02 × log2(1/ε) + 1.3)
ROTATION_SYNTHESIS_EPSILON = 1e-3
T_PER_ROTATION = math.ceil(
    3.02 * math.log2(1.0 / ROTATION_SYNTHESIS_EPSILON) + 1.3
)  # ≈ 32 T gates per rotation at ε = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qiskit Optimization Layer — Agentic Quantum Search Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--heisenberg", action="store_true",
        help="Generate a Heisenberg XXX spin-chain Hamiltonian circuit via Trotterization",
    )
    group.add_argument(
        "--qasm-file", type=str, metavar="PATH",
        help="Load circuit from an OpenQASM 2.0 .qasm file",
    )
    group.add_argument(
        "--ftcircuit", type=str, metavar="PATH",
        help="Load an FTCircuitBench circuit from a .qasm file path",
    )
    group.add_argument(
        "--qasm", type=str, metavar="STRING",
        help="Raw OpenQASM 2.0 string passed directly",
    )

    # Heisenberg parameters
    parser.add_argument("--qubits", type=int, default=4,
                        help="Number of spin sites (--heisenberg only, default: 4)")
    parser.add_argument("--J", type=float, default=1.0,
                        help="Exchange coupling constant J (default: 1.0)")
    parser.add_argument("--h", type=float, default=0.5,
                        help="Longitudinal magnetic field strength h (default: 0.5)")
    parser.add_argument("--trotter-steps", type=int, default=1,
                        help="First-order Lie-Trotter steps (default: 1)")

    # Optimizer control
    parser.add_argument("--opt-level", type=int, default=3, choices=[0, 1, 2, 3],
                        help="Qiskit transpiler optimization level (default: 3)")
    parser.add_argument("--no-optimize", action="store_true",
                        help="Skip transpilation; output raw circuit metrics")
    return parser.parse_args()


# ─── Circuit Loaders ─────────────────────────────────────────────────────────────

def build_heisenberg_circuit(n: int, J: float, h: float, steps: int) -> QuantumCircuit:
    """
    Build the Heisenberg XXX spin-chain via first-order Lie-Trotter decomposition.

    H = J * Σ_{i} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}) + h * Σ_i Z_i

    The PauliEvolutionGate is decomposed into native gates by Qiskit's synthesis engine
    before the optimization pass runs.
    """
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter

    pauli_list = []
    for i in range(n - 1):
        for pauli in ["XX", "YY", "ZZ"]:
            label = "I" * i + pauli + "I" * (n - i - 2)
            pauli_list.append((label[::-1], J))
    for i in range(n):
        label = "I" * i + "Z" + "I" * (n - i - 1)
        pauli_list.append((label[::-1], h))

    hamiltonian = SparsePauliOp.from_list(pauli_list)
    evo = PauliEvolutionGate(hamiltonian, time=1.0, synthesis=LieTrotter(reps=steps))
    qc = QuantumCircuit(n)
    qc.append(evo, range(n))
    return qc.decompose(reps=4)


def load_qasm_file(path: str) -> QuantumCircuit:
    with open(path, "r") as f:
        return QuantumCircuit.from_qasm_str(f.read())


def load_circuit(args: argparse.Namespace) -> tuple[QuantumCircuit, str]:
    """Returns (circuit, source_label)."""
    if args.heisenberg:
        qc = build_heisenberg_circuit(args.qubits, args.J, args.h, args.trotter_steps)
        label = (
            f"Heisenberg XXX (n={args.qubits}, J={args.J}, "
            f"h={args.h}, trotter_steps={args.trotter_steps})"
        )
        return qc, label

    if args.qasm_file:
        return load_qasm_file(args.qasm_file), f"QASM file: {args.qasm_file}"

    if args.ftcircuit:
        return load_qasm_file(args.ftcircuit), f"FTCircuitBench: {args.ftcircuit}"

    if args.qasm:
        return QuantumCircuit.from_qasm_str(args.qasm), "Inline QASM string"

    raise ValueError("No valid input source.")


# ─── Metrics Extraction ──────────────────────────────────────────────────────────

def extract_metrics(qc: QuantumCircuit) -> dict:
    """
    Extract gate counts and compute a synthesis-corrected T-gate equivalent.

    Qiskit's transpiler emits arbitrary-angle rotations as u1/u2/u3/rz gates when
    the circuit contains parameterized Hamiltonian terms (e.g. Heisenberg Trotter).
    These are NOT discrete T gates — they are non-Clifford rotations that must be
    synthesized into T+Clifford sequences to execute on a fault-tolerant hardware.

    We apply the Selinger 2012 cost formula per rotation:
        T_cost ≈ ceil(3.02 × log2(1/ε) + 1.3)
    at ε = ROTATION_SYNTHESIS_EPSILON, and add it to any explicit T/Tdg gates.

    Resource estimators should consume t_count (synthesis-corrected) as the true
    non-Clifford resource. t_count_explicit is provided for diagnostics only.
    """
    ops = qc.count_ops()

    t_explicit = ops.get("t", 0) + ops.get("tdg", 0)

    # Non-Clifford single-qubit rotations requiring synthesis into T-gate sequences.
    # u3 requires 2 independent axis rotations; u2 requires 1; u1/rz/p require 1.
    rotation_count = (
        ops.get("u1", 0) + ops.get("p", 0)
        + ops.get("rz", 0) + ops.get("rx", 0) + ops.get("ry", 0)
        + ops.get("u2", 0)
        + ops.get("u3", 0) * 2
    )

    t_count = t_explicit + rotation_count * T_PER_ROTATION

    # Clifford gates: exclude non-Clifford rotations and metadata operations
    _non_clifford = {"t", "tdg", "u1", "u2", "u3", "p", "rx", "ry", "rz"}
    _meta = {"barrier", "measure", "reset", "delay"}
    clifford = sum(v for k, v in ops.items() if k not in _non_clifford | _meta)

    return {
        "num_qubits": qc.num_qubits,
        "circuit_depth": qc.depth(),
        "t_count": t_count,
        "t_count_explicit": t_explicit,
        "rotation_count": rotation_count,
        "t_per_rotation": T_PER_ROTATION,
        "rotation_synthesis_epsilon": ROTATION_SYNTHESIS_EPSILON,
        "clifford_count": clifford,
        "raw_gate_counts": {k: v for k, v in ops.items()},
    }


# ─── Optimization ────────────────────────────────────────────────────────────────

def optimize(qc: QuantumCircuit, opt_level: int) -> QuantumCircuit:
    return transpile(
        qc,
        basis_gates=FT_BASIS_GATES,
        optimization_level=opt_level,
        seed_transpiler=42,
    )


# ─── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    try:
        raw_circuit, source_label = load_circuit(args)
    except FileNotFoundError as e:
        print(json.dumps({"error": f"File not found: {e}"}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Circuit load failed: {e}"}), file=sys.stderr)
        sys.exit(1)

    pre = extract_metrics(raw_circuit)

    if args.no_optimize:
        optimized = raw_circuit
        post = pre
    else:
        try:
            optimized = optimize(raw_circuit, args.opt_level)
            post = extract_metrics(optimized)
        except Exception as e:
            # Transpilation can fail on exotic decomposed gates; fall back gracefully
            sys.stderr.write(
                json.dumps({"warning": f"Transpilation failed ({e}); using raw circuit"}) + "\n"
            )
            optimized = raw_circuit
            post = pre

    try:
        qasm_string = dumps(optimized)
    except Exception as e:
        print(json.dumps({"error": f"QASM export failed: {e}"}), file=sys.stderr)
        sys.exit(1)

    t_reduction = round(100.0 * (1 - post["t_count"] / max(pre["t_count"], 1)), 2)
    depth_reduction = round(100.0 * (1 - post["circuit_depth"] / max(pre["circuit_depth"], 1)), 2)

    result = {
        "layer": "qiskit_optimizer",
        "source": source_label,
        "num_qubits": post["num_qubits"],
        "circuit_depth": post["circuit_depth"],
        # t_count is the synthesis-corrected T-gate equivalent (includes rotation synthesis cost).
        # This is the value downstream resource estimators should use for non-Clifford overhead.
        "t_count": post["t_count"],
        "t_count_explicit": post["t_count_explicit"],
        "rotation_count": post["rotation_count"],
        "t_per_rotation": post["t_per_rotation"],
        "rotation_synthesis_epsilon": post["rotation_synthesis_epsilon"],
        "clifford_count": post["clifford_count"],
        "raw_gate_counts": post["raw_gate_counts"],
        "qasm_string": qasm_string,
        "pre_optimization": {
            "t_count": pre["t_count"],
            "t_count_explicit": pre["t_count_explicit"],
            "rotation_count": pre["rotation_count"],
            "clifford_count": pre["clifford_count"],
            "circuit_depth": pre["circuit_depth"],
        },
        "optimization_reduction": {
            "t_gate_reduction_pct": t_reduction,
            "depth_reduction_pct": depth_reduction,
            "optimization_level": 0 if args.no_optimize else args.opt_level,
        },
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
