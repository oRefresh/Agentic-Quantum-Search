"""
Superstaq Native Resource Estimator — Agentic Quantum Search Pipeline

Hardware projection layer using the resource-superstaq library (v0.0.1) for
surface-code quantum computing on superconducting architectures. Ingests the
Qiskit preprocessor JSON (optimized QASM + pre-computed gate counts) and maps
outputs to the standardized schema used by all estimators in this pipeline.

Library: resource_estimation.ftqc  (package: resource-superstaq 0.0.1)
  Architecture:  Superconductor(d=code_distance)   — lattice-surgery SC model
  Estimator:     ResourceEstimator(arch)
  Primitives:    Cultivate(pi/4) per T state, cirq.I for idle algo qubits

Resource extraction pipeline:
  1. Parse optimizer JSON → extract t_count, num_qubits, circuit_depth, clifford_count
  2. Create Superconductor(d) architecture and ResourceEstimator
  3. Build a synthetic primitive circuit from the gate counts:
       - num_factories factory qubits, each with ceil(t_count/num_factories)
         sequential Cultivate(pi/4) operations (one per T state produced)
       - n_algo_logical algorithm qubits with cirq.I (zero-cost idle, establishes
         qubit count for est.physical_qubits)
  4. Run est.parallel_circuit_time(primitive_circuit) → wall_clock_us
     Run est.physical_qubits(primitive_circuit)       → total_physical_qubits
  5. Derive all mapped variables from the library output

Variable mapping (all computed from resource-superstaq library):
  qubits_per_logical      = arch.patch.num_physical_qubits             (= 2d²-1)
  data_physical_qubits    = n_algo_logical × qubits_per_logical
  n_factory_logical       = num_factories   (one logical slot per factory)
  factory_physical_qubits = n_factory_logical × qubits_per_logical
  n_t_factories           = num_factories   (input parameter)
  factory_qubits_each     = qubits_per_logical  (one logical slot per factory)
  total_physical_qubits   = est.physical_qubits(primitive_circuit)
  wall_clock_us           = est.parallel_circuit_time(primitive_circuit)
  wall_clock_s            = wall_clock_us × 1e-6
  cycle_time_us           = arch.op_time(SyndromeExtract(1, arch.rounds).on(q))
  total_cycles            = wall_clock_us / cycle_time_us
  algorithm_cycles        = total_cycles   (T factories run in parallel → same path)
  t_distillation_cycles   = arch._cultivate_t_cost["op_time"] / cycle_time_us
  space_time_volume       = total_physical_qubits × total_cycles    (qubit·cycles)

Usage:
  python superstaq_estimator.py --qasm "OPENQASM 2.0; ..."
  python superstaq_estimator.py --qasm-file circuit.qasm
  python superstaq_estimator.py --optimizer-output '{"qasm_string": "...", ...}'
  python superstaq_estimator.py --optimizer-output-file optimizer_out.json
  python superstaq_estimator.py --optimizer-output-file optimizer_out.json --num-factories 10
  python superstaq_estimator.py --optimizer-output-file optimizer_out.json --code-distance 15
"""

import argparse
import json
import math
import sys
from math import pi
from typing import Optional

import cirq
from qiskit import QuantumCircuit
from resource_estimation.ftqc import ResourceEstimator, Superconductor
from resource_estimation.ftqc.lattice_surgery_primitives import Cultivate, SyndromeExtract


# ─── Default Estimation Parameters ───────────────────────────────────────────────
_DEFAULT_NUM_FACTORIES = 20
_DEFAULT_CODE_DISTANCE  = 7   # Superconductor default; determines 2d²-1 physical qubits per logical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Superstaq Native Resource Estimator (resource-superstaq, SC architecture)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--qasm", type=str,
                       help="Raw OpenQASM 2.0 string")
    group.add_argument("--qasm-file", type=str,
                       help="Path to a .qasm file")
    group.add_argument("--optimizer-output", type=str,
                       help="Full JSON blob emitted by qiskit_optimizer.py")
    group.add_argument("--optimizer-output-file", type=str,
                       help="Path to the JSON file emitted by qiskit_optimizer.py")
    parser.add_argument("--num-factories", type=int, default=_DEFAULT_NUM_FACTORIES,
                        help=f"Number of parallel T factories (default: {_DEFAULT_NUM_FACTORIES})")
    parser.add_argument("--code-distance", type=int, default=_DEFAULT_CODE_DISTANCE,
                        help=f"Surface code distance d; sets 2d²-1 physical qubits per logical (default: {_DEFAULT_CODE_DISTANCE})")
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[QuantumCircuit, dict]:
    """
    Returns (qc, optimizer_data).

    optimizer_data carries pre-computed fields (t_count, num_qubits, clifford_count,
    circuit_depth) from qiskit_optimizer.py. For raw QASM inputs, optimizer_data is
    an empty dict and those fields are computed from the parsed circuit.
    """
    if args.optimizer_output_file:
        with open(args.optimizer_output_file) as f:
            opt = json.load(f)
        return QuantumCircuit.from_qasm_str(opt["qasm_string"]), opt
    if args.optimizer_output:
        opt = json.loads(args.optimizer_output)
        return QuantumCircuit.from_qasm_str(opt["qasm_string"]), opt
    if args.qasm_file:
        with open(args.qasm_file) as f:
            return QuantumCircuit.from_qasm_str(f.read()), {}
    return QuantumCircuit.from_qasm_str(args.qasm), {}


def _get_t_distillation_cycles(arch: Superconductor, cycle_time_us: float) -> Optional[float]:
    """
    Return t_distillation_cycles = arch._cultivate_t_cost["op_time"] / cycle_time_us.

    Tries _cultivate_t_cost first (CCZ-based cultivation, default for SC), then
    _distil_t_cost (15-to-1 fallback). Returns None if neither attribute is set.
    """
    for attr in ("_cultivate_t_cost", "_distil_t_cost"):
        cost = getattr(arch, attr, None)
        if cost is None:
            continue
        op_time = (
            cost.get("op_time") if isinstance(cost, dict)
            else getattr(cost, "op_time", None)
        )
        if op_time is not None:
            return float(op_time) / cycle_time_us
    return None


def estimate_resources(
    qc: QuantumCircuit,
    opt: dict,
    num_factories: int = _DEFAULT_NUM_FACTORIES,
    code_distance: int = _DEFAULT_CODE_DISTANCE,
) -> dict:
    """
    Compute physical resource estimates using resource_estimation.ftqc.

    Pipeline:
      1. Create Superconductor(d=code_distance) and ResourceEstimator.
      2. Build a synthetic primitive circuit from optimizer gate counts:
           - num_factories factory qubits with ceil(t_count/num_factories) sequential
             Cultivate(pi/4) operations (one per T state consumed by the algorithm).
           - n_algo_logical algorithm qubits with cirq.I (zero-cost; establishes qubit
             count for est.physical_qubits without adding to the critical path).
         Cirq automatically places independent qubit ops into the same moment,
         so factory and algo qubits run in parallel — the critical path is factory time.
      3. Call est.parallel_circuit_time(primitive_circuit) → wall_clock_us
         Call est.physical_qubits(primitive_circuit)       → total_physical_qubits
      4. Derive all mapped variables and build the standardized output dict.

    The parallel model is exact for the T-factory bottleneck regime (which is almost
    always the case for fault-tolerant circuits with large T counts).
    """
    # ── Step 1: Architecture and estimator ────────────────────────────────────────
    arch = Superconductor(d=code_distance)
    est  = ResourceEstimator(arch)

    # Physical qubits per logical qubit for a rotated surface code patch: 2d²-1
    qubits_per_logical = int(arch.patch.num_physical_qubits)

    # Cycle time: time (µs) for one full SyndromeExtract on 1 logical qubit using arch.rounds rounds.
    # arch.rounds = code_distance (default syndrome_rounds=None → rounds=d).
    _se_qubit    = cirq.GridQubit(0, 0)
    cycle_time_us = arch.op_time(SyndromeExtract(1, arch.rounds).on(_se_qubit))

    # T factory cost: time (µs) for one cultivation of one T state
    t_distillation_cycles_val = _get_t_distillation_cycles(arch, cycle_time_us)
    if t_distillation_cycles_val is None:
        raise RuntimeError(
            "resource-superstaq: could not read _cultivate_t_cost or _distil_t_cost from arch"
        )
    t_distillation_cycles = t_distillation_cycles_val
    t_cultivate_us = t_distillation_cycles * cycle_time_us

    # ── Step 2: Logical-level counts from optimizer JSON ──────────────────────────
    ops            = qc.count_ops()
    t_gates        = opt.get("t_count",        ops.get("t", 0) + ops.get("tdg", 0))
    clifford_gates = opt.get("clifford_count", 0)
    circuit_depth  = opt.get("circuit_depth",  qc.depth())
    n_algo_logical = opt.get("num_qubits",     qc.num_qubits)

    # ── Step 3: Build synthetic primitive circuit ──────────────────────────────────
    # Factory qubits: ceil(t_count / num_factories) sequential Cultivate ops each.
    # Independent qubits execute in parallel → Cirq places them in the same moments.
    n_factory_logical = num_factories
    algo_qubits    = [cirq.GridQubit(i, 0) for i in range(n_algo_logical)]
    factory_qubits = [cirq.GridQubit(n_algo_logical + i, 0) for i in range(n_factory_logical)]

    t_per_factory = math.ceil(t_gates / num_factories) if num_factories > 0 and t_gates > 0 else 1

    factory_ops = [
        Cultivate(pi / 4).on(fq)
        for fq in factory_qubits
        for _ in range(t_per_factory)
    ]
    # Algorithm qubits contribute to the physical qubit count but add zero time:
    # cirq.I has empty moment_cost, so op_time = 0 for the I gate on Superconductor.
    algo_ops = [cirq.I.on(aq) for aq in algo_qubits]

    primitive_circuit = cirq.Circuit(factory_ops + algo_ops)

    # ── Step 4: Library calls ─────────────────────────────────────────────────────
    # parallel_circuit_time: per-qubit timeline; max across all qubits is the result.
    # Factory qubits dominate (each has t_per_factory × t_cultivate_us of sequential work);
    # algo qubits contribute 0 (I gate). Factories run concurrently, so cost = one factory's path.
    wall_clock_us         = est.parallel_circuit_time(primitive_circuit)
    total_physical_qubits = est.physical_qubits(primitive_circuit)   # (n_algo + n_fac) × (2d²-1)

    # ── Step 5: Derived variables per the mapping table ───────────────────────────
    wall_clock_s     = wall_clock_us * 1e-6
    total_cycles     = wall_clock_us / cycle_time_us
    algorithm_cycles = total_cycles  # T factories run in parallel → same critical path

    data_physical_qubits    = n_algo_logical    * qubits_per_logical
    factory_physical_qubits = n_factory_logical * qubits_per_logical
    factory_qubits_each     = float(qubits_per_logical)  # = factory_physical / num_factories
    space_time_volume       = total_physical_qubits * total_cycles
    code_distance_out       = math.isqrt((qubits_per_logical + 1) // 2)  # invert 2d²-1

    # Preprocessor context (logical-layer metrics from qiskit_optimizer JSON)
    preprocessor = {
        k: opt[k]
        for k in (
            "source", "num_qubits", "circuit_depth", "t_count", "t_count_explicit",
            "rotation_count", "t_per_rotation", "rotation_synthesis_epsilon",
            "clifford_count", "optimization_reduction",
        )
        if k in opt
    } or None

    return {
        "estimator": "Superstaq Native Resource Estimator (resource-superstaq, SC architecture)",
        "architecture": {
            "type":                        "Surface code — superconducting qubits (Superconductor)",
            "library":                     "resource-superstaq 0.0.1 / resource_estimation.ftqc",
            "code_distance":               code_distance_out,
            "physical_qubits_per_logical": qubits_per_logical,
            "n_t_factories":               num_factories,
            "t_factory_protocol":          "Gidney cultivation (Cultivate(pi/4)) via DefaultLattice",
            "clock_cycle_us":              cycle_time_us,
            "configuration_method": (
                f"Superconductor(d={code_distance}), "
                f"ResourceEstimator, num_factories={num_factories}"
            ),
        },
        # ── Standardized top-level fields (shared schema across all estimators) ──
        "logical_qubits":    n_algo_logical,
        "physical_qubits":   int(total_physical_qubits),
        "t_gates":           t_gates,
        "clifford_gates":    clifford_gates,
        "space_time_volume": float(space_time_volume),
        "space_time_unit":   "qubit·cycles",
        "wall_clock_time":   wall_clock_s,
        # ── Detailed physical breakdown (matches qualtran/psiquantum schema) ─────
        "breakdown": {
            # Qubit decomposition
            "data_physical_qubits":    int(data_physical_qubits),
            "factory_physical_qubits": int(factory_physical_qubits),
            "qubits_per_logical":      qubits_per_logical,
            "n_algo_logical":          n_algo_logical,
            "n_factory_logical":       n_factory_logical,
            "n_t_factories":           num_factories,
            "factory_qubits_each":     factory_qubits_each,
            # Cycle budget — algorithm_cycles == total_cycles per the parallel model
            "data_path_cycles":        int(round(algorithm_cycles)),
            "factory_path_cycles":     int(round(algorithm_cycles)),
            "algorithm_cycles":        algorithm_cycles,
            "t_distillation_cycles":   t_distillation_cycles,
            "total_cycles":            total_cycles,
            "wall_clock_us":           wall_clock_us,
            "cycle_time_us":           cycle_time_us,
            "circuit_depth":           circuit_depth,
            "t_depth":                 None,   # not exposed by resource-superstaq
            "error_budget":            {},
            # Preprocessor provenance (logical-layer metrics from qiskit_optimizer)
            "qiskit_preprocessor":     preprocessor,
        },
    }


def main() -> None:
    args = parse_args()

    try:
        qc, opt = load_inputs(args)
    except Exception as exc:
        print(json.dumps({"error": f"Circuit load failed: {exc}"}), file=sys.stderr)
        sys.exit(1)

    try:
        output = estimate_resources(
            qc, opt,
            num_factories=args.num_factories,
            code_distance=args.code_distance,
        )
    except Exception as exc:
        print(json.dumps({"error": f"Superstaq estimation failed: {exc}"}), file=sys.stderr)
        sys.exit(1)

    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
