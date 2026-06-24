"""
Qualtran Surface Code Hardware Projector — Agentic Quantum Search Pipeline

Hardware projection layer specifically for 2D superconducting transmon grids
(Google Sycamore style). Ingests an OpenQASM 2.0 string, extracts algorithmic
resource counts (T-gates, logical qubits, T-depth), then runs a full surface
code compilation model via Qualtran's native surface_code module.

Primary computed values come from PhysicalCostModel:
  model.n_cycles(algo_summary)      → total EC cycles (bottleneck of data and factory paths)
  model.duration_hr(algo_summary)   → wall-clock hours
  model.n_phys_qubits(algo_summary) → total physical qubits

Key design decisions:
  - Rotated planar surface code: 2d² physical qubits per logical qubit
  - CCZ2T magic-state distillation (CCZ2TFactory)
  - Code distance d and factory count are NOT chosen by manual heuristics. Instead,
    optimize_configuration() performs a grid search over (d, n_factories) ∈
    {3,5,…,51} × {1,…,20} and selects the configuration that minimises
    space–time volume (qubit·hours) subject to:
        data_block.data_error(...) + factory.factory_error(...) ≤ TARGET_FAILURE_PROB
    This is the correct approach because the optimisation is non-convex — more
    factories shorten total cycles (reducing data-block error) but add physical
    qubits, and the Pareto frontier cannot be found algebraically.
  - Error budget is split between the data block AND the magic-state factory.
    Forcing the data block to absorb the entire budget (as a per-round heuristic)
    overstates the data-block distance requirement and ignores factory error.
  - Factory magic-state demand is passed directly to Qualtran via GateCounts,
    which natively accounts for the 1-to-2 CCZ→T ratio: one CCZ magic state
    produces two T states, so T gates and Toffoli gates are NOT weighted equally.
  - Temporal depth used for scheduling is T-depth (compute_t_depth), not
    raw Qiskit circuit.depth(). Cliffords are tracked via Pauli frames in
    surface code and do not constitute magic-state demand. T-depth is recorded
    as a diagnostic metric; factory sizing is determined by the grid search.
  - 2D nearest-neighbor routing overhead: SimpleDataBlock(routing_overhead=0.5)
    → effective 1.5× qubit factor
  - Wall-clock at 1 µs / error-correction cycle (CLOCK_CYCLE_US)

Gate-set assumption: input QASM is expected to be transpiled into a basis compatible
with the Gidney-Fowler model (Clifford + T + CCX). Rotation gates (rx/ry/rz/u/p/…)
are counted separately but are NOT synthesised into T gates here; if the circuit
contains arbitrary-angle rotations, pre-transpile with Qiskit's solovay_kitaev or
equivalent before running this estimator to get accurate T counts.

Usage:
  python qualtran_estimator.py --qasm "OPENQASM 2.0; ..."
  python qualtran_estimator.py --qasm-file circuit.qasm
  python qualtran_estimator.py --optimizer-output '{"qasm_string": "...", ...}'
"""

import argparse
import json
import sys

from qiskit import QuantumCircuit
from qualtran.resource_counting import GateCounts
from qualtran.surface_code import (
    AlgorithmSummary,
    CCZ2TFactory,
    MultiFactory,
    PhysicalCostModel,
    PhysicalParameters,
    QECScheme,
    SimpleDataBlock,
)


# ─── Superconducting Transmon Physical Parameters (Google Sycamore baseline) ─────
PHYS_ERROR_RATE     = 1e-3   # Per-gate physical error rate (superconducting transmon)
SC_THRESHOLD        = 1e-2   # Surface code fault-tolerance threshold
CLOCK_CYCLE_US      = 1.0    # µs per surface code error-correction cycle
TARGET_FAILURE_PROB = 1e-2   # Acceptable total algorithm failure probability budget

_CLIFFORD_OPS = {
    'h', 's', 'sdg', 'x', 'y', 'z', 'sx', 'sxdg',
    'cx', 'cy', 'cz', 'swap', 'id', 'barrier', 'reset',
}
_ROTATION_OPS = {'rx', 'ry', 'rz', 'r', 'u', 'u1', 'u2', 'u3', 'p', 'ph'}
# Gates that consume magic states and define the T-depth critical path.
_T_LIKE_OPS   = {'t', 'tdg', 'ccx'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualtran Surface Code Projector — "
            "2D Superconducting Transmon Grid (Google Sycamore Style)"
        )
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


def load_inputs(args: argparse.Namespace) -> tuple[str, dict]:
    """
    Returns (qasm_str, optimizer_data).

    When the source is an optimizer JSON (--optimizer-output-file or
    --optimizer-output), optimizer_data carries the pre-computed fields
    (raw_gate_counts, num_qubits, circuit_depth, t_count) so downstream
    code can skip re-parsing the QASM entirely. For raw QASM inputs,
    optimizer_data is an empty dict.
    """
    if args.optimizer_output_file:
        with open(args.optimizer_output_file) as f:
            data = json.loads(f.read())
        return data["qasm_string"], data
    if args.optimizer_output:
        data = json.loads(args.optimizer_output)
        return data["qasm_string"], data
    if args.qasm_file:
        with open(args.qasm_file) as f:
            return f.read(), {}
    return args.qasm, {}


# ─── T-Depth Extraction ───────────────────────────────────────────────────────────

def compute_t_depth(circuit: QuantumCircuit) -> int:
    """
    Critical-path depth counting only T, Tdg, and CCX gates.

    Clifford gates (H, S, CNOT, etc.) do not advance T-depth because in a
    surface code they are tracked via Pauli frames (software) or executed via
    lattice surgery at speeds far faster than magic-state distillation. The
    true scheduling bottleneck is magic-state demand, not raw gate count.

    Algorithm: topological sweep over circuit.data. For each qubit, maintain
    the T-depth at that wire's current position. T-like gates increment by 1;
    all others propagate the max without incrementing.
    """
    qubit_t_depth: dict = {qubit: 0 for qubit in circuit.qubits}
    for instruction in circuit.data:
        name   = instruction.operation.name.lower()
        qubits = instruction.qubits
        pre    = max((qubit_t_depth[q] for q in qubits), default=0)
        post   = pre + (1 if name in _T_LIKE_OPS else 0)
        for q in qubits:
            qubit_t_depth[q] = post
    return max(qubit_t_depth.values(), default=0)


# ─── Space-Time Grid Search ───────────────────────────────────────────────────────

def optimize_configuration(
    gate_counts: GateCounts,
    n_logical: int,
    phys_err: float = PHYS_ERROR_RATE,
    cycle_time_us: float = CLOCK_CYCLE_US,
    target_failure_prob: float = TARGET_FAILURE_PROB,
) -> dict:
    """
    Grid search over (code_distance d, n_factories) to find the configuration
    that minimises space–time volume (qubit·hours) subject to:

        data_block.data_error(algo_summary, logical_error_model)
      + factory.factory_error(algo_summary, logical_error_model)
      ≤ target_failure_prob

    Search grid:
        d         ∈ {3, 5, 7, …, 51}  — 25 odd distances
        n_factories ∈ {1, …, 20}

    Why both axes matter:
        Larger d   → lower logical error rate per cycle, but more physical qubits.
        More factories → shorter total timeline (factory is often the bottleneck),
                         so data qubits idle less, reducing data-block error —
                         but factory qubits increase the physical footprint.
    The optimum is non-convex and cannot be found algebraically.

    No manual n_magic_demand is computed here. GateCounts is passed directly
    to Qualtran, which natively accounts for the 1-to-2 CCZ→T ratio when
    computing factory cycles and error rates.

    Raises RuntimeError if no feasible configuration is found within the grid.

    Returns a dict with:
        d, n_factories, model, data_block, factory, base_factory,
        algo_summary, data_error, factory_error, total_error
    """
    qec_scheme   = QECScheme.make_gidney_fowler()
    phys_params  = PhysicalParameters(physical_error=phys_err, cycle_time_us=cycle_time_us)
    algo_summary = AlgorithmSummary(n_algo_qubits=n_logical, n_logical_gates=gate_counts)

    best_volume   = float('inf')
    best_d        = None
    best_n_fac    = None
    best_result   = None

    for d in range(3, 53, 2):           # odd distances 3 … 51
        data_block = SimpleDataBlock(data_d=d, routing_overhead=0.5)
        for n_fac in range(1, 21):      # 1 … 20 parallel factories
            base_factory = CCZ2TFactory()
            factory = (
                MultiFactory(base_factory=base_factory, n_factories=n_fac)
                if n_fac > 1 else base_factory
            )
            model = PhysicalCostModel(
                physical_params=phys_params,
                data_block=data_block,
                factory=factory,
                qec_scheme=qec_scheme,
            )
            logical_error_model = model.logical_error_model

            # Qualtran native error split — no manual budget heuristics.
            # data_error  : memory + routing errors accumulated by the n_logical
            #               data qubits over the full algorithm timeline (total
            #               bottleneck cycles, including idle rounds waiting for
            #               the factory when it is the critical path).
            # factory_error: probability that the factory produces ≥1 faulty
            #               magic state consumed by the algorithm.
            n_cycles      = int(model.n_cycles(algo_summary))
            data_error    = data_block.data_error(n_logical, n_cycles, logical_error_model)
            factory_error = factory.factory_error(gate_counts, logical_error_model)
            total_error   = data_error + factory_error

            if total_error > target_failure_prob:
                continue

            # Feasible — check space-time volume.
            duration_hr = model.duration_hr(algo_summary)
            n_phys      = model.n_phys_qubits(algo_summary)
            volume      = n_phys * duration_hr   # qubit·hours

            if volume < best_volume:
                best_volume = volume
                best_d      = d
                best_n_fac  = n_fac
                best_result = {
                    'model':         model,
                    'data_block':    data_block,
                    'factory':       factory,
                    'base_factory':  base_factory,
                    'algo_summary':  algo_summary,
                    'data_error':    data_error,
                    'factory_error': factory_error,
                    'total_error':   total_error,
                }

    if best_d is None:
        raise RuntimeError(
            f"No feasible (d, n_factories) configuration satisfies "
            f"total_error ≤ {target_failure_prob}. "
            "Consider relaxing the error budget or extending the search ranges "
            "(d > 51 or n_factories > 20)."
        )

    best_result['d']           = best_d
    best_result['n_factories'] = best_n_fac
    return best_result


# ─── Qualtran Resource Estimation ────────────────────────────────────────────────

def estimate_resources(
    qasm_str: str,
    t_count_override: int | None = None,
    phys_err: float = PHYS_ERROR_RATE,
    cycle_time_us: float = CLOCK_CYCLE_US,
    precomputed_ops: dict | None = None,
    precomputed_n_logical: int | None = None,
    precomputed_depth: int | None = None,
) -> dict:
    """
    Resource estimation using Qualtran's native surface_code module.

    Primary computed values come directly from PhysicalCostModel:
      model.n_cycles(algo_summary)      → total EC cycles        (bottleneck of data and factory paths)
      model.duration_hr(algo_summary)   → wall-clock hours
      model.n_phys_qubits(algo_summary) → total physical qubits

    Code distance and factory count are selected by optimize_configuration(),
    which performs a grid search minimising space–time volume subject to
    total_error ≤ TARGET_FAILURE_PROB. No manual heuristics are used.

    Sub-components are queried from their respective objects:
      data_block.n_cycles / data_block.n_physical_qubits
      factory.n_cycles / factory.n_physical_qubits / base_factory.n_physical_qubits

    When precomputed_ops / precomputed_n_logical / precomputed_depth are provided (from
    qiskit_optimizer's raw_gate_counts / num_qubits / circuit_depth), the expensive
    QuantumCircuit.from_qasm_str() call is skipped. T-depth is only computed when
    parsing a fresh circuit (it requires the circuit DAG structure); when using
    precomputed data, t_depth is None.
    """
    t_depth = None  # Available only when parsing a fresh circuit.

    if precomputed_ops is not None and precomputed_n_logical is not None and precomputed_depth is not None:
        ops           = precomputed_ops
        n_logical     = precomputed_n_logical
        circuit_depth = precomputed_depth   # Full Qiskit depth from optimizer (reference only).
    else:
        circuit       = QuantumCircuit.from_qasm_str(qasm_str)
        ops           = circuit.count_ops()
        n_logical     = circuit.num_qubits
        circuit_depth = circuit.depth()     # Full Qiskit depth, kept for reference.
        t_depth       = compute_t_depth(circuit)   # T-gate critical-path depth.

    t_raw = ops.get('t', 0) + ops.get('tdg', 0)

    gate_counts = GateCounts(
        t=t_count_override if t_count_override is not None else t_raw,
        toffoli=ops.get('ccx', 0),
        cswap=ops.get('cswap', 0),
        clifford=sum(v for k, v in ops.items() if k in _CLIFFORD_OPS),
        rotation=sum(v for k, v in ops.items() if k in _ROTATION_OPS),
        measurement=ops.get('measure', 0),
    )

    # Grid search: finds (d, n_factories) that minimises qubit·hours subject to
    # data_error + factory_error ≤ TARGET_FAILURE_PROB.
    # GateCounts is passed directly; Qualtran handles the 1-to-2 CCZ→T ratio.
    cfg          = optimize_configuration(gate_counts, n_logical, phys_err, cycle_time_us)
    d            = cfg['d']
    n_factories  = cfg['n_factories']
    model        = cfg['model']
    data_block   = cfg['data_block']
    factory      = cfg['factory']
    base_factory = cfg['base_factory']
    algo_summary = cfg['algo_summary']

    # ── Qualtran native calls ──────────────────────────────────────────────────
    total_cycles          = model.n_cycles(algo_summary)
    duration_hr           = model.duration_hr(algo_summary)
    total_physical_qubits = model.n_phys_qubits(algo_summary)

    wall_clock_us     = duration_hr * 3600 * 1e6
    space_time_volume = total_physical_qubits * duration_hr

    # Sub-components queried from their respective objects.
    # data_path_cycles and factory_path_cycles are descriptive sub-metrics —
    # they are NOT additive components of total_cycles. model.n_cycles() is a
    # bottleneck/scheduling result (max of the two paths, possibly with overlap).
    logical_error_model     = model.logical_error_model
    data_path_cycles        = int(data_block.n_cycles(gate_counts, logical_error_model))
    factory_path_cycles     = int(factory.n_cycles(gate_counts, logical_error_model))
    data_physical_qubits    = data_block.n_physical_qubits(n_logical)
    factory_qubits_each     = base_factory.n_physical_qubits()
    factory_physical_qubits = factory.n_physical_qubits()

    return {
        "code_distance":            d,
        "qubits_per_logical":       2 * d ** 2,
        "data_physical_qubits":     int(data_physical_qubits),
        "n_t_factories":            n_factories,
        "factory_qubits_each":      int(factory_qubits_each),
        "factory_physical_qubits":  int(factory_physical_qubits),
        "total_physical_qubits":    int(total_physical_qubits),
        "data_path_cycles":         data_path_cycles,
        "factory_path_cycles":      factory_path_cycles,
        "total_cycles":             int(total_cycles),
        "wall_clock_us":            wall_clock_us,
        "wall_clock_s":             wall_clock_us * 1e-6,
        "space_time_volume":        space_time_volume,
        "num_qubits":               n_logical,
        "t_count":                  gate_counts.t,
        "clifford_count":           gate_counts.clifford,
        "circuit_depth":            circuit_depth,
        "t_depth":                  t_depth,
        "data_error":               cfg['data_error'],
        "factory_error":            cfg['factory_error'],
        "total_error":              cfg['total_error'],
    }


# ─── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    try:
        qasm_str, opt = load_inputs(args)
    except Exception as e:
        print(json.dumps({"error": f"QASM load failed: {e}"}), file=sys.stderr)
        sys.exit(1)

    # When the optimizer JSON is the source, pull the synthesis-corrected T-count and
    # all pre-computed gate/circuit metrics so estimate_resources() can skip the
    # expensive QuantumCircuit.from_qasm_str() call entirely.
    t_count_override      = opt.get("t_count") if opt else None
    precomputed_ops       = opt.get("raw_gate_counts") if opt else None
    precomputed_n_logical = opt.get("num_qubits") if opt else None
    precomputed_depth     = opt.get("circuit_depth") if opt else None

    try:
        sc = estimate_resources(
            qasm_str,
            t_count_override=t_count_override,
            precomputed_ops=precomputed_ops,
            precomputed_n_logical=precomputed_n_logical,
            precomputed_depth=precomputed_depth,
        )
    except Exception as e:
        print(json.dumps({"error": f"Qualtran estimation failed: {e}"}), file=sys.stderr)
        sys.exit(1)

    output = {
        "estimator": "Qualtran Surface Code (2D Superconducting Transmon / Google Sycamore Style)",
        "architecture": {
            "type": "Rotated planar surface code on 2D transmon grid",
            "physical_platform": "Superconducting transmon qubits",
            "physical_error_rate": PHYS_ERROR_RATE,
            "surface_code_threshold": SC_THRESHOLD,
            "clock_cycle_us": CLOCK_CYCLE_US,
            "code_distance": sc["code_distance"],
            "physical_qubits_per_logical": sc["qubits_per_logical"],
            "routing_overhead_factor": 1.5,
            "t_factory_protocol": "CCZ2T magic-state distillation",
            "n_t_factories": sc["n_t_factories"],
            "factory_qubits_each": sc["factory_qubits_each"],
            "configuration_method": "space-time grid search (minimise qubit*hours)",
        },
        "logical_qubits": sc["num_qubits"],
        "physical_qubits": sc["total_physical_qubits"],
        "t_gates": sc["t_count"],
        "clifford_gates": sc["clifford_count"],
        "space_time_volume": sc["space_time_volume"],
        "space_time_unit": "qubit·hours",
        "wall_clock_time": sc["wall_clock_s"],
        "breakdown": {
            "data_physical_qubits":    sc["data_physical_qubits"],
            "factory_physical_qubits": sc["factory_physical_qubits"],
            "data_path_cycles":        sc["data_path_cycles"],
            "factory_path_cycles":     sc["factory_path_cycles"],
            "total_cycles":            sc["total_cycles"],
            "wall_clock_us":           sc["wall_clock_us"],
            "t_depth":                 sc["t_depth"],
            "error_budget": {
                "data_error":    sc["data_error"],
                "factory_error": sc["factory_error"],
                "total_error":   sc["total_error"],
                "target":        TARGET_FAILURE_PROB,
            },
        },
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
