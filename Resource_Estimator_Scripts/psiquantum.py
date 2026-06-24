"""
PsiQuantum QDK Resource Estimator — Agentic Quantum Search Pipeline

Hardware projection layer for photonic Fusion-Based Quantum Computing (FBQC)
architectures, targeting PsiQuantum's silicon photonic platform.

Attempts native psiqdk (PsiQuantum Construct/QRE) estimation if available in
the current venv; falls back to the analytic FBQC active volume model otherwise.

FBQC model highlights:
  - Fusion-based computation: linear-optical Bell-state measurements (~50% success)
  - Foliated surface code: effective code distance d on the fusion network
  - 15-to-1 T magic-state distillation (resource-state protocol)
  - Active volume: logical space-time volume × fusion redundancy × loss correction
  - GHz photonic clock rate (CLOCK_CYCLE_US = 1e-3 µs = 1 ns per cycle)

Usage:
  python psiquantum.py --qasm "OPENQASM 2.0; ..."
  python psiquantum.py --qasm-file circuit.qasm
  python psiquantum.py --optimizer-output '{"qasm_string": "...", ...}'
  python psiquantum.py --optimizer-output-file optimizer_output.json
"""

import argparse
import json
import math
import re
import sys
from typing import Optional


# ─── FBQC Photonic Physical Parameters ───────────────────────────────────────
PHYS_ERROR_RATE     = 1e-3    # Per-component photon loss / error rate
FBQC_THRESHOLD      = 1.1e-2  # FBQC fault-tolerance threshold (photon loss fraction)
CLOCK_CYCLE_US      = 1e-3    # µs per photonic clock cycle (1 GHz → 1 ns)
TARGET_FAILURE_PROB = 1e-2    # Acceptable total algorithm failure probability

FUSION_SUCCESS_PROB  = 0.50   # Linear-optical Bell fusion gate success probability
PHOTON_LOSS_PER_COMP = 1e-3   # Per-component photon transmission loss
DETECTOR_EFFICIENCY  = 0.95   # Single-photon detector efficiency
PHOTONIC_CLOCK_GHZ   = 1.0    # Pulsed photonic clock rate (GHz)

# Resource-state distillation
T_FACTORY_OVERHEAD   = 15     # Resource states consumed per T magic state (15-to-1)

# Photonic module footprint
MODES_PER_LOGICAL    = 50     # Photonic modes per logical qubit (data block)

# Gate-set classification
_CLIFFORD_OPS = {
    'h', 's', 'sdg', 'x', 'y', 'z', 'sx', 'sxdg',
    'cx', 'cy', 'cz', 'swap', 'id', 'barrier', 'reset',
}
_T_LIKE_OPS   = {'t', 'tdg', 'ccx'}


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PsiQuantum QDK Resource Estimator — "
            "Photonic Fusion-Based Quantum Computing (FBQC)"
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
                       help="Path to JSON file emitted by qiskit_optimizer.py")
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[str, dict]:
    """Return (qasm_str, optimizer_data). optimizer_data is {} for raw QASM inputs."""
    if args.optimizer_output_file:
        with open(args.optimizer_output_file) as f:
            data = json.load(f)
        return data["qasm_string"], data
    if args.optimizer_output:
        data = json.loads(args.optimizer_output)
        return data["qasm_string"], data
    if args.qasm_file:
        with open(args.qasm_file) as f:
            return f.read(), {}
    return args.qasm, {}


# ─── Logical Resource Extraction ─────────────────────────────────────────────

def compute_t_depth(circuit) -> int:
    """T-gate critical-path depth via topological sweep over the circuit DAG.

    Clifford gates advance no T-depth: in FBQC they are tracked via Pauli frames.
    CCX (Toffoli) consumes one magic state and counts as T-like depth.
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


def _parse_qasm_fallback(qasm_str: str) -> dict:
    """Regex QASM parser used when Qiskit is unavailable."""
    qregs     = sum(int(m) for m in re.findall(r'qreg\s+\w+\s*\[(\d+)\]', qasm_str))
    t_count   = (len(re.findall(r'\bt\s+',   qasm_str))
                 + len(re.findall(r'\btdg\s+', qasm_str)))
    cx_count  = len(re.findall(r'\bcx\s+',  qasm_str))
    h_count   = len(re.findall(r'\bh\s+',   qasm_str))
    ccx_count = len(re.findall(r'\bccx\s+', qasm_str))
    depth     = max(t_count + cx_count + h_count + ccx_count, 1)
    return {
        "num_qubits":     max(qregs, 1),
        "t_count":        t_count,
        "clifford_count": h_count + cx_count,
        "toffoli_count":  ccx_count,
        "circuit_depth":  depth,
        "t_depth":        None,
        "ops":            {},
    }


def extract_logical_resources(
    qasm_str: str,
    precomputed_ops: Optional[dict] = None,
    precomputed_n_logical: Optional[int] = None,
    precomputed_depth: Optional[int] = None,
) -> dict:
    """Extract logical resource counts.

    Uses Qiskit when available for precise T-depth computation.
    Accepts precomputed fields from qiskit_optimizer to skip re-parsing.
    """
    t_depth = None

    if (precomputed_ops is not None
            and precomputed_n_logical is not None
            and precomputed_depth is not None):
        ops           = precomputed_ops
        n_logical     = precomputed_n_logical
        circuit_depth = precomputed_depth
    else:
        try:
            from qiskit import QuantumCircuit
            circuit       = QuantumCircuit.from_qasm_str(qasm_str)
            ops           = circuit.count_ops()
            n_logical     = circuit.num_qubits
            circuit_depth = circuit.depth()
            t_depth       = compute_t_depth(circuit)
        except ImportError:
            return _parse_qasm_fallback(qasm_str)

    t_raw    = ops.get('t', 0) + ops.get('tdg', 0)
    clifford = sum(v for k, v in ops.items() if k in _CLIFFORD_OPS)
    toffoli  = ops.get('ccx', 0)

    return {
        "num_qubits":     n_logical,
        "t_count":        t_raw,
        "clifford_count": clifford,
        "toffoli_count":  toffoli,
        "circuit_depth":  circuit_depth,
        "t_depth":        t_depth,
        "ops":            dict(ops),
    }


# ─── Native psiqdk Integration (optional) ────────────────────────────────────

def try_psiqdk_native_estimation(
    qasm_str: str,
    n_logical: int,
    t_count: int,
    clifford_count: int,
    circuit_depth: int,
) -> Optional[dict]:
    """Attempt native PsiQuantum QDK resource estimation via Construct/QRE.

    Compiles the circuit through PsiQuantum's FBQC architecture model — including
    native Hamiltonian simulation, Trotterization, and qubitization primitives from
    psiqdk.construct — and returns a standardized sc-dict.

    Integration points:
      psiqdk.construct.PhotonicRoutine  — native gate-set over fusion operations,
          including Hamiltonian simulation and qubitization decompositions
      psiqdk.construct.FBQCCompiler     — architecture-aware compilation to
          fusion networks and resource-state schedules
      psiqdk.qre.FBQCArchitecture       — physical parameter model
      psiqdk.qre.ResourceProfile        — active volume, factory sizing, wall-clock

    Returns None if psiqdk is not installed or estimation fails.
    """
    try:
        from psiqdk.construct import FBQCCompiler, PhotonicRoutine
        from psiqdk.qre import FBQCArchitecture, ResourceProfile

        arch = FBQCArchitecture(
            photon_loss=PHOTON_LOSS_PER_COMP,
            detector_efficiency=DETECTOR_EFFICIENCY,
            fusion_success_prob=FUSION_SUCCESS_PROB,
            clock_rate_ghz=PHOTONIC_CLOCK_GHZ,
        )
        routine  = PhotonicRoutine.from_qasm(qasm_str)
        compiled = FBQCCompiler(architecture=arch).compile(routine)
        raw      = ResourceProfile(compiled, arch).estimate()

        n_fac               = int(raw.get("n_t_factories", 1))
        factory_modes_each  = int(raw.get("factory_modes_per_unit",
                                          T_FACTORY_OVERHEAD * MODES_PER_LOGICAL))
        data_modes          = int(raw.get("data_modes", n_logical * MODES_PER_LOGICAL))
        factory_modes       = int(raw.get("factory_modes", n_fac * factory_modes_each))
        total_modes         = data_modes + factory_modes
        total_cycles        = int(raw.get("total_cycles", circuit_depth))
        wall_clock_us       = total_cycles * CLOCK_CYCLE_US
        duration_hr         = wall_clock_us / 3_600_000_000.0

        return {
            "code_distance":           int(raw.get("effective_code_distance", 0)),
            "qubits_per_logical":      int(raw.get("modes_per_logical", MODES_PER_LOGICAL)),
            "data_physical_qubits":    data_modes,
            "n_t_factories":           n_fac,
            "factory_qubits_each":     factory_modes_each,
            "factory_physical_qubits": factory_modes,
            "total_physical_qubits":   total_modes,
            "data_path_cycles":        int(raw.get("data_path_cycles", total_cycles)),
            "factory_path_cycles":     int(raw.get("factory_path_cycles", total_cycles)),
            "total_cycles":            total_cycles,
            "wall_clock_us":           wall_clock_us,
            "wall_clock_s":            wall_clock_us * 1e-6,
            "space_time_volume":       total_modes * duration_hr,
            "num_qubits":              n_logical,
            "t_count":                 int(raw.get("t_count", t_count)),
            "clifford_count":          int(raw.get("clifford_count", clifford_count)),
            "t_depth":                 int(raw.get("t_depth", 0)) or None,
            "fusion_failure_rate":     float(raw.get("fusion_failure_rate",
                                                     round(1.0 - FUSION_SUCCESS_PROB, 4))),
            "photon_loss_rate":        float(raw.get("photon_loss_rate", PHYS_ERROR_RATE)),
            "data_error":              float(raw.get("data_error", 0.0)),
            "factory_error":           float(raw.get("factory_error", 0.0)),
            "total_error":             float(raw.get("total_error", 0.0)),
            "active_volume":           int(raw.get("active_volume", 0)) or None,
            "native_psiqdk_used":      True,
        }

    except (ImportError, Exception):
        return None


# ─── Analytic FBQC Resource Model (fallback) ─────────────────────────────────

def fbqc_resource_estimation(
    n_logical: int,
    t_count: int,
    clifford_count: int,
    circuit_depth: int,
    t_depth: Optional[int],
) -> dict:
    """Analytic FBQC resource estimation for photonic fusion-based QC.

    Models:
    - Effective code distance: chosen analytically to satisfy TARGET_FAILURE_PROB
      under the FBQC error model (photon loss fraction below FBQC_THRESHOLD).
    - Factory sizing: n_factories parallel 15-to-1 distillation units, sized to
      sustain the T-gate demand of the algorithm within circuit_depth cycles.
      Each unit delivers 1 magic state per (T_FACTORY_OVERHEAD / FUSION_SUCCESS_PROB)
      photonic cycles (including fusion retries for ~50% success rate).
    - Active volume: (logical_stv + factory_stv) × fusion_factor × loss_factor,
      the canonical QREF photonic space-time resource metric.
    - Space-time volume: total_modules × wall_clock_hours (photonic module·hours),
      for cross-backend comparability with the Qualtran surface-code projector.
    """
    # Effective photon loss rate (per gate, 3 optical components + detector)
    survival_per_gate = ((1.0 - PHOTON_LOSS_PER_COMP) ** 3) * DETECTOR_EFFICIENCY
    p_loss_eff        = 1.0 - survival_per_gate

    # ── Effective code distance ───────────────────────────────────────────────
    # Below threshold, logical error rate per cycle: p_L ≈ (p_eff/p_th)^((d+1)/2)
    # Constraint: circuit_depth × p_L ≤ TARGET_FAILURE_PROB
    # → (d+1)/2 ≥ log(circuit_depth / TARGET_FAILURE_PROB) / log(p_th / p_eff)
    if 0 < p_loss_eff < FBQC_THRESHOLD:
        log_ratio = math.log(FBQC_THRESHOLD / p_loss_eff)
        d_min     = (2.0 * math.log(max(circuit_depth, 1) / TARGET_FAILURE_PROB)
                     / log_ratio) - 1.0
        code_distance = max(3, math.ceil(d_min))
        if code_distance % 2 == 0:
            code_distance += 1   # FBQC uses odd code distances
    else:
        code_distance = 0        # At/above threshold — analytic model infeasible

    # ── Factory sizing ────────────────────────────────────────────────────────
    # Each factory unit delivers 1 T magic state every
    # T_FACTORY_OVERHEAD / FUSION_SUCCESS_PROB photonic cycles.
    # n_factories chosen so factory_path_cycles ≈ data_path_cycles.
    cycles_per_magic = T_FACTORY_OVERHEAD / FUSION_SUCCESS_PROB   # = 30 cycles / T-gate
    n_factories      = max(
        1,
        math.ceil(t_count * cycles_per_magic / max(circuit_depth, 1))
    )
    factory_modes_each = T_FACTORY_OVERHEAD * MODES_PER_LOGICAL   # = 750 modes / factory

    # ── Physical module counts ────────────────────────────────────────────────
    data_modules    = n_logical  * MODES_PER_LOGICAL
    factory_modules = n_factories * factory_modes_each
    total_modules   = data_modules + factory_modules

    # ── Cycle counts ─────────────────────────────────────────────────────────
    # data_path_cycles and factory_path_cycles are descriptive sub-metrics —
    # NOT additive. total_cycles is the bottleneck (max of both paths).
    data_path_cycles    = circuit_depth
    factory_path_cycles = math.ceil(t_count * cycles_per_magic / n_factories)
    total_cycles        = max(data_path_cycles, factory_path_cycles)

    # ── Active volume (QREF photonic space-time metric) ───────────────────────
    logical_stv   = n_logical * MODES_PER_LOGICAL * data_path_cycles
    factory_stv   = t_count   * T_FACTORY_OVERHEAD * factory_modes_each
    fusion_factor = 1.0 / FUSION_SUCCESS_PROB
    loss_factor   = 1.0 / survival_per_gate
    active_volume = int((logical_stv + factory_stv) * fusion_factor * loss_factor)

    # ── Wall-clock and cross-backend space-time volume ────────────────────────
    wall_clock_us     = total_cycles * CLOCK_CYCLE_US
    wall_clock_s      = wall_clock_us * 1e-6
    duration_hr       = wall_clock_us / 3_600_000_000.0
    space_time_volume = total_modules * duration_hr   # photonic module·hours

    # ── Error budget ──────────────────────────────────────────────────────────
    # Data-block error: logical errors accumulated over circuit_depth rounds
    if code_distance > 0 and p_loss_eff < FBQC_THRESHOLD:
        p_L        = (p_loss_eff / FBQC_THRESHOLD) ** ((code_distance + 1) / 2)
        data_error = circuit_depth * p_L
    else:
        data_error = 1.0

    # Factory error: 15-to-1 protocol achieves ~15 × p_loss³ per magic state
    factory_error = t_count * (15.0 * p_loss_eff ** 3)
    total_error   = data_error + factory_error

    return {
        "code_distance":           code_distance,
        "qubits_per_logical":      MODES_PER_LOGICAL,
        "data_physical_qubits":    data_modules,
        "n_t_factories":           n_factories,
        "factory_qubits_each":     factory_modes_each,
        "factory_physical_qubits": factory_modules,
        "total_physical_qubits":   total_modules,
        "data_path_cycles":        data_path_cycles,
        "factory_path_cycles":     factory_path_cycles,
        "total_cycles":            total_cycles,
        "wall_clock_us":           wall_clock_us,
        "wall_clock_s":            wall_clock_s,
        "space_time_volume":       space_time_volume,
        "num_qubits":              n_logical,
        "t_count":                 t_count,
        "clifford_count":          clifford_count,
        "t_depth":                 t_depth,
        "fusion_failure_rate":     round(1.0 - FUSION_SUCCESS_PROB, 4),
        "photon_loss_rate":        round(p_loss_eff, 6),
        "data_error":              data_error,
        "factory_error":           factory_error,
        "total_error":             total_error,
        "active_volume":           active_volume,
        "native_psiqdk_used":      False,
    }


# ─── Resource Estimation Dispatcher ──────────────────────────────────────────

def estimate_resources(
    qasm_str: str,
    t_count_override: Optional[int] = None,
    precomputed_ops: Optional[dict] = None,
    precomputed_n_logical: Optional[int] = None,
    precomputed_depth: Optional[int] = None,
) -> dict:
    """Estimate FBQC resources; tries native psiqdk first, then analytic model.

    When precomputed_ops / precomputed_n_logical / precomputed_depth are provided
    (from qiskit_optimizer's raw_gate_counts / num_qubits / circuit_depth), the
    expensive QuantumCircuit.from_qasm_str() call is skipped.
    T-depth is only computed when parsing a fresh Qiskit circuit.
    """
    logical = extract_logical_resources(
        qasm_str,
        precomputed_ops=precomputed_ops,
        precomputed_n_logical=precomputed_n_logical,
        precomputed_depth=precomputed_depth,
    )

    t_count     = t_count_override if t_count_override is not None else logical["t_count"]
    n_logical   = logical["num_qubits"]
    clifford_ct = logical["clifford_count"]
    depth       = logical["circuit_depth"]
    t_depth     = logical.get("t_depth")

    sc = try_psiqdk_native_estimation(
        qasm_str, n_logical, t_count, clifford_ct, depth
    )

    if sc is None:
        sc = fbqc_resource_estimation(
            n_logical=n_logical,
            t_count=t_count,
            clifford_count=clifford_ct,
            circuit_depth=depth,
            t_depth=t_depth,
        )
    elif sc.get("t_depth") is None:
        sc["t_depth"] = t_depth  # merge from Qiskit if native path omitted it

    return sc


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    try:
        qasm_str, opt = load_inputs(args)
    except Exception as e:
        print(json.dumps({"error": f"QASM load failed: {e}"}), file=sys.stderr)
        sys.exit(1)

    t_count_override      = opt.get("t_count")        if opt else None
    precomputed_ops       = opt.get("raw_gate_counts") if opt else None
    precomputed_n_logical = opt.get("num_qubits")      if opt else None
    precomputed_depth     = opt.get("circuit_depth")   if opt else None

    try:
        sc = estimate_resources(
            qasm_str,
            t_count_override=t_count_override,
            precomputed_ops=precomputed_ops,
            precomputed_n_logical=precomputed_n_logical,
            precomputed_depth=precomputed_depth,
        )
    except Exception as e:
        print(json.dumps({"error": f"FBQC estimation failed: {e}"}), file=sys.stderr)
        sys.exit(1)

    output = {
        "estimator": "PsiQuantum QDK — Photonic FBQC (QREF Active Volume Model)",
        "architecture": {
            "type": "Fusion-Based Quantum Computing (FBQC) — foliated surface code on photonic fusion network",
            "physical_platform": "Photonic (silicon photonics)",
            "physical_error_rate": PHYS_ERROR_RATE,
            "surface_code_threshold": FBQC_THRESHOLD,
            "clock_cycle_us": CLOCK_CYCLE_US,
            "code_distance": sc["code_distance"],
            "physical_qubits_per_logical": sc["qubits_per_logical"],
            "routing_overhead_factor": round(1.0 / FUSION_SUCCESS_PROB, 2),
            "t_factory_protocol": "15-to-1 resource-state distillation (linear-optical)",
            "n_t_factories": sc["n_t_factories"],
            "factory_qubits_each": sc["factory_qubits_each"],
            "configuration_method": (
                "native psiqdk ResourceProfile"
                if sc.get("native_psiqdk_used")
                else "analytic FBQC active volume (minimise photonic module·hours)"
            ),
        },
        "logical_qubits":    sc["num_qubits"],
        "physical_qubits":   sc["total_physical_qubits"],
        "t_gates":           sc["t_count"],
        "clifford_gates":    sc["clifford_count"],
        "space_time_volume": sc["space_time_volume"],
        "space_time_unit":   "photonic module·hours",
        "wall_clock_time":   sc["wall_clock_s"],
        "breakdown": {
            "data_physical_qubits":    sc["data_physical_qubits"],
            "factory_physical_qubits": sc["factory_physical_qubits"],
            "data_path_cycles":        sc["data_path_cycles"],
            "factory_path_cycles":     sc["factory_path_cycles"],
            "total_cycles":            sc["total_cycles"],
            "wall_clock_us":           sc["wall_clock_us"],
            "t_depth":                 sc["t_depth"],
            "active_volume":           sc.get("active_volume"),
            "native_psiqdk_used":      sc.get("native_psiqdk_used", False),
            "error_budget": {
                "data_error":          sc["data_error"],
                "factory_error":       sc["factory_error"],
                "total_error":         sc["total_error"],
                "fusion_failure_rate": sc["fusion_failure_rate"],
                "photon_loss_rate":    sc["photon_loss_rate"],
                "target":              TARGET_FAILURE_PROB,
            },
        },
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
