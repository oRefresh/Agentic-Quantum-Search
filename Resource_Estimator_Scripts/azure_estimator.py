import argparse
import json
import re
import sys
from typing import Optional

from qdk.qre import estimate
from qdk.qre.application import OpenQASMApplication
from qdk.qre.models import GateBased, SurfaceCode, RoundBasedFactory


_T_GATES = {"t", "tdg"}
_SKIP_GATES = {
    "barrier", "measure", "reset", "delay", "snapshot",
    "openqasm", "include", "qreg", "creg", "gate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Azure QREv3 Resource Estimator - superconducting-style gate-based setup"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--qasm", type=str, help="Raw OpenQASM 2.0 string")
    group.add_argument("--qasm-file", type=str, help="Path to a .qasm file")
    group.add_argument("--optimizer-output", type=str,
                       help="Full JSON blob containing qasm_string")
    group.add_argument("--optimizer-output-file", type=str,
                       help="Path to JSON file containing qasm_string")
    parser.add_argument(
        "--max-duration", type=int, default=None,
        help="Upper bound on runtime in nanoseconds"
    )
    parser.add_argument(
        "--max-physical-qubits", type=int, default=None,
        help="Upper bound on physical qubit count"
    )
    return parser.parse_args()

def load_inputs(args: argparse.Namespace) -> tuple[str, dict]:
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



def _is_t_like_rotation(line: str) -> bool:
    s = line.replace(" ", "").lower()
    patterns = [
        "rz(pi/4)", "rz(-pi/4)",
        "u1(pi/4)", "u1(-pi/4)",
        "p(pi/4)",  "p(-pi/4)",
    ]
    return any(p in s for p in patterns)

def _parse_qasm(qasm: str) -> tuple[int, int, int]:
    """Return (t_count, clifford_count, logical_qubits) from a raw QASM string."""
    t_count = 0
    clifford_count = 0
    logical_qubits = 0

    for line in qasm.splitlines():
        line = line.split("//")[0].strip()
        if not line:
            continue

        m = re.match(r"^qreg\s+\w+\[(\d+)\]\s*;", line)
        if m:
            logical_qubits += int(m.group(1))
            continue

        m = re.match(r"^([a-z][a-z0-9_]*)", line, re.IGNORECASE)
        if not m:
            continue

        gate = m.group(1).lower()
        if gate in _SKIP_GATES:
            continue
        if gate in _T_GATES or _is_t_like_rotation(line):
            t_count += 1
        else:
            clifford_count += 1

    return t_count, clifford_count, logical_qubits




def _build_architectures() -> list[dict]:
    """
    Superconducting-style Azure QRE configurations:
      - gate-based physical model
      - surface code
      - round-based distillation factory
      - ns timing regime
    """
    gate_isa = SurfaceCode.q() * RoundBasedFactory.q()

    return [
        {
            "label": "Azure QRE gate-based superconducting-style, p=1e-3",
            "qubit_params": {
                "model": "gate-based",
                "platform_style": "superconducting",
                "error_rate": 1e-3,
                "gate_time_ns": 100,
                "measurement_time_ns": 600,
            },
            "qec_scheme": "surface_code + round_based_factory",
            "architecture": GateBased(
                error_rate=1e-3,
                gate_time=100,
                measurement_time=600,
            ),
            "isa_query": gate_isa,
        },
        {
            "label": "Azure QRE gate-based superconducting-style, p=1e-4",
            "qubit_params": {
                "model": "gate-based",
                "platform_style": "superconducting",
                "error_rate": 1e-4,
                "gate_time_ns": 100,
                "measurement_time_ns": 600,
            },
            "qec_scheme": "surface_code + round_based_factory",
            "architecture": GateBased(
                error_rate=1e-4,
                gate_time=100,
                measurement_time=600,
            ),
            "isa_query": gate_isa,
        },
    ]


def _make_record(entry, arch_meta: dict, t_gates: int,
                 clifford_gates: int, logical_qubits: int) -> dict:
    return {
        "estimator": "Microsoft Azure Quantum Resource Estimator (QREv3)",
        "architecture": {
            "name": arch_meta["label"],
            "qubit_params": arch_meta["qubit_params"],
            "qec_scheme": arch_meta["qec_scheme"],
        },
        "logical_qubits": logical_qubits,
        "physical_qubits": entry.qubits,
        "t_gates": t_gates,
        "clifford_gates": clifford_gates,
        "space_time_volume": None,
        "wall_clock_time_us": entry.runtime / 1000.0,
        "breakdown": {
            "error": entry.error,
        },
    }


def _within_constraints(
    record: dict,
    max_duration_ns: Optional[int],
    max_physical_qubits: Optional[int],
) -> bool:
    if max_physical_qubits is not None:
        pq = record.get("physical_qubits")
        if isinstance(pq, int) and pq > max_physical_qubits:
            return False

    if max_duration_ns is not None:
        rt = record.get("wall_clock_time_us")
        if isinstance(rt, (int, float)) and rt > max_duration_ns:
            return False

    return True


def main() -> None:
    args = parse_args()

    try:
        qasm_str, opt = load_inputs(args)

    except Exception as exc:
        print(json.dumps([{"error": f"QASM load failed: {exc}"}]), file=sys.stderr)
        sys.exit(1)

    parsed_t_gates, clifford_gates, logical_qubits = _parse_qasm(qasm_str)
    t_gates = opt.get("t_count", parsed_t_gates)
    logical_qubits = opt.get("num_qubits", logical_qubits)
    raw_gate_counts = opt.get("raw_gate_counts", {})
    clifford_gates = sum(
        v for k, v in raw_gate_counts.items()
        if k in {"h", "s", "sdg", "x", "y", "z", "sx", "sxdg", "cx", "cy", "cz", "swap", "id"}
    ) if raw_gate_counts else clifford_gates
    try:
        architectures = _build_architectures()
    except Exception as exc:
        print(json.dumps([{"error": f"Architecture configuration failed: {exc}"}]), file=sys.stderr)
        sys.exit(1)

    try:
        app = OpenQASMApplication(program=qasm_str)
    except Exception as exc:
        print(json.dumps([{"error": f"OpenQASMApplication init failed: {exc}"}]), file=sys.stderr)
        sys.exit(1)

    outputs = []
    errors = []

    for arch in architectures:
        try:
            table = estimate(app, arch["architecture"], arch["isa_query"])
            for entry in table:
                record = _make_record(
                    entry, arch, t_gates, clifford_gates, logical_qubits
                )
                if _within_constraints(record, args.max_duration, args.max_physical_qubits):
                    outputs.append(record)
        except Exception as exc:
            errors.append({"architecture": arch["label"], "error": str(exc)})

    if errors:
        print(json.dumps(errors, indent=2), file=sys.stderr)

    print(json.dumps(outputs, indent=2, default=str))


if __name__ == "__main__":
    main()
