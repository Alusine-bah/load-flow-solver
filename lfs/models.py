"""
Data models for power system elements.

Every power system is described by three things:
- Buses (nodes where lines and equipment connect)
- Lines (branches between buses)
- Generators (sources connected to buses)

The PowerSystem class collects these into one object and hands them off
to the Y-bus builder and load flow solvers.
"""

from dataclasses import dataclass, field
from typing import Optional


# ── Bus types ────────────────────────────────────────────────
# In load flow analysis, every bus is classified as one of:
#   SLACK  — reference bus. Voltage magnitude AND angle are fixed.
#            The slack bus supplies whatever real and reactive power
#            is needed to balance the system. Exactly one per system.
#   PV     — generator bus. Voltage magnitude (V) and real power (P)
#            are fixed. The solver finds reactive power Q and angle.
#   PQ     — load bus. Real power P and reactive power Q are fixed.
#            The solver finds voltage magnitude V and angle.
SLACK = "slack"
PV = "pv"
PQ = "pq"

@dataclass
class Shunt:
    """
    A shunt admittance connected to a bus.

    Represents a capacitor bank, reactor, or any external
    admittance to ground at a bus. Contributes directly to the
    diagonal of the Y-bus:

        Y_ii += Y_shunt

    where Y_shunt = G + jB.

    Sign convention:
        B > 0  → capacitor (injects reactive power)
        B < 0  → reactor   (absorbs reactive power)
    """
    bus_id: int
    G: float = 0.0
    B: float = 0.0

    @property
    def Y(self) -> complex:
        return complex(self.G, self.B)

    def __repr__(self) -> str:
        return f"Shunt(bus={self.bus_id}, Y={self.Y:+.4f}j)"


@dataclass
class Bus:

    """A node in the power system network."""
    id: int
    name: str = ""
    type: str = PQ

    # Nominal voltage in kV — used for per-unit conversion.
    base_kv: float = 1.0

    # Load at this bus (in per-unit, on the system base).
    # Positive P = consuming real power. Positive Q = consuming reactive.
    P_load: float = 0.0
    Q_load: float = 0.0

    # Generation at this bus (in per-unit).
    # Only meaningful for SLACK and PV buses.
    P_gen: float = 0.0
    Q_gen: float = 0.0

    # Voltage magnitude (per-unit) — for SLACK and PV buses, this
    # is fixed by the user. For PQ buses, it's solved for.
    V: float = 1.0

    # Voltage angle in radians — for SLACK, this is fixed (usually 0).
    # For PV and PQ buses, it's solved for.
    delta: float = 0.0

    def __post_init__(self):
        if not self.name:
            self.name = f"Bus {self.id}"

    @property
    def P_injected(self) -> float:
        """Net real power injected into the bus (gen − load)."""
        return self.P_gen - self.P_load

    @property
    def Q_injected(self) -> float:
        """Net reactive power injected into the bus (gen − load)."""
        return self.Q_gen - self.Q_load

    def __repr__(self) -> str:
        return (f"Bus(id={self.id}, name='{self.name}', type={self.type}, "
                f"V={self.V:.4f}, δ={self.delta:.4f})")


@dataclass
class Line:
    """A transmission line or transformer connecting two buses."""
    from_bus: int
    to_bus: int

    # Series impedance (per-unit on the system base).
    R: float = 0.0
    X: float = 0.0

    # Shunt admittance (per-unit). For transmission lines this is
    # usually just the capacitive charging susceptance, so B > 0.
    # Half is placed at each end in the π model.
    B: float = 0.0
    G: float = 0.0

    # MVA rating (for overload checking, optional).
    rating_mva: Optional[float] = None

    def __repr__(self) -> str:
        return (f"Line({self.from_bus} → {self.to_bus}, "
                f"Z={self.R:.4f}+j{self.X:.4f}, B={self.B:.4f})")


@dataclass
class Generator:
    """A generator connected to a bus."""
    bus_id: int
    # Real power output in per-unit.
    P: float = 0.0
    # Voltage setpoint (per-unit). Only used for PV and slack buses.
    V_setpoint: float = 1.0
    # Reactive power limits (used later for PV→PQ switching).
    Q_min: float = -1.0
    Q_max: float = 1.0

    def __repr__(self) -> str:
        return f"Generator(bus={self.bus_id}, P={self.P:.4f}, V={self.V_setpoint:.4f})"


@dataclass
class PowerSystem:
    """
    Container for a complete power system.

    A typical use is:

        system = PowerSystem(name="Saadat Example 3.7", base_mva=100)
        system.add_bus(Bus(id=1, type=SLACK, V=1.0, base_kv=22))
        system.add_bus(Bus(id=2, type=PV, V=1.0, base_kv=22, P_gen=0.0))
        system.add_bus(Bus(id=3, type=PQ, base_kv=22, P_load=0.5, Q_load=0.3))
        system.add_line(Line(from_bus=1, to_bus=2, R=0.02, X=0.04, B=0.02))
        system.add_line(Line(from_bus=2, to_bus=3, R=0.03, X=0.05, B=0.01))
    """
    name: str = "Unnamed System"
    base_mva: float = 100.0
    buses: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    generators: list = field(default_factory=list)
    shunts: list = field(default_factory=list)

    def add_bus(self, bus: Bus) -> None:
        """Add a bus to the system. IDs must be unique."""
        if any(b.id == bus.id for b in self.buses):
            raise ValueError(f"Bus id {bus.id} already exists")
        self.buses.append(bus)

    def add_line(self, line: Line) -> None:
        """Add a line. Both end buses must already exist."""
        ids = {b.id for b in self.buses}
        if line.from_bus not in ids:
            raise ValueError(f"from_bus {line.from_bus} not found")
        if line.to_bus not in ids:
            raise ValueError(f"to_bus {line.to_bus} not found")
        if line.from_bus == line.to_bus:
            raise ValueError("Line cannot connect a bus to itself")
        self.lines.append(line)

    def add_generator(self, gen: Generator) -> None:
        """Add a generator. The bus it connects to must already exist."""
        ids = {b.id for b in self.buses}
        if gen.bus_id not in ids:
            raise ValueError(f"Generator bus {gen.bus_id} not found")
        self.generators.append(gen)

    def add_shunt(self, shunt: Shunt) -> None:
        """Add a shunt admittance. The bus must already exist."""
        ids = {b.id for b in self.buses}
        if shunt.bus_id not in ids:
            raise ValueError(f"Shunt bus {shunt.bus_id} not found")
        self.shunts.append(shunt)

    @property
    def n_buses(self) -> int:
        return len(self.buses)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    def get_bus(self, bus_id: int) -> Bus:
        """Return the bus with the given id, or raise KeyError."""
        for b in self.buses:
            if b.id == bus_id:
                return b
        raise KeyError(f"No bus with id {bus_id}")

    def bus_index(self, bus_id: int) -> int:
        """
        Return the position (0-based index) of a bus in the buses list.
        This is the index used for all matrix operations.
        """
        for i, b in enumerate(self.buses):
            if b.id == bus_id:
                return i
        raise KeyError(f"No bus with id {bus_id}")

    def slack_bus(self) -> Bus:
        """Return the (single) slack bus, or raise if not exactly one."""
        slacks = [b for b in self.buses if b.type == SLACK]
        if len(slacks) != 1:
            raise ValueError(f"Expected exactly 1 slack bus, found {len(slacks)}")
        return slacks[0]

    def summary(self) -> str:
        """Return a short multi-line summary of the system."""
        lines = [
            f"System: {self.name}",
            f"  Base MVA    : {self.base_mva}",
            f"  Buses       : {self.n_buses}",
            f"  Lines       : {self.n_lines}",
            f"  Generators  : {len(self.generators)}",
        ]
        by_type = {SLACK: 0, PV: 0, PQ: 0}
        for b in self.buses:
            by_type[b.type] = by_type.get(b.type, 0) + 1
        lines.append(f"    Slack buses: {by_type[SLACK]}")
        lines.append(f"    PV buses   : {by_type[PV]}")
        lines.append(f"    PQ buses   : {by_type[PQ]}")
        return "\n".join(lines)