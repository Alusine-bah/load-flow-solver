"""
Per-unit conversion utilities.

Every quantity in power system analysis is expressed as a fraction
of a "base" value with the same units:

    quantity_pu = quantity_actual / quantity_base

The four base quantities are:
    S_base — three-phase power (MVA)
    V_base — line-to-line voltage (kV)
    I_base — line current (kA)
    Z_base — impedance per phase (Ω)

Only two of these can be chosen arbitrarily. Following standard
practice, we choose S_base and V_base, and the other two follow:

    I_base = S_base / (√3 · V_base)              [kA]
    Z_base = V_base² / S_base                    [Ω]

References
----------
Saadat, "Power System Analysis", 2nd ed.
    Eq. 3.79  — base current
    Eq. 3.81  — base impedance
    Eq. 3.89  — per-unit impedance from ohms
    Eq. 3.91  — change of base

Kothari & Nagrath, "Modern Power System Analysis"
    Eq. 4.4   — the same relations written in different units
"""

import math


SQRT3 = math.sqrt(3.0)


# ── Base quantity derivations ────────────────────────────────

def base_current_kA(S_base_MVA: float, V_base_kV: float) -> float:
    """
    Compute base line current in kA.

    From Saadat Eq. 3.79:
        I_base = S_base / (√3 · V_base)

    Parameters
    ----------
    S_base_MVA : three-phase base power in MVA
    V_base_kV  : line-to-line base voltage in kV

    Returns
    -------
    Base current in kA.
    """
    return S_base_MVA / (SQRT3 * V_base_kV)


def base_impedance_ohm(S_base_MVA: float, V_base_kV: float) -> float:
    """
    Compute base impedance per phase in ohms.

    From Saadat Eq. 3.81:
        Z_base = V_base² / S_base

    Parameters
    ----------
    S_base_MVA : three-phase base power in MVA
    V_base_kV  : line-to-line base voltage in kV

    Returns
    -------
    Base impedance in ohms.
    """
    return (V_base_kV ** 2) / S_base_MVA


# ── Per-unit conversions ─────────────────────────────────────

def to_pu(value: float, base: float) -> float:
    """Convert a value to per-unit. Raises if base is zero."""
    if base == 0:
        raise ZeroDivisionError("Base value cannot be zero")
    return value / base


def from_pu(value_pu: float, base: float) -> float:
    """Convert a per-unit value back to physical units."""
    return value_pu * base


def impedance_to_pu(Z_ohm: float, S_base_MVA: float, V_base_kV: float) -> float:
    """
    Convert an impedance in ohms to per-unit on the given base.

    From Saadat Eq. 3.89:
        Z_pu = Z_ohm · S_base / V_base²
    """
    Z_base = base_impedance_ohm(S_base_MVA, V_base_kV)
    return Z_ohm / Z_base


def impedance_from_pu(Z_pu: float, S_base_MVA: float, V_base_kV: float) -> float:
    """Convert a per-unit impedance back to ohms."""
    return Z_pu * base_impedance_ohm(S_base_MVA, V_base_kV)


def current_to_pu(I_kA: float, S_base_MVA: float, V_base_kV: float) -> float:
    """Convert a current in kA to per-unit."""
    return I_kA / base_current_kA(S_base_MVA, V_base_kV)


def current_from_pu(I_pu: float, S_base_MVA: float, V_base_kV: float) -> float:
    """Convert a per-unit current to kA."""
    return I_pu * base_current_kA(S_base_MVA, V_base_kV)


# ── Change of base (Saadat Eq. 3.91) ─────────────────────────

def change_of_base_z(
    Z_pu_old: float,
    S_old_MVA: float,
    V_old_kV: float,
    S_new_MVA: float,
    V_new_kV: float,
) -> float:
    """
    Change a per-unit impedance from one base to another.

    From Saadat Eq. 3.91:

        Z_pu_new = Z_pu_old · (S_new / S_old) · (V_old / V_new)²

    Parameters
    ----------
    Z_pu_old  : per-unit impedance on the old base
    S_old_MVA : old power base in MVA
    V_old_kV  : old voltage base in kV
    S_new_MVA : new power base in MVA
    V_new_kV  : new voltage base in kV

    Returns
    -------
    Per-unit impedance on the new base.
    """
    return Z_pu_old * (S_new_MVA / S_old_MVA) * ((V_old_kV / V_new_kV) ** 2)


def percent_to_pu(percent: float) -> float:
    """Convert a percentage reactance or impedance to per-unit."""
    return percent / 100.0


def pu_to_percent(pu: float) -> float:
    """Convert a per-unit reactance or impedance to percent."""
    return pu * 100.0


# ── Convenience: turn "z at rated kVA, kV" into z on system base ──

def device_z_on_base(
    z_percent: float,
    device_MVA: float,
    device_kV: float,
    system_MVA: float,
    system_kV: float,
) -> float:
    """
    Convert a device's nameplate per-unit impedance (usually given
    in percent) to the system base.

    This wraps Eq. 3.91 with the conversion from percent to pu
    on the nameplate base, then to the system base.

    Example
    -------
    A 50 MVA, 3.3 kV generator has X = 10%. What is X on a
    100 MVA, 5 kV base?

        >>> device_z_on_base(10, 50, 3.3, 100, 5)
        0.08712

    This matches the worked example from the per-unit lecture
    notes: 0.1 × (3.3/5.0)² × (100/50) = 0.087.
    """
    z_pu_on_device = percent_to_pu(z_percent)
    return change_of_base_z(z_pu_on_device, device_MVA, device_kV,
                            system_MVA, system_kV)


# ── Impedance reference (Saadat Eq. 3.86) ────────────────────

def load_impedance_ohm(S_load_MVA: float, V_L_L_kV: float,
                       power_factor: float = 1.0,
                       leading: bool = False) -> complex:
    """
    Compute the load impedance per phase from apparent power and
    voltage at the load.

    From Saadat Eq. 3.86:
        Z_p = |V_L-L|² / S_L(3φ)*

    Parameters
    ----------
    S_load_MVA  : three-phase apparent power in MVA
    V_L_L_kV    : line-to-line voltage at the load in kV
    power_factor: cos(θ) — defaults to 1.0 (purely resistive)
    leading     : if True, the load is capacitive (negative Q)

    Returns
    -------
    Complex load impedance per phase in ohms.
    """
    # Convert PF to a complex power direction.
    theta = math.acos(max(min(power_factor, 1.0), -1.0))
    if leading:
        theta = -theta
    S_complex = S_load_MVA * complex(math.cos(theta), math.sin(theta))
    # Z = |V|² / S*
    return (V_L_L_kV ** 2) / S_complex.conjugate()


def load_impedance_pu(S_load_MVA: float, V_load_kV: float,
                      S_base_MVA: float, V_base_kV: float,
                      power_factor: float = 1.0,
                      leading: bool = False) -> complex:
    """
    Compute the per-unit load impedance on the system base.

    From Saadat Eq. 3.88:
        Z_pu = |V_pu|² / S_L(pu)*
    """
    Z_ohm = load_impedance_ohm(S_load_MVA, V_load_kV, power_factor, leading)
    Z_base = base_impedance_ohm(S_base_MVA, V_base_kV)
    return Z_ohm / Z_base