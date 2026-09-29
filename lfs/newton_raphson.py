"""
Newton-Raphson power flow solver.

The Newton-Raphson method solves the power flow equations by
iterative linearization. At each iteration it computes the
mismatch between scheduled and calculated power injections,
builds the Jacobian matrix of partial derivatives, solves a
linear system for the voltage corrections, and applies them.

Compared to Gauss-Seidel, Newton-Raphson:
    - converges in 4–6 iterations instead of 15–30
    - handles stressed systems that GS cannot
    - has quadratic convergence near the solution

The trade-off is that each iteration is more expensive, since it
builds and solves a linear system. For most systems, the total
cost is still lower than GS.

References
----------
Saadat, "Power System Analysis", 2nd ed., §6.3.2, §6.10
    Table 6.1 — Jacobian element formulas
    Eq. 6.38–6.41 — mismatch equations
Stevenson, "Elements of Power System Analysis", §9.5
"""

import numpy as np

from .models import PowerSystem, SLACK, PV, PQ


def solve_newton_raphson(
    system: PowerSystem,
    Ybus: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 20,
    verbose: bool = False,
) -> dict:
    """
    Solve the power flow problem using the Newton-Raphson method.

    Parameters
    ----------
    system : PowerSystem
        The system to solve. Must have exactly one slack bus.
    Ybus : np.ndarray
        Bus admittance matrix from build_ybus(system).
    tol : float
        Convergence tolerance on the maximum mismatch (pu).
    max_iter : int
        Maximum iterations (NR typically needs 4–6).
    verbose : bool
        If True, print iteration-by-iteration progress.

    Returns
    -------
    result : dict — same structure as solve_gauss_seidel.
    """
    n = system.n_buses
    slack_idx = system.bus_index(system.slack_bus().id)

    # Classify buses and build the ordered list of unknown variables.
    # The NR state vector is [δ_1, ..., δ_{n-1}, |V|_1, ..., |V|_q]
    # where δ excludes the slack and |V| includes only PQ buses.
    pq_indices = [i for i, b in enumerate(system.buses) if b.type == PQ]
    pv_indices = [i for i, b in enumerate(system.buses) if b.type == PV]
    non_slack  = [i for i in range(n) if i != slack_idx]

    n_angle = len(non_slack)          # one angle unknown per non-slack bus
    n_mag   = len(pq_indices)         # one magnitude unknown per PQ bus
    state_size = n_angle + n_mag

    # Scheduled injections.
    P_sched = np.array([b.P_injected for b in system.buses])
    Q_sched = np.array([b.Q_injected for b in system.buses])

    # Initial estimate: use each bus's stored V and delta.
    V = np.array([b.V * np.exp(1j * b.delta) for b in system.buses], dtype=complex)

    # Keep the fixed magnitudes for PV buses.
    V_fixed = {i: system.buses[i].V for i in pv_indices}

    convergence_history = []

    for iteration in range(1, max_iter + 1):
        # ── 1. Compute injections from the current voltage estimate ──
        S_calc = V * np.conj(Ybus @ V)
        P_calc = S_calc.real
        Q_calc = S_calc.imag

        # ── 2. Build the mismatch vector ──
        # ΔP is computed for all non-slack buses.
        # ΔQ is computed only for PQ buses.
        dP = P_sched[non_slack] - P_calc[non_slack]
        dQ = Q_sched[pq_indices] - Q_calc[pq_indices]
        mismatch = np.concatenate([dP, dQ])

        max_mismatch = float(np.max(np.abs(mismatch)))
        convergence_history.append(max_mismatch)

        if verbose:
            print(f"  iter {iteration:3d}   max mismatch = {max_mismatch:.3e}")

        if max_mismatch < tol:
            break

        # ── 3. Build the Jacobian ──
        J = _build_jacobian(V, Ybus, P_calc, Q_calc,
                            non_slack, pq_indices, n_angle, n_mag)

        # ── 4. Solve J · dx = mismatch ──
        try:
            dx = np.linalg.solve(J, mismatch)
        except np.linalg.LinAlgError:
            if verbose:
                print(f"  iter {iteration:3d}   Jacobian singular, stopping")
            break

        # ── 5. Unpack the correction vector ──
        d_delta = dx[:n_angle]
        d_mag   = dx[n_angle:]

        # ── 6. Apply corrections ──
        for k, i in enumerate(non_slack):
            V[i] *= np.exp(1j * d_delta[k])

        for k, i in enumerate(pq_indices):
            V[i] *= (1.0 + d_mag[k])

        # ── 7. Re-impose the PV bus magnitudes ──
        for i, mag in V_fixed.items():
            V[i] = mag * np.exp(1j * np.angle(V[i]))

    # ── Compute final injections ──
    S_calc = V * np.conj(Ybus @ V)
    P_final = S_calc.real
    Q_final = S_calc.imag

    result = {
        "converged": max_mismatch < tol,
        "iterations": iteration,
        "V": V,
        "V_magnitude": np.abs(V),
        "delta_rad": np.angle(V),
        "delta_deg": np.degrees(np.angle(V)),
        "P_injected": P_final,
        "Q_injected": Q_final,
        "convergence_history": convergence_history,
    }
    return result


# ── Jacobian construction ────────────────────────────────────

def _build_jacobian(V, Ybus, P_calc, Q_calc,
                    non_slack, pq_indices, n_angle, n_mag):
    """
    Build the Jacobian matrix using Saadat Table 6.1 formulas.

    The Jacobian has four sub-matrices:
        [ H  N ]     H = ∂P/∂δ   (non_slack × non_slack)
        [ M  L ]     N = ∂P/∂|V| (non_slack × pq)
                     M = ∂Q/∂δ   (pq × non_slack)
                     L = ∂Q/∂|V| (pq × pq)

    Diagonal (i = j):
        H_ii = -Q_i - B_ii · |V_i|²
        N_ii =  P_i / |V_i| + G_ii · |V_i|
        M_ii =  P_i - G_ii · |V_i|²
        L_ii =  Q_i / |V_i| - B_ii · |V_i|

    Off-diagonal (i ≠ j):
        H_ij = |V_i| · |V_j| · (G_ij·sin θ_ij - B_ij·cos θ_ij)
        N_ij = |V_i| · (G_ij·cos θ_ij + B_ij·sin θ_ij)
        M_ij = -|V_i| · |V_j| · (G_ij·cos θ_ij + B_ij·sin θ_ij)
        L_ij = |V_i| · (G_ij·sin θ_ij - B_ij·cos θ_ij)

    where θ_ij = δ_i - δ_j.
    """
    J = np.zeros((n_angle + n_mag, n_angle + n_mag))
    G = Ybus.real
    B = Ybus.imag
    Vmag = np.abs(V)
    Vang = np.angle(V)

    # ── H = ∂P/∂δ  (rows: non_slack, cols: non_slack) ──
    for r, i in enumerate(non_slack):
        for c, j in enumerate(non_slack):
            if i == j:
                J[r, c] = -Q_calc[i] - B[i, i] * Vmag[i] ** 2
            else:
                theta = Vang[i] - Vang[j]
                J[r, c] = Vmag[i] * Vmag[j] * (
                    G[i, j] * np.sin(theta) - B[i, j] * np.cos(theta)
                )

    # ── N = ∂P/∂|V|  (rows: non_slack, cols: pq) ──
    for r, i in enumerate(non_slack):
        for c, j in enumerate(pq_indices):
            col = n_angle + c
            if i == j:
                J[r, col] = P_calc[i] / Vmag[i] + G[i, i] * Vmag[i]
            else:
                theta = Vang[i] - Vang[j]
                J[r, col] = Vmag[i] * (
                    G[i, j] * np.cos(theta) + B[i, j] * np.sin(theta)
                )

    # ── M = ∂Q/∂δ  (rows: pq, cols: non_slack) ──
    for r, i in enumerate(pq_indices):
        row = n_angle + r
        for c, j in enumerate(non_slack):
            if i == j:
                J[row, c] = P_calc[i] - G[i, i] * Vmag[i] ** 2
            else:
                theta = Vang[i] - Vang[j]
                J[row, c] = -Vmag[i] * Vmag[j] * (
                    G[i, j] * np.cos(theta) + B[i, j] * np.sin(theta)
                )

    # ── L = ∂Q/∂|V|  (rows: pq, cols: pq) ──
    for r, i in enumerate(pq_indices):
        row = n_angle + r
        for c, j in enumerate(pq_indices):
            col = n_angle + c
            if i == j:
                J[row, col] = Q_calc[i] / Vmag[i] - B[i, i] * Vmag[i]
            else:
                theta = Vang[i] - Vang[j]
                J[row, col] = Vmag[i] * (
                    G[i, j] * np.sin(theta) - B[i, j] * np.cos(theta)
                )

    return J


# ── Reporting helper (shares format with Gauss-Seidel) ──────

def print_results(system: PowerSystem, result: dict) -> None:
    """Print a formatted table of results."""
    print(f"\nNewton-Raphson Results for '{system.name}'")
    status = "CONVERGED" if result["converged"] else "NOT CONVERGED"
    print(f"Status    : {status}")
    print(f"Iterations: {result['iterations']}")
    print(f"Tolerance : max mismatch = {result['convergence_history'][-1]:.3e}\n")

    header = (f"{'Bus':>5} {'Name':<12} {'Type':<7} "
              f"{'|V| (pu)':>10} {'Angle (°)':>12} "
              f"{'P (pu)':>10} {'Q (pu)':>10}")
    print(header)
    print("-" * len(header))

    for i, bus in enumerate(system.buses):
        print(f"{bus.id:>5} {bus.name[:12]:<12} {bus.type:<7} "
              f"{result['V_magnitude'][i]:>10.4f} "
              f"{result['delta_deg'][i]:>12.4f} "
              f"{result['P_injected'][i]:>10.4f} "
              f"{result['Q_injected'][i]:>10.4f}")
    print()