\# ⚡ Load Flow Solver



A power system load flow analysis toolkit in Python.



Newton-Raphson and Gauss-Seidel solvers built from first principles — per-unit conversion, bus admittance matrix construction, and iterative power flow — verified against textbook examples.



\[!\[Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)

\[!\[NumPy](https://img.shields.io/badge/NumPy-2.x-013243.svg)](https://numpy.org/)

\[!\[License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)



\---



\## What It Does



Given a description of a power system — buses, transmission lines, generators, loads — this toolkit computes the steady-state operating point: the voltage magnitude and phase angle at every bus, and the real and reactive power flowing through every line.



That's the \*\*power flow problem\*\*, and it's the first calculation every power engineer runs on a network. It answers questions like:



\- Will the voltage at bus 4 stay within ±5% under peak load?

\- How much reactive power does the generator at bus 2 need to inject?

\- Which line will be overloaded if demand grows by 20%?



This project implements the two classic solution methods — Gauss-Seidel and Newton-Raphson — from scratch, without any power-systems library.



\---



\## Why This Exists



Load flow is the foundation of every other power system study: fault analysis, stability, protection coordination, economic dispatch. Tools like MATPOWER and PSS®E implement it, but using them doesn't teach you how it works.



This project builds it from the ground up so the algorithm is visible and testable:



\- \*\*Per-unit conversion\*\* — all quantities expressed as fractions of a common base, so transformers disappear from the network model

\- \*\*Bus admittance matrix (Y-bus)\*\* — the network encoded as a single complex matrix

\- \*\*Gauss-Seidel iteration\*\* — the textbook first method, slow but simple

\- \*\*Newton-Raphson iteration\*\* — the Jacobian-based method that real utilities use



Every module is verified against a worked example from a standard textbook.



\---



\## Verification



This is the part that matters. The solver isn't "it seems to work" — it reproduces known answers to machine precision.



| Module | Verified against | Result |

|---|---|---|

| `per\_unit.py` | Saadat, \*Power System Analysis\*, §3.14 | Match to 4 decimals |

| `ybus.py` | Stevenson, \*Elements of Power System Analysis\*, Ex. 7.1 | 12/12 matrix entries match |

| `gauss\_seidel.py` | Stevenson 4-bus system | Converges in 17 iterations, `max\\|ΔV\\| = 9.5e-9` |

| `newton\_raphson.py` | Same system, cross-checked vs GS | Converges in 8 iterations, agrees with GS to `1.3e-9` pu |



\*\*Both solvers produce the same final answer\*\* — that's the strongest possible evidence the Jacobian is correct. If any sign or index were wrong in Newton-Raphson, the two methods would converge to different operating points.



\### Sample output



System: 4 buses, 4 lines, 2 shunts (Stevenson Example 7.1 layout)



```

Bus    Name         Type      |V| (pu)   Angle (°)    P (pu)    Q (pu)

1      Slack        slack       1.0500      0.0000    0.5000    1.5895

2      Gen2         pv          1.0200      0.0800    0.5000    1.4682

3      Load3        pq          0.9770     -5.6362   -0.6000   -0.3000

4      Load4        pq          0.9597     -7.5920   -0.4000   -0.2000

```



Load bus voltages sag under load, the generator bus holds its setpoint, and the phases shift by a few degrees — the physics a power engineer expects.



\### Newton-Raphson vs Gauss-Seidel



On the same 4-bus test, both solvers reach the same answer. The difference is how quickly the mismatch shrinks.



```

NR:  6.0e-1  →  5.3e-2  →  2.0e-3  →  1.2e-4  →  5.7e-6  →  2.5e-7  →  1.1e-8  →  4.4e-10

GS:  9.9e-2  →  4.8e-2  →  1.7e-2  →  4.1e-3  →  6.6e-4  →  ... (17 iterations total)

```



Newton-Raphson has \*\*quadratic convergence\*\* — each iteration roughly squares the number of correct digits. Gauss-Seidel is linear. On large systems, the difference is decisive.



\---



\## How to Run



```bash

\# Clone

git clone https://github.com/Alusine-bah/load-flow-solver.git

cd load-flow-solver



\# No dependencies beyond numpy — install it if you don't have it

pip install numpy



\# Run either verification test

python tests/test\_gs\_stevenson.py

python tests/test\_nr\_stevenson.py

```



You should see the Y-bus matrix printed, iteration-by-iteration convergence, and a final results table.



\---



\## How the Solver Works



\### 1. Per-unit conversion



Every physical quantity is expressed as a fraction of a base value. Choose a system power base (`S\_base`) and voltage base (`V\_base`); the current and impedance bases follow:



```

I\_base = S\_base / (√3 · V\_base)     \[kA]

Z\_base = V\_base² / S\_base            \[Ω]

```



The advantage: when every element is on the same base, transformers vanish from the network diagram. The per-unit impedance of a transformer is identical whether computed from the primary or the secondary side.



\### 2. Y-bus construction



The bus admittance matrix encodes the entire network topology. For a system with `n` buses it is an `n × n` complex matrix:



```

Y\_ii = sum of admittances connected to bus i

Y\_ij = negative of the admittance between buses i and j   (i ≠ j)

```



A transmission line between buses p and q, modeled as a π-equivalent, contributes its series admittance to the off-diagonal and its series plus half its shunt to each end of the diagonal.



\### 3. Power flow iteration



Each bus is classified as one of three types:



| Type | Fixed quantities | Solved quantities |

|---|---|---|

| \*\*Slack\*\* | \\|V\\|, δ | P, Q |

| \*\*PV\*\* (generator) | P, \\|V\\| | Q, δ |

| \*\*PQ\*\* (load) | P, Q | \\|V\\|, δ |



\*\*Gauss-Seidel\*\* updates each bus voltage directly using the mismatch between scheduled and calculated power. Simple, but linear convergence.



\*\*Newton-Raphson\*\* linearizes the power equations around the current estimate, builds the Jacobian matrix of partial derivatives, and solves a linear system for the voltage corrections. Quadratic convergence, at the cost of a matrix factorization per iteration.



\---



\## Project Structure



```

load-flow-solver/

├── README.md

├── .gitignore

├── lfs/                          Python package

│   ├── \_\_init\_\_.py

│   ├── models.py                 Bus, Line, Shunt, Generator, PowerSystem

│   ├── per\_unit.py               Base conversion and per-unit transforms

│   ├── ybus.py                   Bus admittance matrix construction

│   ├── gauss\_seidel.py           Gauss-Seidel power flow solver

│   └── newton\_raphson.py         Newton-Raphson power flow solver

└── tests/

&#x20;   ├── test\_gs\_stevenson.py      Gauss-Seidel verification

&#x20;   └── test\_nr\_stevenson.py      Newton-Raphson verification

```



Every file has a docstring explaining its role and a reference to the textbook section it implements.



\---



\## Using the Library



```python

from lfs import Bus, Line, Shunt, PowerSystem, SLACK, PV, PQ

from lfs.ybus import build\_ybus

from lfs.newton\_raphson import solve\_newton\_raphson, print\_results



\# Build a system

sys = PowerSystem(name="My System", base\_mva=100)

sys.add\_bus(Bus(id=1, type=SLACK, V=1.05))

sys.add\_bus(Bus(id=2, type=PV, V=1.02, P\_gen=0.5))

sys.add\_bus(Bus(id=3, type=PQ, P\_load=0.6, Q\_load=0.3))

sys.add\_line(Line(from\_bus=1, to\_bus=2, R=0.0, X=0.4))

sys.add\_line(Line(from\_bus=2, to\_bus=3, R=0.0, X=0.2))



\# Solve

Ybus = build\_ybus(sys)

result = solve\_newton\_raphson(sys, Ybus, verbose=True)



\# Report

print\_results(sys, result)

```



\---



\## Roadmap



\- \[x] Per-unit conversion utilities

\- \[x] Y-bus construction

\- \[x] Gauss-Seidel solver

\- \[x] Newton-Raphson solver

\- \[ ] IEEE 9-bus example (standard test system with published reference values)

\- \[ ] Balanced fault analysis (short-circuit currents)

\- \[ ] Unbalanced fault analysis (symmetrical components)

\- \[ ] Line flow and loss reporting

\- \[ ] Convergence plot figure

\- \[ ] MATPOWER `.m` file importer



\---



\## References



\- Hadi Saadat, \*Power System Analysis\*, 2nd ed., McGraw-Hill

&#x20; - §3.14 (per-unit), §6.3 (solution of nonlinear equations), §6.10 (Jacobian)

\- William D. Stevenson, \*Elements of Power System Analysis\*, 4th ed., McGraw-Hill

&#x20; - §1.12 (Y-bus), §7.1 (worked example), §9.4–9.5 (load flow)

\- Kothari \& Nagrath, \*Modern Power System Analysis\*, 3rd ed., McGraw-Hill

&#x20; - §4.3 (one-line and impedance diagrams), §6 (load flow)



\---



\## Author



\*\*Alusine Bah\*\*

Electrical \& Electronics Engineering

Islamic University of Technology (IUT), OIC



\[alusine-bah.github.io](https://alusine-bah.github.io) · \[GitHub](https://github.com/Alusine-bah)



\---



\## License



MIT — free to use, modify, and distribute.

