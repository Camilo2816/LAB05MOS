"""
Lab 5 - Problema 2
Optimización multiobjetivo en planificación de rutas de inspección (mTSP)

Método: ε-constraint
Objetivo principal: Maximizar calidad de inspección
  -> en Pyomo: Minimizar -calidad
Objetivos convertidos en restricciones:
  - Distancia total <= ε_dist
  - Riesgo total <= ε_risk

Este script:
- Lee la matriz de distancias desde data/asymmetric_from_symmetric_n10.csv
- Construye un mTSP con 2 equipos
- Aplica ε-constraint para generar un frente de Pareto aproximado
- Permite (opcionalmente) no visitar todas las ciudades para generar trade-offs reales
- Imprime tabla de soluciones, rutas y genera gráficas 3D y 2D
"""

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import matplotlib.pyplot as plt
import csv
import os
import math

# ============================================================
# 0. CONFIGURACIÓN GLOBAL
# ============================================================

# Si True: cada ciudad debe ser visitada EXACTAMENTE una vez.
# Si False: cada ciudad puede ser visitada A LO SUMO una vez (orienteering-style).
ENFORCE_ALL_VISITS = True

# Número de equipos 
K_TEAMS = [1, 2]    


# ============================================================
# 1. DATOS DEL PROBLEMA
# ============================================================


DATA_FILE = os.path.join("data", "asymmetric_from_symmetric_n10.csv")


def load_distance_matrix(path):
    """
    Lector robusto para la matriz de distancias.
    Maneja:
      - CSV normal (0,69,52,...)
      - Separador ; (0;69;52;...)
      - Una sola celda con comas "0,69,52,..."
    """
    matrix = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)  # no fijo delimitador, leo crudo
        for row in reader:
            # quitar celdas vacías
            row = [cell.strip() for cell in row if cell.strip() != ""]
            if not row:
                continue

            # Caso 1: varias columnas ya separadas
            if len(row) > 1:
                try:
                    matrix.append([float(x.replace(",", ".")) for x in row])
                    continue
                except ValueError:
                    pass

            # Caso 2: una sola columna con todo pegado por comas
            if len(row) == 1:
                parts = row[0].split(",")
                matrix.append([float(x.replace(",", ".")) for x in parts])
                continue

            raise ValueError(f"No se pudo parsear la fila: {row}")

    return matrix


distance_matrix = load_distance_matrix(DATA_FILE)

n_nodes = len(distance_matrix)


NODES = list(range(n_nodes))  #
DEPOT = 0
CLIENTS = [i for i in NODES if i != DEPOT]

# ------------------------------------------------------------
# 1.2 Matriz de riesgo
# ------------------------------------------------------------

DEFAULT_RISK = 5


risk_data_partial = {
    (0, 1): 3,
    (0, 2): 2,
    (0, 3): 4,
    (0, 4): 5,
    (0, 5): 6,
    (0, 6): 3,
    (0, 7): 2,
    (0, 8): 4,
    (0, 9): 5,
    (2, 8): 9,
    (2, 9): 8,
    (3, 4): 5,
    (4, 9): 7,
    (5, 6): 7,
    (8, 9): 7,
}

# Matriz completa de riesgos, simétrica, usando DEFAULT_RISK por defecto
risk_matrix = [[DEFAULT_RISK for _ in NODES] for _ in NODES]
for (i, j), r in risk_data_partial.items():
    if i < n_nodes and j < n_nodes:
        risk_matrix[i][j] = r
        risk_matrix[j][i] = r


# ------------------------------------------------------------
# 1.3 Calidad de inspección
# ------------------------------------------------------------

quality = {
    1: 85,
    2: 92,
    3: 78,
    4: 90,
    5: 82,
    6: 88,
    7: 95,
    8: 75,
    9: 84,
}
quality[0] = 0 


# ============================================================
# 2. MODELO BASE PYOMO
# ============================================================

def build_base_model():
    """
    Crea el modelo de mTSP multiobjetivo, pero SOLO con:
    - conjuntos
    - parámetros
    - variables
    - restricciones de ruteo y MTZ
    - definiciones de Z1, Z2, Z3 como expresiones
    El objetivo lo definimos aparte según el experimento.
    """

    m = pyo.ConcreteModel()

    # -------------------------------
    # Conjuntos
    # -------------------------------
    m.N = pyo.Set(initialize=NODES)
    m.C = pyo.Set(initialize=CLIENTS)
    m.K = pyo.Set(initialize=K_TEAMS)

    # -------------------------------
    # Parámetros
    # -------------------------------
    def dist_init(m, i, j):
        return distance_matrix[i][j]

    m.dist = pyo.Param(m.N, m.N, initialize=dist_init, within=pyo.NonNegativeReals)

    def risk_init(m, i, j):
        return risk_matrix[i][j]

    m.risk = pyo.Param(m.N, m.N, initialize=risk_init, within=pyo.NonNegativeReals)

    def qual_init(m, i):
        # si algún nodo no está en quality, asumimos 0
        return quality.get(i, 0.0)

    m.qual = pyo.Param(m.N, initialize=qual_init, within=pyo.NonNegativeReals)

    # -------------------------------
    # Variables
    # -------------------------------
    # x[i,j,k] = 1 si equipo k viaja de i a j
    m.x = pyo.Var(m.N, m.N, m.K, within=pyo.Binary)

    # y[i,k] = 1 si equipo k visita el nodo i
    m.y = pyo.Var(m.C, m.K, within=pyo.Binary)

    # variables MTZ para eliminar subciclos
    m.u = pyo.Var(m.C, m.K, bounds=(1, len(CLIENTS)))  # 1..|C|

    # Fijar explícitamente la diagonal x[i,i,k] = 0
    for i in m.N:
        for k in m.K:
            m.x[i, i, k].fix(0)

    # -------------------------------
    # Restricciones
    # -------------------------------

    # 1) Cada nodo (cliente) es visitado a lo sumo una vez (o exactamente una vez).
    def visit_rule(m, i):
        expr = sum(m.y[i, k] for k in m.K)
        if ENFORCE_ALL_VISITS:
            return expr == 1
        else:
            return expr <= 1

    m.visit_rule = pyo.Constraint(m.C, rule=visit_rule)

    # 2) Flujo: si un nodo i es visitado por el equipo k,
    def flow_out_rule(m, i, k):
        return sum(m.x[i, j, k] for j in m.N if j != i) == m.y[i, k]

    m.flow_out = pyo.Constraint(m.C, m.K, rule=flow_out_rule)

    def flow_in_rule(m, i, k):
        return sum(m.x[j, i, k] for j in m.N if j != i) == m.y[i, k]

    m.flow_in = pyo.Constraint(m.C, m.K, rule=flow_in_rule)

    # 3) Cada equipo sale del depósito una vez
    def depot_out_rule(m, k):
        return sum(m.x[DEPOT, j, k] for j in m.C) == 1

    m.depot_out = pyo.Constraint(m.K, rule=depot_out_rule)

    # 4) Cada equipo regresa al depósito una vez
    def depot_in_rule(m, k):
        return sum(m.x[i, DEPOT, k] for i in m.C) == 1

    m.depot_in = pyo.Constraint(m.K, rule=depot_in_rule)

    # 6) Restricciones MTZ para evitar subciclos
    #    u[i,k] - u[j,k] + |C| * x[i,j,k] <= |C|-1
    M_val = len(CLIENTS)

    def mtz_rule(m, i, j, k):
        if i == j:
            return pyo.Constraint.Skip
        return m.u[i, k] - m.u[j, k] + M_val * m.x[i, j, k] <= M_val - 1

    m.mtz = pyo.Constraint(m.C, m.C, m.K, rule=mtz_rule)

    # -------------------------------
    # Expresiones de funciones objetivo
    # -------------------------------
    # Z1 = distancia total (sin i==j)
    def z1_expr(m):
        return sum(
            m.dist[i, j] * m.x[i, j, k]
            for i in m.N for j in m.N for k in m.K
            if i != j
        )

    m.Z1 = pyo.Expression(rule=z1_expr)

    # Z2 = calidad total
    def z2_expr(m):
        return sum(m.qual[i] * m.y[i, k] for i in m.C for k in m.K)

    m.Z2 = pyo.Expression(rule=z2_expr)

    # Z3 = riesgo total (sin i==j)
    def z3_expr(m):
        return sum(
            m.risk[i, j] * m.x[i, j, k]
            for i in m.N for j in m.N for k in m.K
            if i != j
        )

    m.Z3 = pyo.Expression(rule=z3_expr)

    return m


# ============================================================
# 3. FUNCIONES AUXILIARES PARA RESOLVER Y EXTRAER SOLUCIONES
# ============================================================

def solve_model(model, solver_name="glpk", tee=False):
    solver = pyo.SolverFactory(solver_name)
    results = solver.solve(model, tee=tee)
    return results


def extract_routes(model):
    """
    Extrae las rutas para cada equipo como lista de nodos (ciclo).
    Útil para debug o para poner rutas en el informe.
    """
    routes = {}
    for k in model.K:
        arcs = []
        for i in model.N:
            for j in model.N:
                if i != j and pyo.value(model.x[i, j, k]) > 0.5:
                    arcs.append((i, j))

        if not arcs:
            routes[k] = []
            continue

        route = [DEPOT]
        current = DEPOT
        while True:
            next_nodes = [j for (i, j) in arcs if i == current]
            if not next_nodes:
                break
            nxt = next_nodes[0]
            route.append(nxt)
            current = nxt
            if current == DEPOT:
                break

        routes[k] = route
    return routes


# ============================================================
# 4. VALORES EXTREMOS
# ============================================================

def get_extreme_values():
    """
    Calcula:
    - min_dist: minimizando Z1
    - min_risk: minimizando Z3
    - max_qual: maximizando Z2
    Sirve para construir rangos razonables de ε.
    """

    # 1) Minimizar distancia
    m1 = build_base_model()
    m1.obj = pyo.Objective(expr=m1.Z1, sense=pyo.minimize)
    solve_model(m1)
    min_dist = pyo.value(m1.Z1)

    # 2) Minimizar riesgo
    m2 = build_base_model()
    m2.obj = pyo.Objective(expr=m2.Z3, sense=pyo.minimize)
    solve_model(m2)
    min_risk = pyo.value(m2.Z3)

    # 3) Maximizar calidad (minimizando -Z2)
    m3 = build_base_model()
    m3.obj = pyo.Objective(expr=-m3.Z2, sense=pyo.minimize)
    solve_model(m3)
    max_qual = pyo.value(m3.Z2)

    extremes = {
        "min_dist": min_dist,
        "min_risk": min_risk,
        "max_qual": max_qual,
    }
    return extremes


# ============================================================
# 5. MÉTODO ε-CONSTRAINT
# ============================================================

def epsilon_constraint_experiments():
    """
    Aplica ε-constraint para generar un frente de Pareto aproximado.
    """

    extremes = get_extreme_values()
    print("Extreme values:", extremes)

    min_dist = extremes["min_dist"]
    min_risk = extremes["min_risk"]
    max_qual = extremes["max_qual"]

    # RANGOS DE ε:
    # - Distancia: de 1.1 * min_dist hasta 1.8 * min_dist
    # - Riesgo: de 1.1 * min_risk hasta 1.8 * min_risk
    eps_dists = [
        1.1 * min_dist,
        1.2 * min_dist,
        1.3 * min_dist,
        1.5 * min_dist,
        1.8 * min_dist,
    ]

    eps_risks = [
        1.1 * min_risk,
        1.2 * min_risk,
        1.3 * min_risk,
        1.5 * min_risk,
    ]

    solutions = []

    for eps_d in eps_dists:
        for eps_r in eps_risks:
            print(f"\n=== Resolviendo con ε_dist={eps_d:.2f}, ε_risk={eps_r:.2f} ===")
            m = build_base_model()

            # Restricciones ε-constraint
            m.eps_dist = pyo.Constraint(expr=m.Z1 <= eps_d)
            m.eps_risk = pyo.Constraint(expr=m.Z3 <= eps_r)

            # Objetivo: Max calidad -> Min (-Z2)
            m.obj = pyo.Objective(expr=-m.Z2, sense=pyo.minimize)

            results = solve_model(m, tee=False)
            tc = results.solver.termination_condition

            if tc not in (TerminationCondition.optimal,
                          TerminationCondition.feasible):
                print(f"   -> No se obtuvo solución (status: {tc}), se omite esta combinación.")
                continue

            dist_val = pyo.value(m.Z1)
            risk_val = pyo.value(m.Z3)
            qual_val = pyo.value(m.Z2)
            routes = extract_routes(m)

            print(f"   Dist={dist_val:.2f}, Riesgo={risk_val:.2f}, Calidad={qual_val:.2f}")
            for k, r in routes.items():
                print(f"   Ruta equipo {k}: {r}")

            sol = {
                "eps_dist": eps_d,
                "eps_risk": eps_r,
                "dist": dist_val,
                "risk": risk_val,
                "qual": qual_val,
                "routes": routes,
            }
            solutions.append(sol)

    return solutions


# ============================================================
# 6. VISUALIZACIÓN DEL FRENTE DE PARETO
# ============================================================

def plot_pareto_3d(solutions):
    """
    Grafica en 3D el frente aproximado:
      X = distancia
      Y = riesgo
      Z = calidad
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    xs = [s["dist"] for s in solutions]
    ys = [s["risk"] for s in solutions]
    zs = [s["qual"] for s in solutions]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(xs, ys, zs)

    ax.set_xlabel("Distancia total (Z1)")
    ax.set_ylabel("Riesgo total (Z3)")
    ax.set_zlabel("Calidad total (Z2)")

    ax.set_title("Frente de Pareto aproximado - Problema 2 (ε-constraint)")
    plt.tight_layout()
    plt.show()

def plot_dist_vs_risk(solutions):
    """
    Gráfica 2D: Distancia vs Riesgo
    """
    xs = [s["dist"] for s in solutions]
    ys = [s["risk"] for s in solutions]

    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("Distancia total (Z1)")
    plt.ylabel("Riesgo total (Z3)")
    plt.title("Distancia vs Riesgo - Problema 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_compromise_routes():
    """
    Dibuja las rutas de la solución de compromiso seleccionada
    en la sección 2.4.2 del informe.

    Solución:
      (ε1, ε2) = (509.6, 52.8)
      Z1 = 464, Z3 = 52, Z2 = 769

      Equipo 1: 0 → 4 → 7 → 6 → 0
      Equipo 2: 0 → 3 → 1 → 8 → 5 → 9 → 2 → 0
    """
    # Layout simple: nodos en círculo
    coords = {}
    R = 1.0
    for i in NODES:
        angle = 2 * math.pi * i / len(NODES)
        x = R * math.cos(angle)
        y = R * math.sin(angle)
        coords[i] = (x, y)

    fig, ax = plt.subplots()

    # Dibujar todos los nodos
    for node, (x, y) in coords.items():
        ax.scatter(x, y)
        ax.text(x + 0.03, y + 0.03, str(node))

    # Rutas de la solución de compromiso
    route_team1 = [0, 4, 7, 6, 0]
    route_team2 = [0, 3, 1, 8, 5, 9, 2, 0]

    # Dibujar ruta equipo 1
    xs1 = [coords[i][0] for i in route_team1]
    ys1 = [coords[i][1] for i in route_team1]
    ax.plot(xs1, ys1, linestyle="-", marker="o", label="Equipo 1")

    # Dibujar ruta equipo 2
    xs2 = [coords[i][0] for i in route_team2]
    ys2 = [coords[i][1] for i in route_team2]
    ax.plot(xs2, ys2, linestyle="--", marker="o", label="Equipo 2")

    ax.set_title("Rutas solución de compromiso (2.4.2)")
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.legend()
    plt.tight_layout()
    plt.show()



def print_solutions_table(solutions):
    """
    Imprime una tabla con las soluciones obtenidas.
    """
    if not solutions:
        print("\nNo se generó ninguna solución (todas las combinaciones fueron infactibles).")
        return

    print("\n=== Soluciones obtenidas (ε-constraint) ===")
    print("{:>3} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12}".format(
        "#", "eps_dist", "eps_risk", "Dist", "Riesgo", "Calidad"
    ))
    print("-" * 75)
    for idx, s in enumerate(solutions, start=1):
        print("{:>3} | {:>12.2f} | {:>12.2f} | {:>12.2f} | {:>12.2f} | {:>12.2f}".format(
            idx,
            s["eps_dist"],
            s["eps_risk"],
            s["dist"],
            s["risk"],
            s["qual"],
        ))


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":
    print(f"ENFORCE_ALL_VISITS = {ENFORCE_ALL_VISITS}")
    print(f"Número de nodos: {n_nodes}, clientes: {len(CLIENTS)}, equipos: {len(K_TEAMS)}")

    sols = epsilon_constraint_experiments()
    print_solutions_table(sols)

    if sols:
        plot_pareto_3d(sols)
        plot_dist_vs_risk(sols)
        plot_compromise_routes()