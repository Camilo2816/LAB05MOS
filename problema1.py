# -*- coding: utf-8 -*-

import copy
import math
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    NonNegativeReals, Binary, Integers, minimize, maximize,
    SolverFactory, value
)

# Solver
SOLVER_NAME = 'glpk'
SOLVER_TIME_LIMIT = None

if SOLVER_TIME_LIMIT is None:
    SOLVER = SolverFactory(SOLVER_NAME)
else:
    SOLVER = SolverFactory(SOLVER_NAME)
    try:
        SOLVER.options['tmlim'] = int(SOLVER_TIME_LIMIT)
    except Exception:
        pass

# Datos del problema
def datos_por_defecto():
    Recursos = ['alimentos basicos', 'medicinas', 'equipos medicos',
                'agua potable', 'mantas']
    ValorRecursos = {'alimentos basicos':50, 'medicinas':100, 'equipos medicos':120,
                    'agua potable':60, 'mantas':40}
    PesoRecursos = {'alimentos basicos':5, 'medicinas':2, 'equipos medicos':0.3,
                    'agua potable':6, 'mantas':3}
    VolumenRecursos = {'alimentos basicos':3, 'medicinas':1, 'equipos medicos':0.5,
                       'agua potable':4, 'mantas':2}
    DisponibilidadRecursos = {'alimentos basicos':12, 'medicinas':15, 'equipos medicos':40,
                       'agua potable':15, 'mantas':20}

    Aviones = ['1', '2', '3', '4']
    CapacidadPesoAviones = {'1':40, '2':50, '3':60, '4':45}
    CapacidadVolAviones = {'1':35, '2':40, '3':45, '4':38}
    CostoFijoAviones = {'1':15, '2':20, '3':25, '4':18}
    CostoVarAviones = {'1':0.02, '2':0.025, '3':0.030, '4':0.022}

    Zonas = ['A', 'B', 'C', 'D']
    DistanciaZonas = {'A':800, 'B':1200, 'C':1500, 'D':900}
    PoblacionZonas = {'A':50, 'B':70, 'C':100, 'D':80}
    MultiplicadorZonas = {'A':1.2, 'B':1.5, 'C':1.8, 'D':1.4}

    NecesidadesZonaA = {'alimentos basicos':8, 'agua potable':6,
                        'medicinas':2, 'equipos medicos':0.6,
                        'mantas':3}
    NecesidadesZonaB = {'alimentos basicos':12, 'agua potable':9,
                        'medicinas':3, 'equipos medicos':0.9,
                        'mantas':5}
    NecesidadesZonaC = {'alimentos basicos':8, 'agua potable':12,
                        'medicinas':4, 'equipos medicos':1.2,
                        'mantas':7}
    NecesidadesZonaD = {'alimentos basicos':8, 'agua potable':8,
                        'medicinas':2, 'equipos medicos':0.6,
                        'mantas':4}

    max_costo_transporte = 1e6
    return (Recursos, ValorRecursos, PesoRecursos, VolumenRecursos, DisponibilidadRecursos,
            Aviones, CapacidadPesoAviones, CapacidadVolAviones, CostoFijoAviones, CostoVarAviones,
            Zonas, DistanciaZonas, PoblacionZonas, MultiplicadorZonas,
            NecesidadesZonaA, NecesidadesZonaB, NecesidadesZonaC, NecesidadesZonaD,
            max_costo_transporte)

# Modelo Pyomo
def construir_modelo(paramtuple):
    (Recursos, ValorRecursos, PesoRecursos, VolumenRecursos, DisponibilidadRecursos,
     Aviones, CapacidadPesoAviones, CapacidadVolAviones, CostoFijoAviones, CostoVarAviones,
     Zonas, DistanciaZonas, PoblacionZonas, MultiplicadorZonas,
     NecesidadesZonaA, NecesidadesZonaB, NecesidadesZonaC, NecesidadesZonaD,
     max_costo_transporte) = paramtuple

    m = ConcreteModel()

    # Conjuntos
    m.Recursos = Set(initialize=Recursos)
    m.Aviones = Set(initialize=Aviones)
    m.Zonas = Set(initialize=Zonas)

    # Parámetros
    m.ValorRecursos = Param(m.Recursos, initialize=ValorRecursos)
    m.PesoRecursos = Param(m.Recursos, initialize=PesoRecursos)
    m.VolumenRecursos = Param(m.Recursos, initialize=VolumenRecursos)
    m.DisponibilidadRecursos = Param(m.Recursos, initialize=DisponibilidadRecursos)

    m.CapacidadPesoAviones = Param(m.Aviones, initialize=CapacidadPesoAviones)
    m.CapacidadVolAviones = Param(m.Aviones, initialize=CapacidadVolAviones)
    m.CostoFijoAviones = Param(m.Aviones, initialize=CostoFijoAviones)
    m.CostoVarAviones = Param(m.Aviones, initialize=CostoVarAviones)

    m.DistanciaZonas = Param(m.Zonas, initialize=DistanciaZonas)
    m.MultiplicadorZonas = Param(m.Zonas, initialize=MultiplicadorZonas)

    m.NecesidadesZonaA = Param(m.Recursos, initialize=NecesidadesZonaA)
    m.NecesidadesZonaB = Param(m.Recursos, initialize=NecesidadesZonaB)
    m.NecesidadesZonaC = Param(m.Recursos, initialize=NecesidadesZonaC)
    m.NecesidadesZonaD = Param(m.Recursos, initialize=NecesidadesZonaD)

    # Variables
    m.X = Var(m.Recursos, m.Aviones, m.Zonas, domain=NonNegativeReals)
    m.U_unidades = Var(m.Recursos, m.Aviones, m.Zonas, domain=NonNegativeReals)
    m.U_equipos = Var(m.Aviones, m.Zonas, domain=Integers, bounds=(0, None))
    m.Y = Var(m.Aviones, m.Zonas, domain=Binary)
    m.U = Var(m.Aviones, domain=Binary)

    # Lógica tonelada–unidad
    def relacion_toneladas_unidades(mdl, i, j, k):
        return mdl.X[i,j,k] == mdl.U_unidades[i,j,k] * mdl.PesoRecursos[i]
    m.relacion_ton_unid = Constraint(m.Recursos, m.Aviones, m.Zonas, rule=relacion_toneladas_unidades)

    # Equipos enteros
    def equipos_enteros(mdl, j, k):
        return mdl.U_unidades['equipos medicos', j, k] == mdl.U_equipos[j,k]
    m.equipos_enteros = Constraint(m.Aviones, m.Zonas, rule=equipos_enteros)

    # Restricción de disponibilidad
    def disponibilidad(mdl, i):
        return sum(mdl.U_unidades[i,j,k] for j in mdl.Aviones for k in mdl.Zonas) <= mdl.DisponibilidadRecursos[i]
    m.disponibilidad = Constraint(m.Recursos, rule=disponibilidad)

    # Activación Y
    def activacion_Y(mdl, i, j, k):
        return mdl.U_unidades[i,j,k] <= mdl.Y[j,k] * mdl.DisponibilidadRecursos[i]
    m.activacion = Constraint(m.Recursos, m.Aviones, m.Zonas, rule=activacion_Y)

    # Necesidades
    def necesidades_A(mdl, i): return sum(mdl.X[i,j,'A'] for j in mdl.Aviones) >= mdl.NecesidadesZonaA[i]
    def necesidades_B(mdl, i): return sum(mdl.X[i,j,'B'] for j in mdl.Aviones) >= mdl.NecesidadesZonaB[i]
    def necesidades_C(mdl, i): return sum(mdl.X[i,j,'C'] for j in mdl.Aviones) >= mdl.NecesidadesZonaC[i]
    def necesidades_D(mdl, i): return sum(mdl.X[i,j,'D'] for j in mdl.Aviones) >= mdl.NecesidadesZonaD[i]

    m.necesidades_A = Constraint(m.Recursos, rule=necesidades_A)
    m.necesidades_B = Constraint(m.Recursos, rule=necesidades_B)
    m.necesidades_C = Constraint(m.Recursos, rule=necesidades_C)
    m.necesidades_D = Constraint(m.Recursos, rule=necesidades_D)

    # Hasta 2 zonas por avión
    def max_zonas_por_avion(mdl, j):
        return sum(mdl.Y[j,k] for k in mdl.Zonas) <= 2
    m.max_zonas = Constraint(m.Aviones, rule=max_zonas_por_avion)

    # Restricción avión 1 sin medicinas
    def medicinas_no_avion1(mdl, k):
        return mdl.U_unidades['medicinas','1',k] == 0
    m.medicinas_no_avion1 = Constraint(m.Zonas, rule=medicinas_no_avion1)

    # Capacidad volumen
    def capacidad_volumen(mdl, j, k):
        vol = sum(mdl.U_unidades[i,j,k] * mdl.VolumenRecursos[i] for i in mdl.Recursos)
        return vol <= mdl.CapacidadVolAviones[j] * mdl.Y[j,k]
    m.capacidad_volumen = Constraint(m.Aviones, m.Zonas, rule=capacidad_volumen)

    # Capacidad peso
    def capacidad_peso(mdl, j, k):
        peso = sum(mdl.X[i,j,k] for i in mdl.Recursos)
        return peso <= mdl.CapacidadPesoAviones[j] * mdl.Y[j,k]
    m.capacidad_peso = Constraint(m.Aviones, m.Zonas, rule=capacidad_peso)

    # Relación Y-U
    def relacion_Y_U(mdl, j, k):
        return mdl.Y[j,k] <= mdl.U[j]
    m.relacion_Y_U = Constraint(m.Aviones, m.Zonas, rule=relacion_Y_U)

    # Cobertura mínima
    def cobertura_zonas(mdl, k):
        return sum(mdl.Y[j,k] for j in mdl.Aviones) >= 1
    m.cobertura = Constraint(m.Zonas, rule=cobertura_zonas)

    return m

# Impacto social
def expr_impacto(modelo):
    return sum(modelo.ValorRecursos[i] * modelo.X[i,j,k] * modelo.MultiplicadorZonas[k]
               for i in modelo.Recursos for j in modelo.Aviones for k in modelo.Zonas)

# Costo total
def expr_costo(modelo):
    fijo = sum(modelo.CostoFijoAviones[j] * modelo.U[j] for j in modelo.Aviones)
    var = sum(modelo.CostoVarAviones[j] * modelo.DistanciaZonas[k] * modelo.Y[j,k]
              for j in modelo.Aviones for k in modelo.Zonas)
    return fijo + var

# Resolver modelo con un objetivo
def resolver_modelo_con_obj(modelo, obj_expr, sense=maximize):
    if hasattr(modelo, 'obj'):
        try:
            del modelo.obj
        except Exception:
            pass
    modelo.obj = Objective(expr=obj_expr, sense=sense)
    sol = SOLVER.solve(modelo, tee=False)
    z1 = value(expr_impacto(modelo))
    z2 = value(expr_costo(modelo))
    return z1, z2, copy.deepcopy(modelo), sol

# Extremos mono-objetivo
def calcular_extremos(paramtuple):
    m1 = construir_modelo(paramtuple)
    z1_max, _, m_z1max, _ = resolver_modelo_con_obj(m1, expr_impacto(m1), sense=maximize)

    m2 = construir_modelo(paramtuple)
    _, z2_min, m_z2min, _ = resolver_modelo_con_obj(m2, expr_costo(m2), sense=minimize)

    m3 = construir_modelo(paramtuple)
    z1_min, _, m_z1min, _ = resolver_modelo_con_obj(m3, expr_impacto(m3), sense=minimize)

    m4 = construir_modelo(paramtuple)
    _, z2_max, m_z2max, _ = resolver_modelo_con_obj(m4, expr_costo(m4), sense=maximize)

    return {
        "z1_max": z1_max,
        "z1_min": z1_min,
        "z2_min": z2_min,
        "z2_max": z2_max,
        "models": {"z1max": m_z1max, "z2min": m_z2min, "z1min": m_z1min, "z2max": m_z2max}
    }

# Método suma ponderada
def metodo_suma_ponderada(paramtuple, extremos, alphas):
    results = []
    z1min = extremos['z1_min']; z1max = extremos['z1_max']
    z2min = extremos['z2_min']; z2max = extremos['z2_max']

    for alpha in alphas:
        m = construir_modelo(paramtuple)
        def objetivo_ponderado(mdl):
            Z1 = expr_impacto(mdl)
            Z2 = expr_costo(mdl)
            Z1n = (Z1 - z1min) / (z1max - z1min) if (z1max - z1min) > 0 else Z1
            Z2n = (z2max - Z2) / (z2max - z2min) if (z2max - z2min) > 0 else Z2
            return alpha * Z1n + (1 - alpha) * Z2n
        z1, z2, msol, sol = resolver_modelo_con_obj(m, objetivo_ponderado(m), sense=maximize)
        results.append({"alpha": alpha, "z1": z1, "z2": z2, "model": msol})
    return results

# Método epsilon-constraint
def metodo_epsilon_constraint(paramtuple, extremos, epsilons):
    results = []
    for eps in epsilons:
        m = construir_modelo(paramtuple)
        m.costo_lim = Constraint(expr=expr_costo(m) <= eps)
        z1, z2, msol, sol = resolver_modelo_con_obj(m, expr_impacto(m), sense=maximize)
        results.append({"eps": eps, "z1": z1, "z2": z2, "model": msol})
    return results

# Método lexicográfico
def metodo_lexicografico(paramtuple, nivel_fracciones=5):
    res = []
    extremos = calcular_extremos(paramtuple)
    z1max = extremos['z1_max']
    betas = np.linspace(0.8, 1.0, nivel_fracciones)
    for beta in betas:
        m = construir_modelo(paramtuple)
        m.z1_min_req = Constraint(expr=expr_impacto(m) >= beta * z1max)
        z1, z2, msol, sol = resolver_modelo_con_obj(m, expr_costo(m), sense=minimize)
        res.append({"beta": beta, "z1": z1, "z2": z2, "model": msol})
    return res

# Guardar resultados CSV
def guardar_resumen_csv(result_weighted, result_eps, carpeta='salida'):
    os.makedirs(carpeta, exist_ok=True)
    dfw = pd.DataFrame([{'alpha': r['alpha'], 'z1': r['z1'], 'z2': r['z2']} for r in result_weighted])
    dfe = pd.DataFrame([{'eps': r['eps'], 'z1': r['z1'], 'z2': r['z2']} for r in result_eps])
    dfw.to_csv(os.path.join(carpeta, 'resultados_weighted.csv'), index=False)
    dfe.to_csv(os.path.join(carpeta, 'resultados_epsilon.csv'), index=False)
    print(f"CSV de resumen guardados en: {carpeta}/resultados_weighted.csv  y  {carpeta}/resultados_epsilon.csv")

# Gráfica del frente de Pareto
def graficar_pareto(result_weighted, result_eps, ruta=os.path.join('salida','pareto.png'), titulo_extra=""):
    pts_w = np.array([[r['z2'], r['z1']] for r in result_weighted])
    pts_e = np.array([[r['z2'], r['z1']] for r in result_eps])
    plt.figure(figsize=(9,6))
    if pts_w.size:
        plt.scatter(pts_w[:,0], pts_w[:,1], label='Suma ponderada', marker='o')
    if pts_e.size:
        plt.scatter(pts_e[:,0], pts_e[:,1], label='Epsilon-constraint', marker='x')
    for r in result_weighted:
        plt.annotate(f"{r['alpha']:.2f}", (r['z2'], r['z1']))
    for r in result_eps:
        plt.annotate(f"{r['eps']:.1f}", (r['z2'], r['z1']))
    plt.xlabel("Costo total (miles USD)")
    plt.ylabel("Impacto social (miles USD)")
    plt.title("Frente de Pareto aproximado" + (" - " + titulo_extra if titulo_extra else ""))
    plt.legend()
    plt.grid(True)
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"Gráfico Pareto guardado en {ruta}")

# Exportar detalle de una solución
def export_detalle_sol(modelo_sol, carpeta='salida', nombre='mejor_sol_detalle.csv'):
    os.makedirs(carpeta, exist_ok=True)
    rows = []
    if not hasattr(modelo_sol, 'Recursos'):
        print("Modelo solución no tiene atributos esperados. No se exporta detalle.")
        return
    for i in modelo_sol.Recursos:
        for j in modelo_sol.Aviones:
            for k in modelo_sol.Zonas:
                try:
                    ton = value(modelo_sol.X[i,j,k])
                except Exception:
                    ton = 0.0
                if ton and ton > 1e-6:
                    try:
                        unidades = value(modelo_sol.U_unidades[i,j,k])
                    except Exception:
                        unidades = None
                    rows.append({
                        'Recurso': i,
                        'Avion': j,
                        'Zona': k,
                        'TON': round(float(ton),4),
                        'Unidades': (round(float(unidades),4) if unidades is not None else '')
                    })
    df = pd.DataFrame(rows)
    ruta = os.path.join(carpeta, nombre)
    df.to_csv(ruta, index=False)
    print(f"Detalle de solución guardado en {ruta}")

# Sensibilidad en un multiplicador de zona
def sensibilidad_multiplicador(paramtuple, zona='C', scale=5.0):
    (Recursos, ValorRecursos, PesoRecursos, VolumenRecursos, DisponibilidadRecursos,
     Aviones, CapacidadPesoAviones, CapacidadVolAviones, CostoFijoAviones, CostoVarAviones,
     Zonas, DistanciaZonas, PoblacionZonas, MultiplicadorZonas,
     NecesidadesZonaA, NecesidadesZonaB, NecesidadesZonaC, NecesidadesZonaD,
     max_costo_transporte) = paramtuple

    MultiplicadorZonas_scaled = MultiplicadorZonas.copy()
    if zona in MultiplicadorZonas_scaled:
        MultiplicadorZonas_scaled[zona] *= scale

    paramtuple_scaled = (Recursos, ValorRecursos, PesoRecursos, VolumenRecursos, DisponibilidadRecursos,
                         Aviones, CapacidadPesoAviones, CapacidadVolAviones, CostoFijoAviones, CostoVarAviones,
                         Zonas, DistanciaZonas, PoblacionZonas, MultiplicadorZonas_scaled,
                         NecesidadesZonaA, NecesidadesZonaB, NecesidadesZonaC, NecesidadesZonaD,
                         max_costo_transporte)

    extremos_scaled = calcular_extremos(paramtuple_scaled)
    alphas = np.linspace(0.0, 1.0, 7)
    epsilons = np.linspace(extremos_scaled['z2_min'], extremos_scaled['z2_max'], 7)

    res_w = metodo_suma_ponderada(paramtuple_scaled, extremos_scaled, alphas)
    res_e = metodo_epsilon_constraint(paramtuple_scaled, extremos_scaled, epsilons)

    carpeta = os.path.join('salida', f'sensibilidad_{zona}')
    os.makedirs(carpeta, exist_ok=True)

    pd.DataFrame([{'alpha': r['alpha'], 'z1': r['z1'], 'z2': r['z2']} for r in res_w]).to_csv(os.path.join(carpeta, 'weighted.csv'), index=False)
    pd.DataFrame([{'eps': r['eps'], 'z1': r['z1'], 'z2': r['z2']} for r in res_e]).to_csv(os.path.join(carpeta, 'epsilon.csv'), index=False)

    graficar_pareto(res_w, res_e, ruta=os.path.join(carpeta, 'pareto_sensibilidad.png'),
                    titulo_extra=f"Sensibilidad zona {zona} x{scale}")

    print(f"Sensibilidad guardada en {carpeta}")
    return res_w, res_e

def main():
    print("Problema 1: Optimización Multiobjetivo")
    paramtuple = datos_por_defecto()

    print("Calculando extremos (problemas mono-objetivo)...")
    extremos = calcular_extremos(paramtuple)
    print("Extremos encontrados:")
    print(f"  Z1_min = {extremos['z1_min']:.4f}, Z1_max = {extremos['z1_max']:.4f}")
    print(f"  Z2_min = {extremos['z2_min']:.4f}, Z2_max = {extremos['z2_max']:.4f}")

    alphas = np.linspace(0.0, 1.0, 7)
    epsilons = np.linspace(extremos['z2_min'], extremos['z2_max'], 7)

    print("\nEjecutando método: suma ponderada (alphas):", np.round(alphas,3))
    resultados_weighted = metodo_suma_ponderada(paramtuple, extremos, alphas)

    print("\nEjecutando método: epsilon-constraint (epsilons):", np.round(epsilons,3))
    resultados_eps = metodo_epsilon_constraint(paramtuple, extremos, epsilons)

    print("\nEjecutando método: lexicográfico (breve serie de betas)...")
    resultados_lex = metodo_lexicografico(paramtuple, nivel_fracciones=5)

    print("\nGuardando resultados y graficando frente de Pareto...")
    guardar_resumen_csv(resultados_weighted, resultados_eps, carpeta='salida')
    graficar_pareto(resultados_weighted, resultados_eps, ruta=os.path.join('salida','pareto.png'))

    print("\nSeleccionando solución de compromiso (distancia euclidiana al ideal en espacio normalizado)...")
    z1min = extremos['z1_min']; z1max = extremos['z1_max']
    z2min = extremos['z2_min']; z2max = extremos['z2_max']

    best = None; bestdist = 1e12; best_model = None; best_tag = None

    for r in resultados_weighted:
        zn1 = (r['z1'] - z1min) / (z1max - z1min) if (z1max - z1min) > 0 else r['z1']
        zn2 = (r['z2'] - z2min) / (z2max - z2min) if (z2max - z2min) > 0 else r['z2']
        d = math.sqrt((1 - zn1)**2 + (0 - zn2)**2)
        if d < bestdist:
            bestdist = d; best = r; best_model = r['model']; best_tag = ('weighted', r['alpha'])

    for r in resultados_eps:
        zn1 = (r['z1'] - z1min) / (z1max - z1min) if (z1max - z1min) > 0 else r['z1']
        zn2 = (r['z2'] - z2min) / (z2max - z2min) if (z2max - z2min) > 0 else r['z2']
        d = math.sqrt((1 - zn1)**2 + (0 - zn2)**2)
        if d < bestdist:
            bestdist = d; best = r; best_model = r['model']; best_tag = ('eps', r['eps'])

    print("Mejor solución encontrada (tag, valor):", best_tag)
    print(f"  Z1 = {best['z1']:.4f} (impacto), Z2 = {best['z2']:.4f} (costo), dist_norm = {bestdist:.6f}")

    if best_model is not None:
        export_detalle_sol(best_model, carpeta='salida', nombre='mejor_sol_detalle.csv')

    print("\nEjecutando análisis de sensibilidad: multiplicador zona C x5 ...")
    sensibilidad_multiplicador(paramtuple, zona='C', scale=5.0)

    print("\nSoluciones:")
    for r in resultados_lex:
        print(f"  beta={r['beta']:.3f}  => Z1={r['z1']:.3f}, Z2={r['z2']:.3f}")

    print("\nFinalizado. Revisa la carpeta 'salida/' para CSVs, gráficos y detalle de la mejor solución.")
    print("Archivos clave: salida/resultados_weighted.csv, salida/resultados_epsilon.csv, salida/pareto.png, salida/mejor_sol_detalle.csv")
    return

if __name__ == "__main__":
    main()

