"""
FIMD (Fisher Information Maximization Design) for Beaudeau et al. (2023) Model 1
---------------------------------------------------------------------------------
Este script implementa:
  • El Modelo 1 (derivado de Malherbe) con adición de N durante la fermentación
    y perfil de temperatura dependiente del tiempo (k1(T) = alpha_k1*T - beta_k1).
  • Simulación ODE (solve_ivp, método LSODA) con pulsos de N suaves (rectangulares cortos)
    para evitar discontinuidades violentas en el integrador.
  • Cálculo de sensibilidades por diferencias finitas (central) de la salida medida (CO2).
  • Cálculo de la matriz de Información de Fisher: FIM = Σ_t (J(t)^T Σ_y^{-1} J(t)).
  • Selección FIMD codiciosa (greedy) de 21 fermentaciones de un conjunto de candidatos
    con criterios D-óptimos (log det).

Notas:
  - Unidades de N: las dosis de adición de N se definen en mgN/L en el diseño y se
    convierten a gN/L en el modelo.
  - Medición: por defecto sólo CO2 acumulado (g/L) cada hora, con σ=0.2 g/L.
  - Parámetros nominales tomados de la tabla del artículo (ver paper).
  - Temperatura en °C (ej. 16–28). El perfil se parametriza por 3 segmentos
    (T1, T2, T3) y dos tiempos de cambio (t12, t23).

Requisitos: numpy, scipy, dataclasses
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict
from scipy.integrate import solve_ivp

# ===========================
# Modelo 1 de Beaudeau (2023)
# ===========================

@dataclass
class BeaudeauModel1Params:
    # Parámetros cinéticos (valores nominales)
    alpha_k1: float = 0.0287   # h^{-1}/°C
    beta_k1: float  = 0.3      # h^{-1}

    k2: float = 0.0386         # gS/(gN·1e9cells·h)
    KS: float = 20.67          # gS/L
    KSI: float = 0.006299      # (L/gE)^{alpha_S}
    alpha_S: float = 0.9355    # adim

    YE_over_S: float = 2.17    # gS/gE (yield S→E)
    YCO2_over_S: float = 2.17  # gS/gCO2 (yield S→CO2)

    k3: float = 0.001033       # gN/(1e9cells·h)
    KN: float = 0.04105        # gN/L
    KNI: float = 0.02635       # (L/gE)^{alpha_N}
    alpha_N: float = 1.195     # adim

    Q0: float = 0.0001347      # gN/(1e9cells)
    Emax: float = 94.67        # gE/L

    knst: float = 1.0024       # gN/(1e9cells·h)
    kdnst: float = 1.987       # gN/(1e9cells·h)
    KNST: float = 10.76        # gN/(1e9cells)
    Ynst: float = 694.8        # gN/gN (rendimiento NST) — del paper
    Q0nst: float = 0.0001334   # gN/(1e9cells)

    alpha1: float = 0.0004795  # gN/(1e9cells)
    kappa: float = 0.03        # h^{-1}

    # Config del integrador
    atol: float = 1e-8
    rtol: float = 1e-6

@dataclass
class Design:
    # Perfil de temperatura de 3 escalones: [0, t12), [t12, t23), [t23, t_end]
    T1: float
    T2: float
    T3: float
    t12: float
    t23: float
    # Dos pulsos de N (mgN/L) en tiempos tN1, tN2 con duraciones cortas tauN (h)
    doseN1_mgL: float
    tN1: float
    doseN2_mgL: float
    tN2: float
    tauN: float = 0.25  # horas, ancho del pulso rectangular

@dataclass
class SimulationConfig:
    t_end: float = 200.0       # h
    dt: float = 1.0            # muestreo para mediciones (h)
    # Condiciones iniciales típicas
    X0: float = 1.0            # 1e9 cells/L
    S0: float = 180.0          # g/L
    N0_mgL: float = 140.0      # mgN/L
    E0: float = 0.0            # g/L
    CO20: float = 0.0          # g/L
    Nintra0: float = 5e-4      # gN/(1e9cells) (pequeña reserva)
    NST0: float = 1e-3         # gN/(1e9cells)
    A0: float = 1.0            # actividad celular

@dataclass
class MeasurementModel:
    sigma_CO2: float = 0.2     # g/L (desvío estándar)
    measure_CO2: bool = True

# Utilidades de perfil de temperatura y alimentación de N

def T_of_t(t: float, phi: Design) -> float:
    if t < phi.t12:
        return phi.T1
    elif t < phi.t23:
        return phi.T2
    else:
        return phi.T3


def N_feed_rate(t: float, phi: Design) -> float:
    """
    Término de entrada de N al medio (gN/L/h), representado como suma de
    dos pulsos rectangulares de duración tauN y dosis totales especificadas.
    """
    r = 0.0
    for dose_mgL, tN in [(phi.doseN1_mgL, phi.tN1), (phi.doseN2_mgL, phi.tN2)]:
        if dose_mgL <= 0:
            continue
        dose_gL = dose_mgL / 1000.0
        if tN <= t < tN + phi.tauN:
            r += dose_gL / max(phi.tauN, 1e-6)
    return r

# Dinámica del modelo (8 estados): X, S, N, E, CO2, Nintra, NST, A

STATE_IDX = {"X":0, "S":1, "N":2, "E":3, "CO2":4, "Nintra":5, "NST":6, "A":7}


def rhs(t: float, x: np.ndarray, p: BeaudeauModel1Params, phi: Design) -> np.ndarray:
    X, S, N, E, CO2, Nintra, NST, A = x

    # Seguridad ante dominios no físicos
    X = max(X, 1e-9)
    S = max(S, 0.0)
    N = max(N, 0.0)
    Nintra = max(Nintra, 1e-9)
    NST = max(NST, 0.0)
    E = max(E, 0.0)
    A = np.clip(A, 0.0, 1.0)

    # Perfil de temperatura y k1(T)
    T = T_of_t(t, phi)
    k1 = p.alpha_k1 * T - p.beta_k1
    k1 = max(k1, 0.0)  # no permitir crecimiento negativo

    # Velocidad de consumo de azúcar vía transportador con inhibición por EtOH
    nu_ST = p.k2 * S / (S + p.KS + p.KSI * (S * (E ** p.alpha_S)))
    rS = - nu_ST * NST * X

    # Rendimientos a E y CO2
    rE = - (1.0 / p.YE_over_S) * rS
    rCO2 = - (1.0 / p.YCO2_over_S) * rS

    # Síntesis / degradación de NST
    rNST = p.knst * (1.0 - p.Q0nst / Nintra) - p.kdnst * (NST / (p.KNST + NST))

    # Consumo de N con inhibición por EtOH * actividad
    muN = p.k3 * N / (N + p.KN + p.KNI * (E ** p.alpha_N))
    rN = - muN * X * A

    # Nitrogeno intracelular (entra por rN, sale por crecimiento y NST)
    rNintra = muN - p.alpha1 * (k1 * X * (1.0 - p.Q0 / Nintra) * (1.0 - E / p.Emax) * A) \
              - (1.0 / p.Ynst) * (p.knst * (1.0 - p.Q0nst / Nintra) - p.kdnst)

    # Crecimiento de biomasa con efecto de Nintra, E y A
    rX = k1 * X * (1.0 - p.Q0 / Nintra) * (1.0 - E / p.Emax) * A

    # Actividad celular
    dA_dt = (rX / max(X, 1e-9)) * (1.0 - A) - p.kappa * A

    # Entrada externa de N (pulsos) — term source
    rN_ext = N_feed_rate(t, phi)

    # Ensamblar derivadas
    dX = rX
    dS = rS
    dN = rN + rN_ext
    dE = rE
    dCO2 = rCO2
    dNintra = rNintra - Nintra * (rX / max(X, 1e-9))  # dilución
    dNST = rNST - NST * (rX / max(X, 1e-9))           # dilución

    return np.array([dX, dS, dN, dE, dCO2, dNintra, dNST, dA_dt], dtype=float)


# Simulación con eventos de parada opcionales (S → ~0)

def simulate(phi: Design, p: BeaudeauModel1Params, sim: SimulationConfig,
             t_eval: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if t_eval is None:
        t_eval = np.arange(0.0, sim.t_end + 1e-9, sim.dt)

    # Convertir N0
    N0_gL = sim.N0_mgL / 1000.0
    x0 = np.array([
        sim.X0, sim.S0, N0_gL, sim.E0, sim.CO20, sim.Nintra0, sim.NST0, sim.A0
    ], dtype=float)

    def stop_when_sugar_exhausted(t, x):
        return x[STATE_IDX["S"]] - 1.0  # detener si S < 1 g/L
    stop_when_sugar_exhausted.terminal = True
    stop_when_sugar_exhausted.direction = -1

    sol = solve_ivp(
        fun=lambda t, x: rhs(t, x, p, phi),
        t_span=(t_eval[0], t_eval[-1]),
        y0=x0,
        method="LSODA",
        t_eval=t_eval,
        atol=p.atol,
        rtol=p.rtol,
        events=stop_when_sugar_exhausted,
        max_step=0.5,
    )
    if not sol.success:
        raise RuntimeError(f"ODE failed: {sol.message}")
    return sol.t, sol.y.T  # (n_t, n_states)


# Medición (sólo CO2)

def measure(ytraj: np.ndarray, meas: MeasurementModel) -> np.ndarray:
    idx = STATE_IDX["CO2"]
    return ytraj[:, [idx]]  # matriz n_t x 1


# Sensibilidades por diferencias finitas centrales

def sensitivities_fd(theta_names: List[str], p_nom: BeaudeauModel1Params,
                     phi: Design, sim: SimulationConfig,
                     meas: MeasurementModel, rel_eps: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (J, t_eval): Jacobiano concatenado por tiempos (n_t x n_theta) y vector de tiempos.
    """
    # Trayectoria nominal
    t_eval = np.arange(0.0, sim.t_end + 1e-9, sim.dt)
    t_nom, Y_nom = simulate(phi, p_nom, sim, t_eval)
    y_nom = measure(Y_nom, meas)  # n_t x m  (m=1 aquí)

    J = []
    for name in theta_names:
        val = getattr(p_nom, name)
        h = rel_eps * max(abs(val), 1e-6)
        # p+
        p_plus = BeaudeauModel1Params(**{**p_nom.__dict__, name: val + h})
        _, Yp = simulate(phi, p_plus, sim, t_eval)
        yp = measure(Yp, meas)
        # p-
        p_minus = BeaudeauModel1Params(**{**p_nom.__dict__, name: max(val - h, 1e-12)})
        _, Ym = simulate(phi, p_minus, sim, t_eval)
        ym = measure(Ym, meas)
        # derivada central
        dy_dtheta = (yp - ym) / (2.0 * h)  # n_t x m
        # Aquí m=1, así que tomamos columna 0
        J.append(dy_dtheta[:, 0])
    J_mat = np.column_stack(J)  # n_t x n_theta
    return J_mat, t_nom


# Fisher Information Matrix

def fisher_information(J: np.ndarray, meas: MeasurementModel) -> np.ndarray:
    # Σ_y^{-1} (m x m). Aquí m=1.
    inv_sigma2 = 1.0 / (meas.sigma_CO2 ** 2)
    # FIM = Σ_t J(t)^T Σ_y^{-1} J(t)
    # Con m=1, J ya es n_t x n_theta para CO2, entonces:
    F = inv_sigma2 * (J.T @ J)
    return F


# Generación de candidatos aleatorios

def sample_designs(n: int, rng: np.random.Generator, sim: SimulationConfig) -> List[Design]:
    designs: List[Design] = []
    for _ in range(n):
        # Temperaturas en [16, 28] °C
        T1, T2, T3 = rng.uniform(16.0, 28.0, size=3)
        # Tiempos en h: t12 en [6, 48], t23 en [48, 120]
        t12 = rng.uniform(6.0, 48.0)
        t23 = rng.uniform(max(t12 + 6.0, 24.0), 120.0)
        # Pulsos de N: tiempos en [10, 120] y [20, 160], dosis en [0, 200] mg/L
        tN1 = rng.uniform(10.0, 120.0)
        tN2 = rng.uniform(max(tN1 + 6.0, 20.0), 160.0)
        dose1 = rng.uniform(0.0, 200.0)
        dose2 = rng.uniform(0.0, 200.0)
        tauN = rng.uniform(0.1, 0.5)
        designs.append(Design(T1=T1, T2=T2, T3=T3, t12=t12, t23=t23,
                              doseN1_mgL=dose1, tN1=tN1,
                              doseN2_mgL=dose2, tN2=tN2, tauN=tauN))
    return designs


# FIMD codicioso (D-óptimo)

def fimd_select(phi_list: List[Design], theta_names: List[str], p_nom: BeaudeauModel1Params,
                sim: SimulationConfig, meas: MeasurementModel,
                n_select: int, seed: int = 0, verbose: bool = True) -> Tuple[List[int], np.ndarray]:
    rng = np.random.default_rng(seed)
    chosen_idx: List[int] = []
    F_accum = np.zeros((len(theta_names), len(theta_names)))

    for k in range(n_select):
        best_obj = -np.inf
        best_idx = None
        if verbose:
            print("="*80)
            print(f"[FIMD] Iter {k+1}/{n_select} (candidatos={len(phi_list)})")
        for j, phi in enumerate(phi_list):
            if j in chosen_idx:
                continue
            try:
                J, _ = sensitivities_fd(theta_names, p_nom, phi, sim, meas)
                F = fisher_information(J, meas)
            except Exception as e:
                if verbose:
                    print(f"[WARN] candidato {j} falló simulación: {e}")
                continue
            F_total = F_accum + F
            # Criterio D: log det con regularización para estabilidad
            obj = np.linalg.slogdet(F_total + 1e-12*np.eye(F_total.shape[0]))[1]
            if obj > best_obj:
                best_obj = obj
                best_idx = j
        if best_idx is None:
            raise RuntimeError("No se pudo seleccionar ningún experimento — revise el espacio de diseño")
        # Aceptar
        chosen_idx.append(best_idx)
        J_b, _ = sensitivities_fd(theta_names, p_nom, phi_list[best_idx], sim, meas)
        F_accum += fisher_information(J_b, meas)
        if verbose:
            print(f"[FIMD] -> elegido idx={best_idx} | logdet={best_obj:.3f}")
    return chosen_idx, F_accum


# =====================
# Ejecución de ejemplo
# =====================
if __name__ == "__main__":
    # Configuración
    p = BeaudeauModel1Params()
    sim = SimulationConfig(t_end=200.0, dt=1.0, N0_mgL=140.0)
    meas = MeasurementModel(sigma_CO2=0.2)

    # Parámetros a estimar (subconjunto para mantener coste computacional razonable)
    theta_names = [
        "k2", "KS", "KSI", "alpha_S",
        "k3", "KN",
        "Q0", "Emax",
        "Ynst", "Q0nst",
        "alpha_k1", "beta_k1",
    ]

    # Generar candidatos
    seed = 42
    rng = np.random.default_rng(seed)
    n_candidates = 60
    print(f"[MAIN] Generando {n_candidates} candidatos (seed={seed})...")
    candidates = sample_designs(n_candidates, rng, sim)

    # Seleccionar 21 fermentaciones via FIMD
    n_select = 21
    print(f"[MAIN] Seleccionando {n_select} diseños D-óptimos (FIMD)...")
    chosen_idx, F_final = fimd_select(candidates, theta_names, p, sim, meas,
                                      n_select=n_select, seed=seed, verbose=True)

    # Resumen
    print("\nResumen de diseños seleccionados (índice, temperaturas, tiempos, pulsos de N mg/L):")
    for rank, idx in enumerate(chosen_idx, start=1):
        d = candidates[idx]
        print(
            f"#{rank:02d} idx={idx} | T1={d.T1:.1f}°C T2={d.T2:.1f}°C T3={d.T3:.1f}°C | "
            f"t12={d.t12:.1f}h t23={d.t23:.1f}h | N1={d.doseN1_mgL:.0f}mg/L@{d.tN1:.1f}h "
            f"N2={d.doseN2_mgL:.0f}mg/L@{d.tN2:.1f}h τ={d.tauN:.2f}h"
        )

    # Métrica final
    sign, logdet = np.linalg.slogdet(F_final + 1e-12*np.eye(len(theta_names)))
    print(f"\n[RESULT] log det(FIM acumulada) = {logdet:.3f}")

    # Ejemplo: simular el primer diseño elegido y mostrar tiempo de fin y CO2 final
    first = candidates[chosen_idx[0]]
    t, Y = simulate(first, p, sim)
    CO2_final = Y[-1, STATE_IDX["CO2"]]
    S_final = Y[-1, STATE_IDX["S"]]
    print(f"\n[CHECK] Diseño top idx={chosen_idx[0]}: t_end_sim={t[-1]:.1f} h | CO2f={CO2_final:.1f} g/L | Sf={S_final:.1f} g/L")
