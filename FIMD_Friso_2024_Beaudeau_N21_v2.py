"""
FIMD (D-optimal Experimental Design) for Beaudeau et al. (2023) Model 1
with multi-output measurements (X,S,N,E) and work-hours sampling constraints.
-----------------------------------------------------------------------------

Qué hace este script
- Usa el Modelo 1 de Beaudeau (X,S,N,E,CO2,Nintra,NST,A) con T por escalones y
  2 pulsos de N.
- Considera que medimos: X, S, N, E (no CO2), con ruidos configurables.
- Para **cada candidato** de diseño, selecciona **tiempos de muestreo óptimos**
  bajo restricciones operativas (L–V, 08:00–17:00; máx. 2 muestras por día) y
  con un presupuesto `K_per_exp` por experimento usando un **greedy D-óptimo**
  (Woodbury rank-m, m=4) para construir la FIM de ese candidato.
- Selección **codiciosa** de `n_select` fermentaciones que maximizan el logdet
  de la FIM acumulada.

Puntos clave
- Si un diseño excita el sistema fuera de horario hábil, el algoritmo **solo**
  considera tiempos de medición válidos: el diseño pierde información efectiva
  y tenderá a no ser elegido. Opcionalmente, podemos forzar que los pulsos de N
  caigan **dentro** del horario hábil (`feeds_in_workhours=True`).

Requisitos: numpy, scipy
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.integrate import solve_ivp
import numpy.linalg as npl

STATE_IDX = {"X":0, "S":1, "N":2, "E":3, "CO2":4, "Nintra":5, "NST":6, "A":7}

# ===========================
# Modelo 1 de Beaudeau (2023)
# ===========================
@dataclass
class BeaudeauModel1Params:
    alpha_k1: float = 0.0287   # h^{-1}/°C
    beta_k1: float  = 0.3      # h^{-1}
    k2: float = 0.0386         # gS/(gN·1e9cells·h)
    KS: float = 20.67          # gS/L
    KSI: float = 0.006299      # (L/gE)^{alpha_S}
    alpha_S: float = 0.9355    # adim
    YE_over_S: float = 2.17    # gS/gE
    YCO2_over_S: float = 2.17  # gS/gCO2
    k3: float = 0.001033       # gN/(1e9cells·h)
    KN: float = 0.04105        # gN/L
    KNI: float = 0.02635       # (L/gE)^{alpha_N}
    alpha_N: float = 1.195     # adim
    Q0: float = 0.0001347      # gN/(1e9cells)
    Emax: float = 94.67        # gE/L
    knst: float = 1.0024       # gN/(1e9cells·h)
    kdnst: float = 1.987       # gN/(1e9cells·h)
    KNST: float = 10.76        # gN/(1e9cells)
    Ynst: float = 694.8        # gN/gN
    Q0nst: float = 0.0001334   # gN/(1e9cells)
    alpha1: float = 0.0004795  # gN/(1e9cells)
    kappa: float = 0.03        # h^{-1}
    atol: float = 1e-8
    rtol: float = 1e-6

@dataclass
class Design:
    T1: float; T2: float; T3: float
    t12: float; t23: float
    doseN1_mgL: float; tN1: float
    doseN2_mgL: float; tN2: float
    tauN: float = 0.25

@dataclass
class SimulationConfig:
    t_end: float = 200.0
    dt: float = 1.0
    X0: float = 1.0
    S0: float = 180.0
    N0_mgL: float = 140.0
    E0: float = 0.0
    CO20: float = 0.0
    Nintra0: float = 5e-4
    NST0: float = 1e-3
    A0: float = 1.0
    method: str = "Radau"           # robusto para rigidez
    stop_on_s_exhausted: bool = False # mantener línea de tiempo
    # Calendario operativo
    start_weekday: int = 0      # 0=Lunes
    start_hour: int = 8         # 08:00
    end_hour_exclusive: int = 17# 17:00 → horas válidas 08..16

@dataclass
class MeasurementModel:
    # ruido (desvío estándar) para X,S,N,E
    sigma_X: float = 0.05   # 1e9 cells/L
    sigma_S: float = 1.0    # g/L
    sigma_N: float = 0.01   # g/L
    sigma_E: float = 0.2    # g/L
    @property
    def Sigma(self) -> np.ndarray:
        return np.diag([self.sigma_X**2, self.sigma_S**2, self.sigma_N**2, self.sigma_E**2])

# ===========================
# Utilidades de diseño/tiempo
# ===========================

def T_of_t(t: float, phi: Design) -> float:
    if t < phi.t12: return phi.T1
    elif t < phi.t23: return phi.T2
    else: return phi.T3

def N_feed_rate(t: float, phi: Design) -> float:
    r = 0.0
    for dose_mgL, tN in [(phi.doseN1_mgL, phi.tN1), (phi.doseN2_mgL, phi.tN2)]:
        if dose_mgL <= 0: continue
        dose_gL = dose_mgL / 1000.0
        if tN <= t < tN + phi.tauN:
            r += dose_gL / max(phi.tauN, 1e-6)
    return r

# horario hábil

def is_workhour(t: float, sim: SimulationConfig) -> bool:
    abs_hour = sim.start_hour + t
    day_index = int(np.floor(abs_hour / 24.0))
    weekday = (day_index + sim.start_weekday) % 7  # 0=Lunes
    hour_of_day = abs_hour % 24.0
    return (weekday <= 4) and (sim.start_hour <= hour_of_day < sim.end_hour_exclusive)

def compute_calendar_masks(t_eval: np.ndarray, sim: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
    abs_hour = sim.start_hour + t_eval
    day_index = np.floor(abs_hour / 24.0).astype(int)
    hour_of_day = np.mod(abs_hour, 24.0)
    weekday = np.mod(day_index + sim.start_weekday, 7)
    in_hours = (hour_of_day >= sim.start_hour) & (hour_of_day < sim.end_hour_exclusive)
    is_weekday = (weekday <= 4)
    allowed = in_hours & is_weekday
    return allowed, day_index

# =================
# Dinámica del ODE
# =================

def rhs(t: float, x: np.ndarray, p: BeaudeauModel1Params, phi: Design) -> np.ndarray:
    X, S, N, E, CO2, Nintra, NST, A = x
    X = max(X, 1e-9); S = max(S, 0.0); N = max(N, 0.0)
    Nintra = max(Nintra, 1e-9); NST = max(NST, 0.0); E = max(E, 0.0)
    A = np.clip(A, 0.0, 1.0)
    T = T_of_t(t, phi)
    k1 = max(p.alpha_k1 * T - p.beta_k1, 0.0)
    nu_ST = p.k2 * S / (S + p.KS + p.KSI * (S * (E ** p.alpha_S)))
    rS = - nu_ST * NST * X
    rE = - (1.0 / p.YE_over_S) * rS
    rCO2 = - (1.0 / p.YCO2_over_S) * rS
    rNST = p.knst * (1.0 - p.Q0nst / Nintra) - p.kdnst * (NST / (p.KNST + NST))
    muN = p.k3 * N / (N + p.KN + p.KNI * (E ** p.alpha_N))
    rN = - muN * X * A
    rNintra = muN - p.alpha1 * (k1 * X * (1.0 - p.Q0 / Nintra) * (1.0 - E / p.Emax) * A) \
              - (1.0 / p.Ynst) * (p.knst * (1.0 - p.Q0nst / Nintra) - p.kdnst)
    rX = k1 * X * (1.0 - p.Q0 / Nintra) * (1.0 - E / p.Emax) * A
    dA_dt = (rX / max(X, 1e-9)) * (1.0 - A) - p.kappa * A
    rN_ext = N_feed_rate(t, phi)
    dX = rX; dS = rS; dN = rN + rN_ext; dE = rE; dCO2 = rCO2
    dNintra = rNintra - Nintra * (rX / max(X, 1e-9))
    dNST = rNST - NST * (rX / max(X, 1e-9))
    return np.array([dX, dS, dN, dE, dCO2, dNintra, dNST, dA_dt], dtype=float)


def simulate(phi: Design, p: BeaudeauModel1Params, sim: SimulationConfig,
             t_eval: np.ndarray | None = None, use_event: bool | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if t_eval is None: t_eval = np.arange(0.0, sim.t_end + 1e-9, sim.dt)
    if use_event is None: use_event = sim.stop_on_s_exhausted
    N0_gL = sim.N0_mgL / 1000.0
    x0 = np.array([sim.X0, sim.S0, N0_gL, sim.E0, sim.CO20, sim.Nintra0, sim.NST0, sim.A0], dtype=float)
    events=None
    if use_event:
        def stop_when_sugar_exhausted(t, x):
            return x[STATE_IDX["S"]] - 1.0
        stop_when_sugar_exhausted.terminal=True; stop_when_sugar_exhausted.direction=-1
        events=stop_when_sugar_exhausted
    sol = solve_ivp(lambda t,x: rhs(t,x,p,phi), (t_eval[0], t_eval[-1]), x0,
                    method=sim.method, t_eval=t_eval, atol=p.atol, rtol=p.rtol,
                    events=events, max_step=0.5)
    if not sol.success:
        raise RuntimeError(f"ODE failed: {sol.message}")
    return sol.t, sol.y.T

# ===============================
# Medición y sensibilidades (m=4)
# ===============================

def measure_XSNE(ytraj: np.ndarray) -> np.ndarray:
    idxs = [STATE_IDX["X"], STATE_IDX["S"], STATE_IDX["N"], STATE_IDX["E"]]
    return ytraj[:, idxs]  # (n_t x 4)


def sensitivities_fd_multi(theta_names: List[str], p_nom: BeaudeauModel1Params,
                           phi: Design, sim: SimulationConfig, rel_eps: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    t_eval = np.arange(0.0, sim.t_end + 1e-9, sim.dt)
    _, Y_nom = simulate(phi, p_nom, sim, t_eval, use_event=False)
    Y_nom_m = measure_XSNE(Y_nom)
    n_t, m = Y_nom_m.shape
    n_theta = len(theta_names)
    J = np.zeros((n_t, m, n_theta))
    for k, name in enumerate(theta_names):
        val = getattr(p_nom, name)
        h = rel_eps * max(abs(val), 1e-6)
        p_plus  = BeaudeauModel1Params(**{**p_nom.__dict__, name: val + h})
        p_minus = BeaudeauModel1Params(**{**p_nom.__dict__, name: max(val - h, 1e-12)})
        _, Yp = simulate(phi, p_plus,  sim, t_eval, use_event=False)
        _, Ym = simulate(phi, p_minus, sim, t_eval, use_event=False)
        J[:, :, k] = (measure_XSNE(Yp) - measure_XSNE(Ym)) / (2.0 * h)
    # sanitizar (por si acaso)
    return np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0), t_eval

# =======================================
# Greedy de tiempos (rank-m, robusto SPD)
# =======================================

def _sym(M: np.ndarray) -> np.ndarray: return 0.5*(M+M.T)

def chol_logdet(M: np.ndarray, jitter: float = 1e-12, max_tries: int = 6) -> float:
    eps = jitter
    for _ in range(max_tries):
        try:
            L = npl.cholesky(_sym(M) + eps*np.eye(M.shape[0]))
            return 2.0*float(np.sum(np.log(np.diag(L))))
        except npl.LinAlgError:
            eps *= 10.0
    sign, ld = npl.slogdet(_sym(M) + eps*np.eye(M.shape[0]))
    return float(ld)

def robust_inv_spd(M: np.ndarray, jitter: float = 1e-12, max_tries: int = 6) -> np.ndarray:
    eps=jitter
    for _ in range(max_tries):
        try:
            L = npl.cholesky(_sym(M) + eps*np.eye(M.shape[0]))
            Linv = npl.inv(L); return Linv.T @ Linv
        except npl.LinAlgError:
            eps *= 10.0
    return npl.pinv(_sym(M) + eps*np.eye(M.shape[0]))


def greedy_time_subset_multi(J3d: np.ndarray, Sigma_y: np.ndarray, k: int,
                             allowed_mask: np.ndarray, day_index: np.ndarray,
                             daily_cap: int = 2, jitter: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy D-óptimo con restricciones (L–V 08–17 y 2/día).Devuelve (idx_sel, gains)."""
    n_t, m, n_theta = J3d.shape
    Sigma_inv = npl.inv(Sigma_y)
    Ainv = (1.0/jitter)*np.eye(n_theta)
    logdet_Sinv = chol_logdet(Sigma_inv)
    remaining = np.where(allowed_mask)[0]
    selected=[]; gains=[]; per_day={}
    for _ in range(min(k, remaining.size)):
        best_i=None; best_gain=-np.inf
        for i in remaining:
            d=int(day_index[i])
            if per_day.get(d,0) >= daily_cap: continue
            S = J3d[i,:,:]
            M = Sigma_inv + S @ Ainv @ S.T
            gain = chol_logdet(M) - logdet_Sinv
            if gain > best_gain:
                best_gain=gain; best_i=int(i)
        if best_i is None: break
        S = J3d[best_i,:,:]
        Minv = robust_inv_spd(Sigma_y + S @ Ainv @ S.T)
        Ainv = Ainv - Ainv @ S.T @ Minv @ S @ Ainv
        selected.append(best_i); gains.append(best_gain)
        dsel=int(day_index[best_i]); per_day[dsel]=per_day.get(dsel,0)+1
        # filtrar restantes (mismo día si ya está lleno y quitar best_i)
        keep=[]
        for j in remaining:
            if j==best_i: continue
            dd=int(day_index[j])
            if dd==dsel and per_day.get(dd,0)>=daily_cap: continue
            keep.append(j)
        remaining=np.array(keep, dtype=int)
    return np.array(sorted(selected),dtype=int), np.array(gains)

# =======================
# FIM y objetivo D-óptimo
# =======================

def fim_from_subset(J3d: np.ndarray, Sigma_y: np.ndarray, idx_sel: np.ndarray) -> np.ndarray:
    n_theta = J3d.shape[2]
    F = np.zeros((n_theta, n_theta))
    Sigma_inv = npl.inv(Sigma_y)
    for i in idx_sel:
        S = J3d[i,:,:]
        F += S.T @ Sigma_inv @ S
    return F

# ========================================
# Candidatos y selección FIMD (con horario)
# ========================================

def sample_designs(n: int, rng: np.random.Generator, sim: SimulationConfig,
                   feeds_in_workhours: bool = True) -> List[Design]:
    designs: List[Design] = []
    for _ in range(n):
        T1, T2, T3 = rng.uniform(16.0, 28.0, size=3)
        t12 = rng.uniform(6.0, 48.0)
        t23 = rng.uniform(max(t12 + 6.0, 24.0), 120.0)
        # tiempos de pulsos; si feeds_in_workhours=True, forzar horario hábil
        def draw_time(low, high):
            for _ in range(200):
                t = rng.uniform(low, high)
                if (not feeds_in_workhours) or is_workhour(t, sim):
                    return t
            return rng.uniform(low, high)  # fallback
        tN1 = draw_time(10.0, 120.0)
        tN2 = draw_time(max(tN1 + 6.0, 20.0), 160.0)
        dose1 = rng.uniform(0.0, 200.0)
        dose2 = rng.uniform(0.0, 200.0)
        tauN = rng.uniform(0.1, 0.5)
        designs.append(Design(T1,T2,T3,t12,t23,dose1,tN1,dose2,tN2,tauN))
    return designs

@dataclass
class FIMDConfig:
    n_candidates: int = 60
    n_select: int = 21
    K_per_exp: int = 12              # presupuesto de muestras por experimento
    daily_cap: int = 2               # máx 2 por día
    seed: int = 42
    feeds_in_workhours: bool = True  # forzar pulsos dentro de horario hábil
    verbose: bool = True


def fimd_select(phi_list: List[Design], theta_names: List[str], p_nom: BeaudeauModel1Params,
                sim: SimulationConfig, meas: MeasurementModel, cfg: FIMDConfig) -> Tuple[List[int], np.ndarray, List[np.ndarray]]:
    rng = np.random.default_rng(cfg.seed)
    chosen_idx: List[int] = []
    chosen_times: List[np.ndarray] = []
    n_theta = len(theta_names)
    F_accum = np.zeros((n_theta, n_theta))

    for k in range(cfg.n_select):
        best_obj = -np.inf
        best_idx = None
        best_times = None
        if cfg.verbose:
            print("="*80)
            print(f"[FIMD] Iter {k+1}/{cfg.n_select} (candidatos={len(phi_list)})")
        for j, phi in enumerate(phi_list):
            if j in chosen_idx: continue
            try:
                J3d, t = sensitivities_fd_multi(theta_names, p_nom, phi, sim)
                allowed, day_idx = compute_calendar_masks(t, sim)
                idx_sel, _ = greedy_time_subset_multi(J3d, meas.Sigma, cfg.K_per_exp,
                                                      allowed, day_idx, daily_cap=cfg.daily_cap)
                if idx_sel.size == 0:
                    continue
                F_j = fim_from_subset(J3d, meas.Sigma, idx_sel)
            except Exception as e:
                if cfg.verbose:
                    print(f"[WARN] candidato {j} falló simulación/selección: {e}")
                continue
            F_total = F_accum + F_j
            obj = chol_logdet(F_total + 1e-12*np.eye(n_theta))
            if obj > best_obj:
                best_obj = obj; best_idx = j; best_times = t[idx_sel]
        if best_idx is None:
            raise RuntimeError("No se pudo seleccionar ningún experimento bajo las restricciones; revise el espacio de diseño.")
        # aceptar y acumular
        chosen_idx.append(best_idx)
        chosen_times.append(best_times)
        # recomputar y sumar la FIM del elegido (para mantener consistencia exacta)
        J3d_b, t_b = sensitivities_fd_multi(theta_names, p_nom, phi_list[best_idx], sim)
        allowed_b, day_idx_b = compute_calendar_masks(t_b, sim)
        idx_sel_b, _ = greedy_time_subset_multi(J3d_b, meas.Sigma, cfg.K_per_exp,
                                                allowed_b, day_idx_b, daily_cap=cfg.daily_cap)
        F_accum += fim_from_subset(J3d_b, meas.Sigma, idx_sel_b)
        if cfg.verbose:
            print(f"[FIMD] -> elegido idx={best_idx} | logdet={best_obj:.3f} | #times={idx_sel_b.size}")
    return chosen_idx, F_accum, chosen_times

# =====================
# Ejecución de ejemplo
# =====================
if __name__ == "__main__":
    # Configuración base
    p = BeaudeauModel1Params()
    sim = SimulationConfig(t_end=200.0, dt=1.0, method="Radau",
                           stop_on_s_exhausted=False, start_weekday=0,
                           start_hour=8, end_hour_exclusive=17)
    meas = MeasurementModel(sigma_X=0.05, sigma_S=1.0, sigma_N=0.01, sigma_E=0.2)

    # Parámetros a estimar (12 para costo razonable)
    theta_names = [
        "k2", "KS", "KSI", "alpha_S",
        "k3", "KN",
        "Q0", "Emax",
        "Ynst", "Q0nst",
        "alpha_k1", "beta_k1",
    ]

    # Candidatos y selección
    cfg = FIMDConfig(n_candidates=60, n_select=21, K_per_exp=12, daily_cap=2, seed=42,
                     feeds_in_workhours=True, verbose=True)
    print(f"[MAIN] Generando {cfg.n_candidates} candidatos (feeds_in_workhours={cfg.feeds_in_workhours})...")
    rng = np.random.default_rng(cfg.seed)
    candidates = sample_designs(cfg.n_candidates, rng, sim, feeds_in_workhours=cfg.feeds_in_workhours)

    print(f"[MAIN] Seleccionando {cfg.n_select} diseños D-óptimos con restricciones de muestreo...")
    chosen_idx, F_final, chosen_times = fimd_select(candidates, theta_names, p, sim, meas, cfg)

    # Resumen
    print("\nResumen de diseños seleccionados (índice, perfil, pulsos de N, #times y ejemplo de horas):")
    for rank, idx in enumerate(chosen_idx, start=1):
        d = candidates[idx]
        times = chosen_times[rank-1]
        preview = np.array2string(np.round(times[:min(6, times.size)],1), separator=", ")
        print(
            f"#{rank:02d} idx={idx} | T1={d.T1:.1f}°C T2={d.T2:.1f}°C T3={d.T3:.1f}°C | "
            f"t12={d.t12:.1f}h t23={d.t23:.1f}h | N1={d.doseN1_mgL:.0f}mg/L@{d.tN1:.1f}h "
            f"N2={d.doseN2_mgL:.0f}mg/L@{d.tN2:.1f}h τ={d.tauN:.2f}h | times≈{preview}"
        )

    # Métrica final
    logdet_final = chol_logdet(F_final + 1e-12*np.eye(len(theta_names)))
    print(f"\n[RESULT] log det(FIM acumulada) = {logdet_final:.3f}")
