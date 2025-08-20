"""
FIMD Global v7 — Operaciones en horario hábil con colación + pre/post pulso forzados
+ objetivo de nitrógeno total N0 + (N1+N2) = 450 ppm
-----------------------------------------------------------------------------
Cambios vs v6:
  • t=0 ahora es **lunes 12:00** (inoculación a mediodía).
  • No se puede **samplear entre 13:00–14:00** ni **después de 17:00** (ventana hábil 12:00–17:00 excl., con pausa).
  • Forzado de muestreo **antes** y **después** de cada pulso de N (además de opciones previas).
  • Restricción de N: en todos los diseños se ajusta **N1+N2 = 450 − N0** (mg/L).

Mantiene:
  • Operaciones (t12, t23, tN1, tN2) en **horario hábil L–V**.
  • Muestreo ampliado (K_per_exp=18) con **máx. 2 muestras/día** y calendario laboral.
  • Forzados en set-points (activables por flag).
  • Modo de reasignación global de tiempos (más óptimo).

Requisitos: numpy, scipy, pandas
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import numpy.linalg as npl
import pandas as pd
from scipy.integrate import solve_ivp

# -------------------- índices de estado --------------------
STATE_IDX = {"X":0, "S":1, "N":2, "E":3, "CO2":4, "Nintra":5, "NST":6, "A":7}

# -------------------- Parámetros y configs --------------------
@dataclass
class BeaudeauModel1Params:
    alpha_k1: float = 0.0287
    beta_k1:  float = 0.3
    k2: float = 0.0386
    KS: float = 20.67
    KSI: float = 0.006299
    alpha_S: float = 0.9355
    YE_over_S: float = 2.17
    YCO2_over_S: float = 2.17
    k3: float = 0.001033
    KN: float = 0.04105
    KNI: float = 0.02635
    alpha_N: float = 1.195
    Q0: float = 0.0001347
    Emax: float = 94.67
    knst: float = 1.0024
    kdnst: float = 1.987
    KNST: float = 10.76
    Ynst: float = 694.8
    Q0nst: float = 0.0001334
    alpha1: float = 0.0004795
    kappa: float = 0.03
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
    S0: float = 240.0
    N0_mgL: float = 235.0   # N inicial (ppm ≈ mg/L)
    E0: float = 0.0
    CO20: float = 0.0
    Nintra0: float = 5e-4
    NST0: float = 1e-3
    A0: float = 1.0
    method: str = "Radau"
    stop_on_s_exhausted: bool = False
    # Calendario operativo/muestreo
    start_weekday: int = 0   # 0=Lunes
    start_hour: int = 12     # ← inoculación a las 12:00
    end_hour_exclusive: int = 17  # válido 12:00..16:59
    lunch_start_hour: int = 13    # sin muestreo 13:00–14:00
    lunch_end_hour_exclusive: int = 14

@dataclass
class MeasurementModel:
    sigma_X: float = 0.05
    sigma_S: float = 1.0
    sigma_N: float = 0.01
    sigma_E: float = 0.2
    @property
    def Sigma(self) -> np.ndarray:
        return np.diag([self.sigma_X**2, self.sigma_S**2, self.sigma_N**2, self.sigma_E**2])

@dataclass
class FIMDGlobalConfig:
    n_candidates: int = 60
    n_select: int = 21
    K_per_exp: int = 18
    daily_cap: int = 2
    seed: int = 42
    K_total: int | None = None
    # Operaciones en horario hábil
    feeds_in_workhours: bool = True
    setpoints_in_workhours: bool = True
    repair_ops_to_workhours: bool = True
    # Forzados en pulsos y set-points
    force_pre_post_pulse: bool = True   # NUEVO: siempre pre y post
    force_at_pulse: bool = False        # opcional (off por cap 2/día)
    force_postpulse: bool = False       # ya cubierto con pre/post
    prepost_window_h: float = 1.0       # distancia mínima al pulso

    force_at_setpoint: bool = True
    force_post_setpoint: bool = False
    post_setpoint_window_h: float = 2.0

    reassign_all_each_iter: bool = True
    target_total_N_mgL: float = 450.0   # N0 + Ndos = 450 ppm
    verbose: bool = True

# -------------------- Utilidades calendario --------------------

def _is_weekday(day_index: int, start_weekday: int) -> bool:
    wd = (day_index + start_weekday) % 7
    return wd <= 4


def compute_calendar_masks(t_eval: np.ndarray, sim: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
    abs_hour = sim.start_hour + t_eval
    day_index = np.floor(abs_hour/24.0).astype(int)
    hod = np.mod(abs_hour, 24.0)
    in_base = (hod >= sim.start_hour) & (hod < sim.end_hour_exclusive)
    not_lunch = ~((hod >= sim.lunch_start_hour) & (hod < sim.lunch_end_hour_exclusive))
    weekday = np.array([_is_weekday(int(d), sim.start_weekday) for d in day_index])
    return (in_base & not_lunch & weekday), day_index


def is_workhour(t: float, sim: SimulationConfig) -> bool:
    abs_hour = sim.start_hour + t
    d = int(np.floor(abs_hour/24.0))
    hod = abs_hour % 24.0
    if not _is_weekday(d, sim.start_weekday):
        return False
    if not (sim.start_hour <= hod < sim.end_hour_exclusive):
        return False
    if sim.lunch_start_hour <= hod < sim.lunch_end_hour_exclusive:
        return False
    return True


def snap_to_next_workhour(t: float, sim: SimulationConfig) -> float:
    tt = float(np.ceil(t))
    for _ in range(14*24):  # dos semanas de margen
        if is_workhour(tt, sim):
            return tt
        tt += 1.0
    return t


def map_time_to_calendar(t: float, sim: SimulationConfig) -> str:
    weekdays = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    abs_hour = sim.start_hour + t
    day_index = int(np.floor(abs_hour/24.0))
    wd = weekdays[(day_index + sim.start_weekday) % 7]
    hod = abs_hour % 24.0
    hh = int(hod); mm = int(round((hod - hh)*60))
    return f"{wd} {hh:02d}:{mm:02d}"

# -------------------- Modelo --------------------

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


def rhs(t: float, x: np.ndarray, p: BeaudeauModel1Params, phi: Design) -> np.ndarray:
    X,S,N,E,CO2,Nintra,NST,A = x
    X = max(X,1e-9); S=max(S,0.0); N=max(N,0.0)
    Nintra = max(Nintra,1e-9); NST=max(NST,0.0); E=max(E,0.0)
    A = np.clip(A,0.0,1.0)
    T = T_of_t(t,phi)
    k1 = max(p.alpha_k1*T - p.beta_k1, 0.0)
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
    dA_dt = (rX / max(X,1e-9)) * (1.0 - A) - p.kappa * A
    rN_ext = N_feed_rate(t,phi)
    dX=rX; dS=rS; dN=rN+rN_ext; dE=rE; dCO2=rCO2
    dNintra = rNintra - Nintra * (rX / max(X,1e-9))
    dNST = rNST - NST * (rX / max(X,1e-9))
    return np.array([dX,dS,dN,dE,dCO2,dNintra,dNST,dA_dt], float)


def simulate(phi: Design, p: BeaudeauModel1Params, sim: SimulationConfig,
             t_eval: np.ndarray | None = None, use_event: bool | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if t_eval is None: t_eval = np.arange(0.0, sim.t_end+1e-9, sim.dt)
    if use_event is None: use_event = sim.stop_on_s_exhausted
    N0_gL = sim.N0_mgL/1000.0
    x0 = np.array([sim.X0, sim.S0, N0_gL, sim.E0, sim.CO20, sim.Nintra0, sim.NST0, sim.A0], float)
    events=None
    if use_event:
        def stop_when_sugar_exhausted(t,x): return x[STATE_IDX["S"]]-1.0
        stop_when_sugar_exhausted.terminal=True; stop_when_sugar_exhausted.direction=-1
        events=stop_when_sugar_exhausted
    sol = solve_ivp(lambda t,x: rhs(t,x,p,phi), (t_eval[0], t_eval[-1]), x0, method=sim.method,
                    t_eval=t_eval, atol=p.atol, rtol=p.rtol, events=events, max_step=0.5)
    if not sol.success: raise RuntimeError(f"ODE failed: {sol.message}")
    return sol.t, sol.y.T

# -------------------- Mediciones y sensibilidades --------------------

def measure_XSNE(ytraj: np.ndarray) -> np.ndarray:
    return ytraj[:, [STATE_IDX["X"], STATE_IDX["S"], STATE_IDX["N"], STATE_IDX["E"]]]


def sensitivities_fd_multi(theta_names: List[str], p_nom: BeaudeauModel1Params,
                           phi: Design, sim: SimulationConfig, rel_eps: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    t_eval = np.arange(0.0, sim.t_end+1e-9, sim.dt)
    _, Y_nom = simulate(phi, p_nom, sim, t_eval, use_event=False)
    n_t = Y_nom.shape[0]
    m = 4; n_theta = len(theta_names)
    J = np.zeros((n_t, m, n_theta))
    for k, name in enumerate(theta_names):
        val = getattr(p_nom, name); h = rel_eps*max(abs(val),1e-6)
        p_plus  = BeaudeauModel1Params(**{**p_nom.__dict__, name: val + h})
        p_minus = BeaudeauModel1Params(**{**p_nom.__dict__, name: max(val - h, 1e-12)})
        _, Yp = simulate(phi, p_plus,  sim, t_eval, use_event=False)
        _, Ym = simulate(phi, p_minus, sim, t_eval, use_event=False)
        J[:,:,k] = (measure_XSNE(Yp) - measure_XSNE(Ym)) / (2.0*h)
    return np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0), t_eval

# -------------------- Linalg robusto --------------------

def _sym(M: np.ndarray) -> np.ndarray: return 0.5*(M+M.T)

def chol_logdet(M: np.ndarray, jitter: float = 1e-12, max_tries: int = 6) -> float:
    eps=jitter
    for _ in range(max_tries):
        try:
            L = npl.cholesky(_sym(M)+eps*np.eye(M.shape[0]))
            return 2.0*float(np.sum(np.log(np.diag(L))))
        except npl.LinAlgError:
            eps*=10.0
    sign,ld = npl.slogdet(_sym(M)+eps*np.eye(M.shape[0]))
    return float(ld)


def robust_inv_spd(M: np.ndarray, jitter: float = 1e-12, max_tries: int = 6) -> np.ndarray:
    eps=jitter
    for _ in range(max_tries):
        try:
            L = npl.cholesky(_sym(M)+eps*np.eye(M.shape[0]))
            Linv = npl.inv(L)
            return Linv.T @ Linv
        except npl.LinAlgError:
            eps*=10.0
    return npl.pinv(_sym(M)+eps*np.eye(M.shape[0]))

# -------------------- FIM utilitaria --------------------

def fim_from_selected(J_list: List[np.ndarray], Sigma_y: np.ndarray, idx_lists: List[np.ndarray]) -> np.ndarray:
    n_theta = J_list[0].shape[2]
    F = np.zeros((n_theta,n_theta))
    Sinv = npl.inv(Sigma_y)
    for J3d, idx in zip(J_list, idx_lists):
        for i in idx:
            S = J3d[i,:,:]
            F += S.T @ Sinv @ S
    return F

# -------------------- Forzados (pulsos y set-points) --------------------

def _nearest_allowed_index(t_eval: np.ndarray, t: float, allowed_mask: np.ndarray) -> int | None:
    if t < t_eval[0] or t > t_eval[-1]:
        return None
    cand = int(round(t))
    neigh = [cand, cand-1, cand+1]
    for tt in neigh:
        if 0 <= tt < len(t_eval) and allowed_mask[tt]:
            return tt
    idx_allowed = np.where(allowed_mask)[0]
    if idx_allowed.size==0: return None
    k = int(idx_allowed[np.argmin(np.abs(t_eval[idx_allowed] - t))])
    return k


def _nearest_allowed_before(t_eval: np.ndarray, t: float, allowed_mask: np.ndarray) -> int | None:
    k = int(np.floor(t))
    for i in range(k, -1, -1):
        if allowed_mask[i]:
            return i
    return None


def _nearest_allowed_after(t_eval: np.ndarray, t: float, allowed_mask: np.ndarray) -> int | None:
    k = int(np.ceil(t))
    for i in range(k, len(t_eval)):
        if allowed_mask[i]:
            return i
    return None


def forced_indices_for_experiment(t_eval: np.ndarray, phi: Design, sim: SimulationConfig,
                                  allowed_mask: np.ndarray, day_idx: np.ndarray,
                                  cfg: FIMDGlobalConfig) -> Tuple[np.ndarray, Dict[int, Dict[str,bool]]]:
    forced: List[int] = []
    flags: Dict[int, Dict[str,bool]] = {}

    # PRE/POST pulso
    if cfg.force_pre_post_pulse:
        for tN, tag in [(phi.tN1, "pulse1"), (phi.tN2, "pulse2")]:
            i_pre = _nearest_allowed_before(t_eval, tN - 1e-6, allowed_mask)
            i_post = _nearest_allowed_after(t_eval, tN + 1e-6, allowed_mask)
            if i_pre is not None:
                forced.append(i_pre)
                d = int(day_idx[i_pre]); flags.setdefault(d, {})[f"pre_{tag}"] = True
            if i_post is not None:
                forced.append(i_post)
                d = int(day_idx[i_post]); flags.setdefault(d, {})[f"post_{tag}"] = True
    # (opcional) muestra exactamente al pulso
    if cfg.force_at_pulse:
        for tN, tag in [(phi.tN1, "pulse1"), (phi.tN2, "pulse2")]:
            if is_workhour(tN, sim):
                i_at = _nearest_allowed_index(t_eval, tN, allowed_mask)
                if i_at is not None:
                    forced.append(i_at)
                    d = int(day_idx[i_at]); flags.setdefault(d, {})[f"at_{tag}"] = True
    # Set-points
    if cfg.force_at_setpoint:
        for tC, tag in [(phi.t12, "t12"), (phi.t23, "t23")]:
            if is_workhour(tC, sim):
                i_at = _nearest_allowed_index(t_eval, tC, allowed_mask)
                if i_at is not None:
                    forced.append(i_at)
                    d = int(day_idx[i_at]); flags.setdefault(d, {})[f"at_{tag}"] = True
    if cfg.force_post_setpoint:
        for tC, tag in [(phi.t12, "t12"), (phi.t23, "t23")]:
            i_ps = _nearest_allowed_after(t_eval, tC + 1e-6, allowed_mask)
            if i_ps is not None:
                forced.append(i_ps)
                d = int(day_idx[i_ps]); flags.setdefault(d, {})[f"post_{tag}"] = True

    return np.array(sorted(set(forced)), dtype=int), flags


def enforce_daily_cap(pre_idx: np.ndarray, day_idx: np.ndarray, cap: int,
                      priority_mask: Dict[int, Dict[str,bool]]) -> np.ndarray:
    # prioridad: pre_/post_ (pulsos) > at_setpoint/post_setpoint > at_pulse > otros
    keep: List[int] = []
    per_day: Dict[int, List[int]] = {}
    for i in sorted(pre_idx.tolist()):
        d = int(day_idx[i])
        per_day.setdefault(d, []).append(i)
    for d, idxs in per_day.items():
        if len(idxs) <= cap:
            keep.extend(idxs); continue
        cat1=[]; cat2=[]; cat3=[]; cat4=[]; cat5=[]
        tags = priority_mask.get(d, {})
        for i in idxs:
            keys = set(k for k,v in tags.items() if v)
            if any(k.startswith("pre_") or k.startswith("post_") for k in keys):
                cat1.append(i)
            elif any(k in ("at_t12","at_t23") for k in keys):
                cat2.append(i)
            elif any(k.startswith("post_t") for k in keys):
                cat3.append(i)
            elif any(k.startswith("at_pulse") for k in keys):
                cat4.append(i)
            else:
                cat5.append(i)
        chosen=[]
        for L in [cat1,cat2,cat3,cat4,cat5]:
            for i in L:
                if len(chosen) < cap:
                    chosen.append(i)
        keep.extend(chosen)
    return np.array(sorted(set(keep)), dtype=int)

# -------------------- Asignación global (greedy rank-m) --------------------

def global_greedy_allocate_forced(J_list: List[np.ndarray], Sigma_y: np.ndarray,
                                  allowed_list: List[np.ndarray], day_list: List[np.ndarray],
                                  forced_lists: List[np.ndarray], forced_flags: List[Dict[int,Dict[str,bool]]],
                                  pre_idx_lists: List[np.ndarray], K_target: int, daily_cap: int = 2,
                                  fix_previous: bool = False) -> List[np.ndarray]:
    n_exp = len(J_list)
    n_theta = J_list[0].shape[2]
    Sinv = npl.inv(Sigma_y)

    idx_lists = []
    total_selected = 0
    for e in range(n_exp):
        pre = pre_idx_lists[e] if fix_previous else np.array([], dtype=int)
        pre = np.array(sorted(set(np.concatenate([pre, forced_lists[e]]) if forced_lists[e].size else pre)), dtype=int)
        pre = enforce_daily_cap(pre, day_list[e], daily_cap, forced_flags[e])
        idx_lists.append(pre)
        total_selected += pre.size

    F_pre = fim_from_selected(J_list, Sigma_y, idx_lists)
    Ainv = robust_inv_spd(F_pre + 1e-8*np.eye(n_theta))

    remaining: List[Tuple[int,int]] = []
    for e in range(n_exp):
        allowed = allowed_list[e]
        pre_set = set(idx_lists[e].tolist())
        for i in np.where(allowed)[0]:
            if i in pre_set: continue
            remaining.append((e, int(i)))

    logdet_Sinv = chol_logdet(Sinv)
    while total_selected < K_target and remaining:
        best=None; best_gain=-np.inf
        for (e,i) in remaining:
            d = int(day_list[e][i])
            used_today = np.sum(day_list[e][idx_lists[e]]==d)
            if used_today >= daily_cap: continue
            S = J_list[e][i,:,:]
            M = Sinv + S @ Ainv @ S.T
            gain = chol_logdet(M) - logdet_Sinv
            if gain > best_gain:
                best_gain = gain; best = (e,i)
        if best is None: break
        e,i = best
        S = J_list[e][i,:,:]
        Minv = robust_inv_spd(Sigma_y + S @ Ainv @ S.T)
        Ainv = Ainv - Ainv @ S.T @ Minv @ S @ Ainv
        idx_lists[e] = np.append(idx_lists[e], i)
        total_selected += 1
        # filtrar por cap diario
        new_remaining=[]
        dsel = int(day_list[e][i])
        for (ee,ii) in remaining:
            if ee==e and ii==i: continue
            if ee==e:
                dd = int(day_list[ee][ii])
                if dd==dsel and np.sum(day_list[e][idx_lists[e]]==dsel) >= daily_cap:
                    continue
            new_remaining.append((ee,ii))
        remaining = new_remaining

    out = []
    for e in range(n_exp):
        arr = np.array(sorted(set(idx_lists[e].tolist())), dtype=int)
        out.append(arr)
    return out

# -------------------- Generación / reparación candidatos --------------------

def enforce_total_nitrogen(phi: Design, sim: SimulationConfig, target_total_mgL: float) -> Design:
    target_dos = max(target_total_mgL - sim.N0_mgL, 0.0)
    if target_dos == 0.0:
        return Design(phi.T1,phi.T2,phi.T3,phi.t12,phi.t23,0.0,phi.tN1,0.0,phi.tN2,phi.tauN)
    # si ya hay dosis, mantener proporción; si no, dividir 50/50
    s = phi.doseN1_mgL + phi.doseN2_mgL
    if s <= 1e-9:
        r = 0.5
    else:
        r = phi.doseN1_mgL / s
        r = float(np.clip(r, 0.1, 0.9))
    d1 = r * target_dos
    d2 = (1.0 - r) * target_dos
    return Design(phi.T1,phi.T2,phi.T3,phi.t12,phi.t23,d1,phi.tN1,d2,phi.tN2,phi.tauN)


def draw_time_workhour(rng: np.random.Generator, lo: float, hi: float, sim: SimulationConfig) -> float:
    for _ in range(800):
        t = rng.uniform(lo, hi)
        if is_workhour(t, sim):
            return t
    t = rng.uniform(lo, hi)
    return snap_to_next_workhour(t, sim)


def repair_design_operational(phi: Design, sim: SimulationConfig, cfg: FIMDGlobalConfig) -> Design:
    T1,T2,T3 = phi.T1, phi.T2, phi.T3
    t12, t23 = phi.t12, phi.t23
    tN1, tN2 = phi.tN1, phi.tN2
    if cfg.setpoints_in_workhours:
        if not is_workhour(t12, sim): t12 = snap_to_next_workhour(t12, sim)
        min_t23 = max(t12 + 6.0, 24.0)
        if t23 < min_t23: t23 = min_t23
        if not is_workhour(t23, sim): t23 = snap_to_next_workhour(t23, sim)
    if cfg.feeds_in_workhours:
        if not is_workhour(tN1, sim): tN1 = snap_to_next_workhour(tN1, sim)
        min_tN2 = max(tN1 + 6.0, 20.0)
        if tN2 < min_tN2: tN2 = min_tN2
        if not is_workhour(tN2, sim): tN2 = snap_to_next_workhour(tN2, sim)
    # ajustar dosis a objetivo N total
    phi2 = Design(T1,T2,T3,t12,t23,phi.doseN1_mgL,tN1,phi.doseN2_mgL,tN2,phi.tauN)
    phi2 = enforce_total_nitrogen(phi2, sim, cfg.target_total_N_mgL)
    return phi2


def sample_designs(n: int, rng: np.random.Generator, sim: SimulationConfig,
                   cfg: FIMDGlobalConfig) -> List[Design]:
    designs: List[Design] = []
    for _ in range(n):
        T1,T2,T3 = rng.uniform(16.0, 28.0, size=3)
        t12 = draw_time_workhour(rng, 6.0, 48.0, sim)
        t23 = draw_time_workhour(rng, max(t12+6.0, 24.0), 120.0, sim)
        tN1 = draw_time_workhour(rng, 10.0, 120.0, sim)
        tN2 = draw_time_workhour(rng, max(tN1+6.0, 20.0), 160.0, sim)
        # dosis iniciales (proporción aleatoria) → luego se ajustan al objetivo exacto
        r = rng.uniform(0.3, 0.7)
        target_dos = max(cfg.target_total_N_mgL - sim.N0_mgL, 0.0)
        d1 = r*target_dos; d2 = (1.0-r)*target_dos
        tauN = rng.uniform(0.1, 0.5)
        d = Design(T1,T2,T3,t12,t23,d1,tN1,d2,tN2,tauN)
        d = repair_design_operational(d, sim, cfg)
        designs.append(d)
    return designs

# -------------------- FIMD Global (selección con forzados) --------------------

def fimd_global_forced(phi_list: List[Design], theta_names: List[str], p_nom: BeaudeauModel1Params,
                       sim: SimulationConfig, meas: MeasurementModel, cfg: FIMDGlobalConfig):
    rng = np.random.default_rng(cfg.seed)
    K_total = cfg.K_total if cfg.K_total is not None else cfg.n_select * cfg.K_per_exp

    chosen_idx: List[int] = []
    J_bank: Dict[int, np.ndarray] = {}
    t_bank: Dict[int, np.ndarray] = {}
    allowed_bank: Dict[int, np.ndarray] = {}
    day_bank: Dict[int, np.ndarray] = {}
    forced_bank: Dict[int, np.ndarray] = {}
    forced_flags_bank: Dict[int, Dict[int, Dict[str,bool]]] = {}
    idxsel_bank: Dict[int, np.ndarray] = {}

    n_theta = len(theta_names)
    F_accum = np.zeros((n_theta,n_theta))

    for k in range(cfg.n_select):
        best_obj = -np.inf; best_idx = None
        best_alloc_tmp: Dict[int, np.ndarray] | None = None

        if cfg.verbose:
            mode = "REASSIGN-ALL" if cfg.reassign_all_each_iter else "INCREMENTAL"
            print("="*96)
            print(f"[GLOBAL FIMD v7] Iter {k+1}/{cfg.n_select} (candidatos={len(phi_list)}) | modo={mode}")

        current_total = int(np.sum([len(v) for v in idxsel_bank.values()]))
        K_target = min(K_total, current_total + cfg.K_per_exp)

        for j, phi in enumerate(phi_list):
            if j in chosen_idx: continue
            phi = repair_design_operational(phi, sim, cfg)
            # preparar listas para exps ya elegidos + candidato j
            exps = chosen_idx + [j]
            J_list: List[np.ndarray] = []
            allowed_list: List[np.ndarray] = []
            day_list: List[np.ndarray] = []
            pre_idx_lists: List[np.ndarray] = []
            forced_lists: List[np.ndarray] = []
            forced_flags: List[Dict[int,Dict[str,bool]]] = []

            try:
                for e in exps:
                    if e not in J_bank:
                        J_bank[e], t_bank[e] = sensitivities_fd_multi(theta_names, p_nom, phi_list[e], sim)
                        allowed_bank[e], day_bank[e] = compute_calendar_masks(t_bank[e], sim)
                        f_idx, f_flags = forced_indices_for_experiment(t_bank[e], phi_list[e], sim,
                                                                       allowed_bank[e], day_bank[e], cfg)
                        forced_bank[e] = f_idx; forced_flags_bank[e] = f_flags
                    J_list.append(J_bank[e]); allowed_list.append(allowed_bank[e]); day_list.append(day_bank[e])
                    if cfg.reassign_all_each_iter:
                        pre_idx_lists.append(np.array([], dtype=int))
                    else:
                        pre_idx_lists.append(idxsel_bank.get(e, np.array([],dtype=int)))
                    forced_lists.append(forced_bank[e])
                    forced_flags.append(forced_flags_bank[e])
                alloc_lists = global_greedy_allocate_forced(J_list, meas.Sigma, allowed_list, day_list,
                                                            forced_lists, forced_flags, pre_idx_lists,
                                                            K_target, daily_cap=cfg.daily_cap,
                                                            fix_previous=not cfg.reassign_all_each_iter)
                F_total = fim_from_selected(J_list, meas.Sigma, alloc_lists)
            except Exception as e:
                if cfg.verbose:
                    print(f"[WARN] candidato {j} falló evaluación: {e}")
                continue

            obj = chol_logdet(F_total + 1e-12*np.eye(n_theta))
            if obj > best_obj:
                best_obj = obj; best_idx = j
                best_alloc_tmp = {exp: sel for exp, sel in zip(exps, alloc_lists)}

        if best_idx is None:
            raise RuntimeError("No se pudo seleccionar ningún experimento (restricciones muy estrictas).")

        chosen_idx.append(best_idx)
        if best_alloc_tmp is not None:
            for exp_id in chosen_idx:
                if exp_id in best_alloc_tmp:
                    idxsel_bank[exp_id] = np.array(sorted(set(best_alloc_tmp[exp_id].tolist())), dtype=int)
        F_accum = fim_from_selected([J_bank[e] for e in chosen_idx], meas.Sigma,
                                    [idxsel_bank[e] for e in chosen_idx])
        if cfg.verbose:
            counts = {e: len(idxsel_bank[e]) for e in chosen_idx}
            print(f"[GLOBAL FIMD v7] -> elegido idx={best_idx} | logdet={best_obj:.3f} | asignación={counts}")

    chosen_times = [ t_bank[e][idxsel_bank[e]] if len(idxsel_bank[e]) else np.array([]) for e in chosen_idx ]
    return chosen_idx, F_accum, chosen_times, t_bank, idxsel_bank, forced_bank

# -------------------- Main con export a Excel --------------------
if __name__ == "__main__":
    p = BeaudeauModel1Params()
    sim = SimulationConfig(t_end=200.0, dt=1.0, method="Radau", stop_on_s_exhausted=False,
                           start_weekday=0, start_hour=12, end_hour_exclusive=17,
                           lunch_start_hour=13, lunch_end_hour_exclusive=14)
    meas = MeasurementModel(sigma_X=0.05, sigma_S=1.0, sigma_N=0.01, sigma_E=0.2)

    theta_names = [
        "k2", "KS", "KSI", "alpha_S",
        "k3", "KN",
        "Q0", "Emax",
        "Ynst", "Q0nst",
        "alpha_k1", "beta_k1",
    ]

    cfg = FIMDGlobalConfig(
        n_candidates=60, n_select=21,
        K_per_exp=18, daily_cap=2, seed=42, K_total=None,
        feeds_in_workhours=True, setpoints_in_workhours=True, repair_ops_to_workhours=True,
        force_pre_post_pulse=True, force_at_pulse=False, force_postpulse=False, prepost_window_h=1.0,
        force_at_setpoint=True, force_post_setpoint=False, post_setpoint_window_h=2.0,
        reassign_all_each_iter=True, target_total_N_mgL=450.0, verbose=True
    )

    print(f"[MAIN] Generando {cfg.n_candidates} candidatos (operaciones en horario hábil + N total a 450 ppm)…")
    rng = np.random.default_rng(cfg.seed)
    candidates = sample_designs(cfg.n_candidates, rng, sim, cfg)

    print(f"[MAIN] Seleccionando {cfg.n_select} diseños (GLOBAL+FORCED, v7)…")
    chosen_idx, F_final, chosen_times, t_bank, idxsel_bank, forced_bank = fimd_global_forced(
        candidates, theta_names, p, sim, meas, cfg
    )

    # -------- Excel plan --------
    rows = []
    for rank, idx in enumerate(chosen_idx, start=1):
        d = candidates[idx]
        sel = t_bank[idx][idxsel_bank[idx]] if len(idxsel_bank[idx]) else np.array([])
        forced = t_bank[idx][forced_bank[idx]] if len(forced_bank[idx]) else np.array([])
        cal_sel = ", ".join(map(lambda t: map_time_to_calendar(float(t), sim), sel[:12]))
        ops_cal = ", ".join([
            f"t12→{map_time_to_calendar(d.t12, sim)}",
            f"t23→{map_time_to_calendar(d.t23, sim)}",
            f"N1@{map_time_to_calendar(d.tN1, sim)}",
            f"N2@{map_time_to_calendar(d.tN2, sim)}",
        ])
        rows.append({
            "rank": rank, "idx": idx,
            "T1": d.T1, "T2": d.T2, "T3": d.T3, "t12_h": d.t12, "t23_h": d.t23,
            "N1_mgL": d.doseN1_mgL, "tN1_h": d.tN1, "N2_mgL": d.doseN2_mgL, "tN2_h": d.tN2,
            "tau_h": d.tauN,
            "N_total_ppm": sim.N0_mgL + d.doseN1_mgL + d.doseN2_mgL,
            "n_times": int(sel.size),
            "times_h": list(map(float, sel)),
            "forced_times_h": list(map(float, forced)),
            "times_calendar_preview": cal_sel,
            "ops_calendar": ops_cal,
        })

    df = pd.DataFrame(rows)
    out_path = "FIMD_GLOBAL_v7_opsLunch_prePostPulse_targetN450.xlsx"
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="plan")
        cols = ["rank","idx","T1","T2","T3","t12_h","t23_h","N1_mgL","tN1_h","N2_mgL","tN2_h","tau_h","N_total_ppm","n_times","times_calendar_preview","ops_calendar"]
        df[cols].to_excel(w, index=False, sheet_name="compact")

    sign, logdet = npl.slogdet(F_final + 1e-12*np.eye(len(theta_names)))
    print(f"\n[RESULT] log det(FIM acumulada) = {logdet:.3f}")
    print(f"[FILES] Exportado plan: {out_path}")
    print("Restricciones: t=0 Lunes 12:00; muestreo hábil sin 13–14h ni >17h; pre/post pulso forzados; N_total=450ppm.")
