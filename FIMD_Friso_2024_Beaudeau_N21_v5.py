"""
FIMD Global (multi-output) con restricciones operacionales en horario hábil
+ muestreo ampliado y forzados alrededor de pulsos
--------------------------------------------------------------------------
Basado en la versión v4 que integra muestreo en horario hábil y forzados en pulsos.
Cambios clave en esta v5:
  1) **NUEVA restricción operativa dura**: los **cambios de set-point** (t12, t23) y los
     **pulsos de N (tN1, tN2)** deben caer en **horario hábil L–V 08:00–17:00**.
     - Se garantiza desde la **generación de candidatos** y, opcionalmente,
       con un **reparador** que ajusta al **siguiente slot hábil** (flag configurable).
  2) **Muestreo ampliado**: por defecto **K_per_exp = 18** (antes 12) manteniendo
     el tope de **2 muestras/día/fermentación** en horario hábil.
  3) Se conservan las reglas de muestreo **forzado en pulsos** (muestra en el pulso
     y ≥1 muestra en [tN, tN+2 h] si es hábil), respetando el tope diario.
  4) Exporta Excel con resumen y vista calendario.

Notas:
- τ (tauN) es la **duración (h)** del pulso rectangular de N; la dosis mg/L se reparte
  a caudal constante durante τ (término fuente en dN/dt).
- El mapeo a calendario asume **t=0 → Lunes 08:00**.

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
    S0: float = 180.0
    N0_mgL: float = 140.0
    E0: float = 0.0
    CO20: float = 0.0
    Nintra0: float = 5e-4
    NST0: float = 1e-3
    A0: float = 1.0
    method: str = "Radau"
    stop_on_s_exhausted: bool = False
    # Calendario operativo
    start_weekday: int = 0   # 0=Lunes
    start_hour: int = 8      # 08:00 → t=0
    end_hour_exclusive: int = 17  # válido 08..16:59

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
    K_per_exp: int = 18          # ← ampliado
    daily_cap: int = 2
    seed: int = 42
    K_total: int | None = None
    # NUEVAS banderas
    feeds_in_workhours: bool = True        # pulsos tN1,tN2 en hábil
    setpoints_in_workhours: bool = True    # t12,t23 en hábil
    repair_ops_to_workhours: bool = True   # si algún tiempo no cae en hábil, ajustar al siguiente slot hábil
    # Forzados en pulsos (muestreo)
    force_postpulse: bool = True
    force_at_pulse: bool = True
    verbose: bool = True

# -------------------- Utilidades calendario --------------------

def compute_calendar_masks(t_eval: np.ndarray, sim: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
    abs_hour = sim.start_hour + t_eval
    day_index = np.floor(abs_hour/24.0).astype(int)
    hod = np.mod(abs_hour, 24.0)
    wd = np.mod(day_index + sim.start_weekday, 7)
    in_hours = (hod >= sim.start_hour) & (hod < sim.end_hour_exclusive)
    is_weekday = (wd <= 4)
    return (in_hours & is_weekday), day_index


def is_workhour(t: float, sim: SimulationConfig) -> bool:
    abs_hour = sim.start_hour + t
    d = int(np.floor(abs_hour/24.0))
    wd = (d + sim.start_weekday) % 7
    hod = abs_hour % 24.0
    return (wd <= 4) and (sim.start_hour <= hod < sim.end_hour_exclusive)


def snap_to_next_workhour(t: float, sim: SimulationConfig) -> float:
    """Empuja t al siguiente slot hábil (redondeo a hora entera)."""
    tt = float(np.ceil(t))
    # probar 7 días hacia adelante
    for _ in range(7*24):
        if is_workhour(tt, sim):
            return tt
        tt += 1.0
    return t  # fallback improbable


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
    Y_nom_meas = measure_XSNE(Y_nom)
    for k, name in enumerate(theta_names):
        val = getattr(p_nom, name); h = rel_eps*max(abs(val),1e-6)
        p_plus  = BeaudeauModel1Params(**{**p_nom.__dict__, name: val + h})
        p_minus = BeaudeauModel1Params(**{**p_nom.__dict__, name: max(val - h, 1e-12)})
        _, Yp = simulate(phi, p_plus,  sim, t_eval, use_event=False)
        _, Ym = simulate(phi, p_minus, sim, t_eval, use_event=False)
        J[:,:,k] = (measure_XSNE(Yp) - measure_XSNE(Ym)) / (2.0*h)
    # seguridad ante NaN/Inf
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

# -------------------- Forzados alrededor de pulsos --------------------

def forced_indices_for_experiment(t_eval: np.ndarray, phi: Design, sim: SimulationConfig,
                                  allowed_mask: np.ndarray, day_idx: np.ndarray,
                                  cfg: FIMDGlobalConfig) -> Tuple[np.ndarray, Dict[int, Dict[str,bool]]]:
    forced: List[int] = []
    flags: Dict[int, Dict[str,bool]] = {}

    def nearest_allowed_index(t_target: float) -> int | None:
        if t_target < t_eval[0] or t_target > t_eval[-1]:
            return None
        cand = int(round(t_target))
        neigh = [cand, cand-1, cand+1]
        for tt in neigh:
            if 0 <= tt < len(t_eval) and allowed_mask[tt]:
                return tt
        idx_allowed = np.where(allowed_mask)[0]
        if idx_allowed.size==0: return None
        k = int(idx_allowed[np.argmin(np.abs(t_eval[idx_allowed] - t_target))])
        return k

    def post_window_indices(tN: float, window: float = 2.0) -> np.ndarray:
        lo = int(np.ceil(tN)); hi = min(int(np.ceil(tN+window)), int(t_eval[-1]))
        idx = np.arange(lo, hi+1, dtype=int)
        idx = idx[(idx >= 0) & (idx < len(t_eval))]
        idx = idx[allowed_mask[idx]]
        return idx

    if cfg.force_at_pulse or cfg.force_postpulse:
        for tN in [phi.tN1, phi.tN2]:
            if cfg.force_at_pulse and is_workhour(tN, sim):
                i_at = nearest_allowed_index(tN)
                if i_at is not None:
                    forced.append(i_at)
                    d = int(day_idx[i_at])
                    flags.setdefault(d, {"at_pulse": False, "post_pulse": False})
                    flags[d]["at_pulse"] = True
            if cfg.force_postpulse:
                idx_win = post_window_indices(tN, window=2.0)
                idx_win = [i for i in idx_win if i not in forced]
                if idx_win:
                    i_pp = int(idx_win[0])
                    forced.append(i_pp)
                    d = int(day_idx[i_pp])
                    flags.setdefault(d, {"at_pulse": False, "post_pulse": False})
                    flags[d]["post_pulse"] = True
    return np.array(sorted(set(forced)), dtype=int), flags


def enforce_daily_cap(pre_idx: np.ndarray, day_idx: np.ndarray, cap: int,
                      priority_mask: Dict[int, Dict[str,bool]]) -> np.ndarray:
    keep: List[int] = []
    per_day: Dict[int, List[int]] = {}
    for i in sorted(pre_idx.tolist()):
        d = int(day_idx[i])
        per_day.setdefault(d, []).append(i)
    for d, idxs in per_day.items():
        if len(idxs) <= cap:
            keep.extend(idxs); continue
        at_list = []; post_list = []; others = []
        for i in idxs:
            tag = priority_mask.get(d, {"at_pulse": False, "post_pulse": False})
            if tag.get("at_pulse", False) and i in pre_idx:
                at_list.append(i)
            elif tag.get("post_pulse", False) and i in pre_idx:
                post_list.append(i)
            else:
                others.append(i)
        chosen = []
        for L in [at_list, post_list, others]:
            for i in L:
                if len(chosen) < cap:
                    chosen.append(i)
        keep.extend(chosen)
    return np.array(sorted(set(keep)), dtype=int)

# -------------------- Asignación global (greedy rank-m) --------------------

def global_greedy_allocate_forced(J_list: List[np.ndarray], Sigma_y: np.ndarray,
                                  allowed_list: List[np.ndarray], day_list: List[np.ndarray],
                                  forced_lists: List[np.ndarray], forced_flags: List[Dict[int,Dict[str,bool]]],
                                  pre_idx_lists: List[np.ndarray], K_target: int, daily_cap: int = 2) -> List[np.ndarray]:
    n_exp = len(J_list)
    n_theta = J_list[0].shape[2]
    Sinv = npl.inv(Sigma_y)

    idx_lists = []
    total_selected = 0
    for e in range(n_exp):
        pre = np.array(sorted(set(np.concatenate([pre_idx_lists[e], forced_lists[e]]) if forced_lists[e].size else pre_idx_lists[e])), dtype=int)
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
        best = None; best_gain = -np.inf
        for (e,i) in remaining:
            d = int(day_list[e][i])
            used_today = np.sum(day_list[e][idx_lists[e]]==d)
            if used_today >= daily_cap: continue
            S = J_list[e][i,:,:]
            M = Sinv + S @ Ainv @ S.T
            gain = chol_logdet(M) - logdet_Sinv
            if gain > best_gain:
                best_gain = gain; best = (e,i)
        if best is None:
            break
        e,i = best
        S = J_list[e][i,:,:]
        Minv = robust_inv_spd(Sigma_y + S @ Ainv @ S.T)
        Ainv = Ainv - Ainv @ S.T @ Minv @ S @ Ainv
        idx_lists[e] = np.append(idx_lists[e], i)
        total_selected += 1
        new_remaining = []
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

# -------------------- Generación y reparación de candidatos --------------------

def draw_time_workhour(rng: np.random.Generator, lo: float, hi: float, sim: SimulationConfig) -> float:
    for _ in range(400):
        t = rng.uniform(lo, hi)
        if is_workhour(t, sim):
            return t
    # fallback: empujar al siguiente hábil
    t = rng.uniform(lo, hi)
    return snap_to_next_workhour(t, sim)


def repair_design_operational(phi: Design, sim: SimulationConfig, cfg: FIMDGlobalConfig) -> Design:
    """Ajusta t12,t23,tN1,tN2 al siguiente slot hábil si no lo son.
       Mantiene separaciones mínimas: t23 ≥ max(t12+6, 24) y tN2 ≥ tN1+6.
    """
    T1,T2,T3 = phi.T1, phi.T2, phi.T3
    t12, t23 = phi.t12, phi.t23
    tN1, tN2 = phi.tN1, phi.tN2
    if cfg.setpoints_in_workhours:
        if not is_workhour(t12, sim):
            t12 = snap_to_next_workhour(t12, sim)
        min_t23 = max(t12 + 6.0, 24.0)
        if t23 < min_t23: t23 = min_t23
        if not is_workhour(t23, sim):
            t23 = snap_to_next_workhour(t23, sim)
    if cfg.feeds_in_workhours:
        if not is_workhour(tN1, sim):
            tN1 = snap_to_next_workhour(tN1, sim)
        min_tN2 = max(tN1 + 6.0, 20.0)
        if tN2 < min_tN2: tN2 = min_tN2
        if not is_workhour(tN2, sim):
            tN2 = snap_to_next_workhour(tN2, sim)
    return Design(T1,T2,T3,t12,t23,phi.doseN1_mgL,tN1,phi.doseN2_mgL,tN2,phi.tauN)


def sample_designs(n: int, rng: np.random.Generator, sim: SimulationConfig,
                   feeds_in_workhours: bool = True, setpoints_in_workhours: bool = True,
                   repair_ops_to_workhours: bool = True) -> List[Design]:
    designs: List[Design] = []
    for _ in range(n):
        T1,T2,T3 = rng.uniform(16.0, 28.0, size=3)
        # set-points en horario hábil
        if setpoints_in_workhours:
            t12 = draw_time_workhour(rng, 6.0, 48.0, sim)
            t23 = draw_time_workhour(rng, max(t12+6.0, 24.0), 120.0, sim)
        else:
            t12 = rng.uniform(6.0, 48.0)
            t23 = rng.uniform(max(t12+6.0, 24.0), 120.0)
        # pulsos en horario hábil
        if feeds_in_workhours:
            tN1 = draw_time_workhour(rng, 10.0, 120.0, sim)
            tN2 = draw_time_workhour(rng, max(tN1+6.0, 20.0), 160.0, sim)
        else:
            tN1 = rng.uniform(10.0, 120.0)
            tN2 = rng.uniform(max(tN1+6.0, 20.0), 160.0)
        dose1 = rng.uniform(0.0, 200.0); dose2 = rng.uniform(0.0, 200.0)
        tauN = rng.uniform(0.1, 0.5)
        d = Design(T1,T2,T3,t12,t23,dose1,tN1,dose2,tN2,tauN)
        if repair_ops_to_workhours:
            d = repair_design_operational(d, sim, FIMDGlobalConfig(setpoints_in_workhours=setpoints_in_workhours,
                                                                    feeds_in_workhours=feeds_in_workhours,
                                                                    repair_ops_to_workhours=repair_ops_to_workhours))
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
            print("="*80)
            print(f"[GLOBAL FIMD FORCED] Iter {k+1}/{cfg.n_select} (candidatos={len(phi_list)})")

        current_total = int(np.sum([len(v) for v in idxsel_bank.values()]))
        K_target = min(K_total, current_total + cfg.K_per_exp)

        for j, phi in enumerate(phi_list):
            if j in chosen_idx: continue
            # reparar por si acaso (si vino de fuera)
            if cfg.repair_ops_to_workhours:
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
                    pre_idx_lists.append(idxsel_bank.get(e, np.array([],dtype=int)))
                    forced_lists.append(forced_bank[e])
                    forced_flags.append(forced_flags_bank[e])
                alloc_lists = global_greedy_allocate_forced(J_list, meas.Sigma, allowed_list, day_list,
                                                            forced_lists, forced_flags, pre_idx_lists,
                                                            K_target, daily_cap=cfg.daily_cap)
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
            print(f"[GLOBAL FIMD FORCED] -> elegido idx={best_idx} | logdet={best_obj:.3f} | asignación={counts}")

    chosen_times = [ t_bank[e][idxsel_bank[e]] if len(idxsel_bank[e]) else np.array([]) for e in chosen_idx ]
    return chosen_idx, F_accum, chosen_times, t_bank, idxsel_bank, forced_bank

# -------------------- Main con export a Excel --------------------
if __name__ == "__main__":
    p = BeaudeauModel1Params()
    sim = SimulationConfig(t_end=200.0, dt=1.0, method="Radau", stop_on_s_exhausted=False,
                           start_weekday=0, start_hour=8, end_hour_exclusive=17)
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
        force_postpulse=True, force_at_pulse=True, verbose=True
    )

    print(f"[MAIN] Generando {cfg.n_candidates} candidatos con t12/t23 y tN1/tN2 en horario hábil…")
    rng = np.random.default_rng(cfg.seed)
    candidates = sample_designs(cfg.n_candidates, rng, sim,
                                feeds_in_workhours=cfg.feeds_in_workhours,
                                setpoints_in_workhours=cfg.setpoints_in_workhours,
                                repair_ops_to_workhours=cfg.repair_ops_to_workhours)

    print(f"[MAIN] Seleccionando {cfg.n_select} diseños (GLOBAL+FORCED)…")
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
        cal_for = ", ".join(map(lambda t: map_time_to_calendar(float(t), sim), forced))
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
            "n_times": int(sel.size),
            "times_h": list(map(float, sel)),
            "forced_times_h": list(map(float, forced)),
            "times_calendar_preview": cal_sel,
            "forced_calendar": cal_for,
            "ops_calendar": ops_cal,
        })

    df = pd.DataFrame(rows)
    out_path = "FIMD_GLOBAL_opsWorkHours_plusSampling_plan.xlsx"
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="plan")
        cols = ["rank","idx","T1","T2","T3","t12_h","t23_h","N1_mgL","tN1_h","N2_mgL","tN2_h","tau_h","n_times","times_calendar_preview","forced_calendar","ops_calendar"]
        df[cols].to_excel(w, index=False, sheet_name="compact")

    sign, logdet = npl.slogdet(F_final + 1e-12*np.eye(len(theta_names)))
    print(f"\n[RESULT] log det(FIM acumulada) = {logdet:.3f}")
    print(f"[FILES] Exportado plan: {out_path}")
    print("Restricción operativa: t12,t23,tN1,tN2 ∈ horario hábil (L–V 08–17). Muestreo ampliado: K_per_exp=18, 2/día.")
