"""
FIMD Global (multi-output + work-hours) — Beaudeau et al. (2023) Model 1
------------------------------------------------------------------------
Selecciona diseños (perfiles T + pulsos N) y **distribuye un presupuesto GLOBAL**
de tiempos de muestreo entre TODAS las fermentaciones (L–V, 08:00–17:00; máx 2/día/exp).

Idea
- Iteración externa (greedy) sobre **diseños**.
- En cada evaluación de candidato, se hace una **asignación global incremental**
  de tiempos (greedy D-óptimo rank-m) sobre los experimentos ya elegidos + el candidato,
  hasta un presupuesto `K_target = min(K_total, (n_elegidos+1)*K_per_exp)`.
- La asignación **respeta** el muestreo ya comprometido de iteraciones anteriores
  (no reoptimiza lo ya tomado; si quieres reoptimización total, puedo hacer una
  variante que lo permita a costo computacional mayor).

Incluye:
- Modelo Beaudeau 1 con X,S,N,E,CO2,Nintra,NST,A.
- Mediciones: X,S,N,E (ruidos configurables en `MeasurementModel`).
- Restricciones: L–V 08:00–17:00, 2 muestras/día por fermentación, inicio Lunes 08:00.
- Posibilidad de **forzar** que los pulsos N caigan en horario hábil al generar candidatos.

Requisitos: numpy, scipy
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from scipy.integrate import solve_ivp
import numpy.linalg as npl

STATE_IDX = {"X":0, "S":1, "N":2, "E":3, "CO2":4, "Nintra":5, "NST":6, "A":7}

# -------------------- Parámetros y configuraciones --------------------
@dataclass
class BeaudeauModel1Params:
    alpha_k1: float = 0.0287
    beta_k1: float  = 0.3
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
    start_weekday: int = 0
    start_hour: int = 8
    end_hour_exclusive: int = 17

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
    K_per_exp: int = 12
    daily_cap: int = 2
    seed: int = 42
    K_total: int | None = None  # si None, usa n_select * K_per_exp
    feeds_in_workhours: bool = True
    verbose: bool = True

# -------------------- Utilidades de diseño y calendario --------------------

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

def is_workhour(t: float, sim: SimulationConfig) -> bool:
    abs_hour = sim.start_hour + t
    d = int(np.floor(abs_hour/24.0))
    wd = (d + sim.start_weekday) % 7
    hod = abs_hour % 24.0
    return (wd <= 4) and (sim.start_hour <= hod < sim.end_hour_exclusive)

def compute_calendar_masks(t_eval: np.ndarray, sim: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
    abs_hour = sim.start_hour + t_eval
    day_index = np.floor(abs_hour/24.0).astype(int)
    hod = np.mod(abs_hour, 24.0)
    wd = np.mod(day_index + sim.start_weekday, 7)
    in_hours = (hod >= sim.start_hour) & (hod < sim.end_hour_exclusive)
    is_weekday = (wd <= 4)
    return (in_hours & is_weekday), day_index

# -------------------- Modelo --------------------

def rhs(t: float, x: np.ndarray, p: BeaudeauModel1Params, phi: Design) -> np.ndarray:
    X, S, N, E, CO2, Nintra, NST, A = x
    X = max(X, 1e-9); S = max(S, 0.0); N = max(N, 0.0)
    Nintra = max(Nintra, 1e-9); NST = max(NST, 0.0); E = max(E, 0.0)
    A = np.clip(A, 0.0, 1.0)
    T = T_of_t(t, phi)
    k1 = max(0.0, p.alpha_k1*T - p.beta_k1)
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
    dX=rX; dS=rS; dN=rN+rN_ext; dE=rE; dCO2=rCO2
    dNintra = rNintra - Nintra*(rX / max(X, 1e-9))
    dNST = rNST - NST*(rX / max(X, 1e-9))
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
        val = getattr(p_nom, name); h = rel_eps*max(abs(val), 1e-6)
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

# -------------------- Greedy global de tiempos --------------------

def fim_from_selected(J_list: List[np.ndarray], Sigma_y: np.ndarray, idx_lists: List[np.ndarray]) -> np.ndarray:
    n_theta = J_list[0].shape[2]
    F = np.zeros((n_theta,n_theta))
    Sinv = npl.inv(Sigma_y)
    for J3d, idx in zip(J_list, idx_lists):
        for i in idx:
            S = J3d[i,:,:]
            F += S.T @ Sinv @ S
    return F


def global_greedy_allocate(J_list: List[np.ndarray], Sigma_y: np.ndarray,
                           allowed_list: List[np.ndarray], day_list: List[np.ndarray],
                           pre_idx_lists: List[np.ndarray], K_target: int, daily_cap: int = 2) -> List[np.ndarray]:
    """Asigna tiempos globalmente (sobre todas las corridas) dada una selección previa.
    Devuelve idx_lists actualizadas (preselección + nuevos). No re-optimiza lo ya elegido.
    """
    n_exp = len(J_list)
    n_theta = J_list[0].shape[2]
    Sinv = npl.inv(Sigma_y)

    # Ainv inicial = (F_pre)^{-1}
    F_pre = fim_from_selected(J_list, Sigma_y, pre_idx_lists)
    Ainv = robust_inv_spd(F_pre + 1e-8*np.eye(n_theta))

    # Per-day counts iniciales considerando lo ya seleccionado
    per_day_counts: Dict[Tuple[int,int], int] = {}
    total_selected = 0
    for e in range(n_exp):
        for i in pre_idx_lists[e]:
            d = int(day_list[e][i])
            per_day_counts[(e,d)] = per_day_counts.get((e,d), 0) + 1
            total_selected += 1

    # Pool de candidatos restantes
    remaining: List[Tuple[int,int]] = []  # (exp, i)
    for e in range(n_exp):
        allowed = allowed_list[e]
        pre_set = set(pre_idx_lists[e].tolist())
        for i in np.where(allowed)[0]:
            if i in pre_set: continue
            remaining.append((e, int(i)))

    # Greedy hasta K_target
    while total_selected < K_target and remaining:
        best = None; best_gain = -np.inf
        for (e,i) in remaining:
            day = int(day_list[e][i])
            if per_day_counts.get((e,day),0) >= daily_cap:
                continue
            S = J_list[e][i,:,:]
            M = Sinv + S @ Ainv @ S.T
            gain = chol_logdet(M) - chol_logdet(Sinv)
            if gain > best_gain:
                best_gain = gain; best = (e,i)
        if best is None:
            break
        e,i = best
        # actualizar Ainv por Woodbury
        S = J_list[e][i,:,:]
        Minv = robust_inv_spd(Sigma_y + S @ Ainv @ S.T)
        Ainv = Ainv - Ainv @ S.T @ Minv @ S @ Ainv
        # registrar
        pre_idx_lists[e] = np.append(pre_idx_lists[e], i)
        dsel = int(day_list[e][i])
        per_day_counts[(e,dsel)] = per_day_counts.get((e,dsel),0) + 1
        total_selected += 1
        # quitar del pool y, si día lleno para ese exp, filtrar ese día
        new_remaining = []
        for (ee,ii) in remaining:
            if (ee==e and ii==i):
                continue
            if ee==e:
                dd = int(day_list[ee][ii])
                if per_day_counts.get((ee,dd),0) >= daily_cap:
                    continue
            new_remaining.append((ee,ii))
        remaining = new_remaining

    # devolver copias ordenadas
    out = []
    for e in range(n_exp):
        arr = np.array(sorted(set(pre_idx_lists[e].tolist())), dtype=int)
        out.append(arr)
    return out

# -------------------- Candidatos --------------------

def sample_designs(n: int, rng: np.random.Generator, sim: SimulationConfig,
                   feeds_in_workhours: bool = True) -> List[Design]:
    designs: List[Design] = []
    for _ in range(n):
        T1, T2, T3 = rng.uniform(16.0, 28.0, size=3)
        t12 = rng.uniform(6.0, 48.0)
        t23 = rng.uniform(max(t12 + 6.0, 24.0), 120.0)
        def draw_time(low, high):
            for _ in range(200):
                t = rng.uniform(low, high)
                if (not feeds_in_workhours) or is_workhour(t, sim):
                    return t
            return rng.uniform(low, high)
        tN1 = draw_time(10.0, 120.0)
        tN2 = draw_time(max(tN1 + 6.0, 20.0), 160.0)
        dose1 = rng.uniform(0.0, 200.0)
        dose2 = rng.uniform(0.0, 200.0)
        tauN = rng.uniform(0.1, 0.5)
        designs.append(Design(T1,T2,T3,t12,t23,dose1,tN1,dose2,tN2,tauN))
    return designs

# -------------------- FIMD Global (selección de diseños) --------------------

def fimd_global(phi_list: List[Design], theta_names: List[str], p_nom: BeaudeauModel1Params,
                sim: SimulationConfig, meas: MeasurementModel, cfg: FIMDGlobalConfig):
    rng = np.random.default_rng(cfg.seed)
    K_total = cfg.K_total if cfg.K_total is not None else cfg.n_select * cfg.K_per_exp

    chosen_idx: List[int] = []
    # Almacenamiento de info por experimento elegido
    J_bank: Dict[int, np.ndarray] = {}
    t_bank: Dict[int, np.ndarray] = {}
    allowed_bank: Dict[int, np.ndarray] = {}
    day_bank: Dict[int, np.ndarray] = {}
    idxsel_bank: Dict[int, np.ndarray] = {}

    n_theta = len(theta_names)
    F_accum = np.zeros((n_theta,n_theta))

    for k in range(cfg.n_select):
        best_obj = -np.inf
        best_idx = None
        best_idxsel_tmp: Dict[int, np.ndarray] | None = None

        if cfg.verbose:
            print("="*80)
            print(f"[GLOBAL FIMD] Iter {k+1}/{cfg.n_select} (candidatos={len(phi_list)})")

        # Conteo actual de muestras ya comprometidas
        current_total = int(np.sum([len(v) for v in idxsel_bank.values()]))
        K_target = min(K_total, current_total + cfg.K_per_exp)

        for j, phi in enumerate(phi_list):
            if j in chosen_idx: continue
            # Construir listas de J,t,allowed,day para exps actuales + candidato j
            J_list: List[np.ndarray] = []
            allowed_list: List[np.ndarray] = []
            day_list: List[np.ndarray] = []
            pre_idx_lists: List[np.ndarray] = []
            idx_to_exp: List[int] = []  # mapea posición local -> exp_id real

            exps = chosen_idx + [j]
            try:
                for e in exps:
                    if e not in J_bank:
                        J_bank[e], t_bank[e] = sensitivities_fd_multi(theta_names, p_nom, phi_list[e], sim)
                        allowed_bank[e], day_bank[e] = compute_calendar_masks(t_bank[e], sim)
                    J_list.append(J_bank[e]); allowed_list.append(allowed_bank[e]); day_list.append(day_bank[e])
                    pre_idx_lists.append(idxsel_bank.get(e, np.array([],dtype=int)))
                    idx_to_exp.append(e)
                # Asignación global incremental hasta K_target
                alloc_lists = global_greedy_allocate(J_list, meas.Sigma, allowed_list, day_list,
                                                     [arr.copy() for arr in pre_idx_lists],
                                                     K_target, daily_cap=cfg.daily_cap)
                # Construir FIM total hipotética
                F_total = np.zeros_like(F_accum)
                Sinv = npl.inv(meas.Sigma)
                for J3d, idxs in zip(J_list, alloc_lists):
                    for i in idxs:
                        S = J3d[i,:,:]
                        F_total += S.T @ Sinv @ S
            except Exception as e:
                if cfg.verbose:
                    print(f"[WARN] candidato {j} falló en evaluación global: {e}")
                continue

            obj = chol_logdet(F_total + 1e-12*np.eye(n_theta))
            if obj > best_obj:
                best_obj = obj
                best_idx = j
                # Guardar la asignación asociada al mejor candidato
                best_idxsel_tmp = {exp: sel for exp, sel in zip(exps, alloc_lists)}

        if best_idx is None:
            raise RuntimeError("No se pudo seleccionar ningún experimento (restricciones demasiado estrictas).")

        # Aceptar candidato y fijar la asignación resultante (para TODOS los ya elegidos + nuevo)
        chosen_idx.append(best_idx)
        # Actualizar bancos de idx seleccionados (manteniendo los no afectados)
        if best_idxsel_tmp is not None:
            for exp_id in chosen_idx:  # sólo exps presentes
                if exp_id in best_idxsel_tmp:
                    idxsel_bank[exp_id] = np.array(sorted(set(best_idxsel_tmp[exp_id].tolist())), dtype=int)
        # Recalcular F_accum por prolijidad
        F_accum = fim_from_selected([J_bank[e] for e in chosen_idx], meas.Sigma,
                                    [idxsel_bank[e] for e in chosen_idx])
        if cfg.verbose:
            counts = {e: len(idxsel_bank[e]) for e in chosen_idx}
            print(f"[GLOBAL FIMD] -> elegido idx={best_idx} | logdet={best_obj:.3f} | asignación por exp={counts}")

    # Salida
    chosen_times = []
    for e in chosen_idx:
        times = t_bank[e][idxsel_bank[e]] if len(idxsel_bank[e]) else np.array([])
        chosen_times.append(times)
    return chosen_idx, F_accum, chosen_times

# -------------------- Ejemplo de ejecución --------------------
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

    cfg = FIMDGlobalConfig(n_candidates=60, n_select=21, K_per_exp=12, daily_cap=2,
                           seed=42, K_total=None, feeds_in_workhours=True, verbose=True)

    print(f"[MAIN] Generando {cfg.n_candidates} candidatos (feeds_in_workhours={cfg.feeds_in_workhours})…")
    rng = np.random.default_rng(cfg.seed)
    def sample_designs(n: int, rng: np.random.Generator, sim: SimulationConfig,
                       feeds_in_workhours: bool = True) -> List[Design]:
        designs: List[Design] = []
        for _ in range(n):
            T1, T2, T3 = rng.uniform(16.0, 28.0, size=3)
            t12 = rng.uniform(6.0, 48.0)
            t23 = rng.uniform(max(t12 + 6.0, 24.0), 120.0)
            def draw_time(low, high):
                for _ in range(200):
                    t = rng.uniform(low, high)
                    if (not feeds_in_workhours) or is_workhour(t, sim):
                        return t
                return rng.uniform(low, high)
            tN1 = draw_time(10.0, 120.0)
            tN2 = draw_time(max(tN1 + 6.0, 20.0), 160.0)
            dose1 = rng.uniform(0.0, 200.0)
            dose2 = rng.uniform(0.0, 200.0)
            tauN = rng.uniform(0.1, 0.5)
            designs.append(Design(T1,T2,T3,t12,t23,dose1,tN1,dose2,tN2,tauN))
        return designs

    candidates = sample_designs(cfg.n_candidates, rng, sim, feeds_in_workhours=cfg.feeds_in_workhours)

    print(f"[MAIN] Seleccionando {cfg.n_select} diseños (GLOBAL) con presupuesto K_total = {cfg.K_total or cfg.n_select*cfg.K_per_exp}…")
    chosen_idx, F_final, chosen_times = fimd_global(candidates, theta_names, p, sim, meas, cfg)

    print("\nResumen de diseños seleccionados (global):")
    for rank, idx in enumerate(chosen_idx, start=1):
        d = candidates[idx]
        times = chosen_times[rank-1]
        preview = np.array2string(np.round(times[:min(6, times.size)],1), separator=", ")
        print(
            f"#{rank:02d} idx={idx} | T1={d.T1:.1f}°C T2={d.T2:.1f}°C T3={d.T3:.1f}°C | "
            f"t12={d.t12:.1f}h t23={d.t23:.1f}h | N1={d.doseN1_mgL:.0f}mg/L@{d.tN1:.1f}h "
            f"N2={d.doseN2_mgL:.0f}mg/L@{d.tN2:.1f}h τ={d.tauN:.2f}h | times≈{preview}"
        )

    print(f"\n[RESULT] log det(FIM acumulada) = {chol_logdet(F_final + 1e-12*np.eye(len(theta_names))):.3f}")
