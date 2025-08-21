"""
Programación y calendarización (3 ICS) con reglas adicionales
--------------------------------------------------------------
Extiende `schedule_calendarize_triple_with_constraints.py` con:
  • Evento adicional en **t = 240 h** desde el inicio de cada experimento:
      - "Cambiar setpoint a **5 °C** (sedimentación)" en el **calendario de temperatura**.
      - Si t=240 h cae **fuera de horario laboral**, reprograma al **siguiente día hábil a las 09:00**.
  • Evento del **día hábil siguiente** a la sedimentación (09:00–13:00):
      - "**Muestreo final + descube + captura de aromas**" (se agenda en el **calendario de temperatura**).
  • Resto igual: 3 .ics → muestreo / operación temperatura / operación nutriente.

Uso típico:
  python schedule_calendarize_triple_with_constraints_v2.py \
    --excel FIMD_GLOBAL_v7_opsLunch_prePostPulse_targetN450.xlsx \
    --sheet plan \
    --tz America/Santiago \
    --monday-hour 12 \
    --exp-duration-h 240 \
    --max-concurrent 3 \
    --sedimentation-after-h 240 \
    --biz-start-hour 9 --biz-end-hour 17 \
    --sampling-duration-min 15 --operational-duration-min 15 \
    --ics-sampling sampling_plan.ics \
    --ics-oper-temp oper_temp_plan.ics \
    --ics-oper-nutr oper_nutr_plan.ics

Nota: si quieres offset fijo (sin DST) usa --fixed-utc-offset -04:00
"""
from __future__ import annotations
import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timedelta, date, time, timezone
from zoneinfo import ZoneInfo
import uuid
import heapq
import pandas as pd

# --------------------- Configuración ---------------------
@dataclass
class CalendarConfig:
    tz: str = "America/Santiago"      # Chile (con DST)
    fixed_utc_offset: str | None = None  # opcional, ej. "-04:00" (sin DST)
    start_date: date | None = None     # si None → próximo lunes
    monday_hour: int = 12              # hora local de inicio el lunes para inoculación
    sampling_duration_min: int = 15
    operational_duration_min: int = 15
    title_sampling: str = "Muestreo Fermentación"
    title_oper_temp: str = "Operación Temperatura"
    title_oper_nutr: str = "Operación Nutriente"
    location_sampling: str = "Laboratorio"
    location_operational: str = "Bodega"
    # Horario laboral genérico para eventos especiales (sedimentación, post)
    biz_start_hour: int = 9
    biz_end_hour: int = 17

# --------------------- Utilidades tiempo ---------------------

def parse_fixed_offset(s: str) -> timezone:
    sign = 1 if s.strip()[0] == "+" else -1
    hh, mm = map(int, s.strip()[1:].split(":"))
    return timezone(sign * timedelta(hours=hh, minutes=mm))


def tzinfo_from_cfg(cfg: CalendarConfig):
    return parse_fixed_offset(cfg.fixed_utc_offset) if cfg.fixed_utc_offset else ZoneInfo(cfg.tz)


def next_weekday(d: date, target_wd: int) -> date:
    # 0=Mon..6=Sun
    days_ahead = target_wd - d.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return d + timedelta(days=days_ahead)


def compute_first_monday(cfg: CalendarConfig):
    tz = tzinfo_from_cfg(cfg)
    today = datetime.now(tz).date()
    base_date = next_weekday(today, 0) if cfg.start_date is None else cfg.start_date
    return datetime(base_date.year, base_date.month, base_date.day, cfg.monday_hour, 0, 0, tzinfo=tz)


def is_business_hour_dt(dt: datetime, cfg: CalendarConfig) -> bool:
    wd = dt.weekday()  # 0=Mon..6=Sun
    if wd > 4:
        return False
    h = dt.hour + dt.minute/60
    return (cfg.biz_start_hour <= h < cfg.biz_end_hour)


def next_business_day_at(dt: datetime, hour: int, cfg: CalendarConfig) -> datetime:
    tz = dt.tzinfo
    # ir al día siguiente (al menos) y buscar primer hábil
    d = dt.date() + timedelta(days=1)
    while d.weekday() > 4:  # 5=Sat,6=Sun
        d += timedelta(days=1)
    return datetime(d.year, d.month, d.day, hour, 0, 0, tzinfo=tz)

# --------------------- Lectura plan ---------------------

def parse_times_cell(val) -> list[float]:
    if isinstance(val, list):
        return [float(x) for x in val]
    if isinstance(val, str):
        try:
            arr = ast.literal_eval(val)
            if isinstance(arr, (list, tuple)):
                return [float(x) for x in arr]
        except Exception:
            pass
    return []

# --------------------- Construcción ICS ---------------------

def ics_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def format_dt_local(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S")


def build_ics(events: list[dict], tzid: str, cal_name: str) -> str:
    now_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//FIMD//Scheduler//EN",
        "CALSCALE:GREGORIAN",
        f"X-WR-CALNAME:{ics_escape(cal_name)}",
        f"X-WR-TIMEZONE:{tzid}",
    ]
    for ev in events:
        uid = ev.get("uid") or f"{uuid.uuid4()}@fimd"
        dtstart_local = format_dt_local(ev["start"])  # local time
        dtend_local   = format_dt_local(ev["end"])    # local time
        summary = ics_escape(ev["summary"]) 
        description = ics_escape(ev.get("description", ""))
        location = ics_escape(ev.get("location", ""))
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{now_utc}",
            f"DTSTART;TZID={tzid}:{dtstart_local}",
            f"DTEND;TZID={tzid}:{dtend_local}",
            f"SUMMARY:{summary}",
            f"LOCATION:{location}",
            f"DESCRIPTION:{description}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"

# --------------------- Programador con capacidad ---------------------

def schedule_starts(df: pd.DataFrame, cfg: CalendarConfig, exp_duration_h: float, max_concurrent: int) -> dict:
    """Devuelve dict {row_index: start_datetime} respetando:
       - sólo lunes a monday_hour
       - máximo max_concurrent activos
       - duración por experimento (de df['dur_h'] si existe; si no, exp_duration_h)
       - orden por 'rank' ascendente.
    """
    df2 = df.copy()
    if "dur_h" in df2.columns:
        df2["dur_h"] = df2["dur_h"].fillna(exp_duration_h)
    else:
        df2["dur_h"] = exp_duration_h
    df2 = df2.sort_values(by=["rank"]).reset_index()  # preserva index original en 'index'

    tz = tzinfo_from_cfg(cfg)
    monday = compute_first_monday(cfg)

    active_heap: list[tuple[datetime,int]] = []  # (end_dt, count)
    assigned: dict[int, datetime] = {}

    ptr = 0
    total = len(df2)
    while ptr < total:
        # liberar activos terminados antes del lunes actual
        while active_heap and active_heap[0][0] <= monday:
            heapq.heappop(active_heap)
        capacity = max(0, max_concurrent - len(active_heap))
        # asignar hasta 'capacity' experimentos a este lunes
        assigned_this_monday = 0
        while assigned_this_monday < capacity and ptr < total:
            row = df2.iloc[ptr]
            orig_idx = int(row["index"])  # índice original en df
            dur_h = float(row["dur_h"]) if not pd.isna(row["dur_h"]) else exp_duration_h
            start_dt = monday
            end_dt = start_dt + timedelta(hours=dur_h)
            assigned[orig_idx] = start_dt
            heapq.heappush(active_heap, (end_dt, 1))
            ptr += 1
            assigned_this_monday += 1
        # ir al lunes siguiente
        monday = monday + timedelta(days=7)
    return assigned

# --------------------- Eventos (sampling / oper temp / oper nutr) ---------------------

def build_sampling_events(df: pd.DataFrame, starts: dict, cfg: CalendarConfig) -> list[dict]:
    events: list[dict] = []
    for ridx, r in df.iterrows():
        if ridx not in starts: continue
        start0 = starts[ridx]
        idx = int(r.get("idx", ridx)); rank = int(r.get("rank", 0))
        T1,T2,T3 = float(r.get("T1",0)), float(r.get("T2",0)), float(r.get("T3",0))
        t12,t23 = float(r.get("t12_h",0)), float(r.get("t23_h",0))
        N1,tN1 = float(r.get("N1_mgL",0)), float(r.get("tN1_h",0))
        N2,tN2 = float(r.get("N2_mgL",0)), float(r.get("tN2_h",0))
        tau = float(r.get("tau_h",0))
        times_h = parse_times_cell(r.get("times_h", []))
        # incluir t=0
        all_times = sorted(set([0.0] + [float(t) for t in times_h]))
        desc = (
            f"Diseño idx={idx} (rank {rank})\n"
            f"T1={T1:.1f}C, T2={T2:.1f}C, T3={T3:.1f}C; t12={t12:.1f}h, t23={t23:.1f}h\n"
            f"N1={N1:.0f} mg/L @ {tN1:.1f}h, N2={N2:.0f} mg/L @ {tN2:.1f}h, tau={tau:.2f}h"
        )
        for k, t in enumerate(all_times, start=1):
            st = start0 + timedelta(hours=float(t))
            ev = {
                "uid": f"S-{idx}-{k}-{int(st.timestamp())}@fimd",
                "start": st,
                "end": st + timedelta(minutes=cfg.sampling_duration_min),
                "summary": f"{cfg.title_sampling} — Exp #{rank} (idx={idx})",
                "location": cfg.location_sampling,
                "description": desc + f"\nMuestra #{k} a t={t:.1f} h",
            }
            events.append(ev)
    return events


def build_oper_temp_events(df: pd.DataFrame, starts: dict, cfg: CalendarConfig,
                           sedimentation_after_h: float) -> list[dict]:
    events: list[dict] = []
    for ridx, r in df.iterrows():
        if ridx not in starts: continue
        start0 = starts[ridx]
        idx = int(r.get("idx", ridx)); rank = int(r.get("rank", 0))
        T1,T2,T3 = float(r.get("T1",0)), float(r.get("T2",0)), float(r.get("T3",0))
        t12,t23 = float(r.get("t12_h",0)), float(r.get("t23_h",0))
        # Inicio (T1)
        events.append({
            "uid": f"T-START-{idx}-{int(start0.timestamp())}@fimd",
            "start": start0,
            "end": start0 + timedelta(minutes=cfg.operational_duration_min),
            "summary": f"{cfg.title_oper_temp} — INICIO Exp #{rank} (idx={idx})",
            "location": cfg.location_operational,
            "description": f"Set-point inicial T1={T1:.1f}C",
        })
        # Cambios set-point t12, t23
        if t12 > 0:
            t12_dt = start0 + timedelta(hours=t12)
            events.append({
                "uid": f"T-T12-{idx}-{int(t12_dt.timestamp())}@fimd",
                "start": t12_dt,
                "end": t12_dt + timedelta(minutes=cfg.operational_duration_min),
                "summary": f"{cfg.title_oper_temp} — Cambio set-point T1→T2 (Exp #{rank})",
                "location": cfg.location_operational,
                "description": f"T1={T1:.1f}C → T2={T2:.1f}C a t={t12:.1f} h",
            })
        if t23 > 0:
            t23_dt = start0 + timedelta(hours=t23)
            events.append({
                "uid": f"T-T23-{idx}-{int(t23_dt.timestamp())}@fimd",
                "start": t23_dt,
                "end": t23_dt + timedelta(minutes=cfg.operational_duration_min),
                "summary": f"{cfg.title_oper_temp} — Cambio set-point T2→T3 (Exp #{rank})",
                "location": cfg.location_operational,
                "description": f"T2={T2:.1f}C → T3={T3:.1f}C a t={t23:.1f} h",
            })
        # Sedimentación a 5°C en t = sedimentation_after_h (ajustada a horario laboral)
        sed_dt = start0 + timedelta(hours=sedimentation_after_h)
        if not is_business_hour_dt(sed_dt, cfg):
            sed_dt = next_business_day_at(sed_dt, cfg.biz_start_hour, cfg)
        events.append({
            "uid": f"T-SED-{idx}-{int(sed_dt.timestamp())}@fimd",
            "start": sed_dt,
            "end": sed_dt + timedelta(minutes=cfg.operational_duration_min),
            "summary": f"{cfg.title_oper_temp} — Sedimentación a 5°C (Exp #{rank})",
            "location": cfg.location_operational,
            "description": "Cambiar set-point a 5 °C (sedimentación).",
        })
        # Cita del día hábil siguiente 09:00–13:00 (muestreo final + descube + captura de aromas)
        post_day = next_business_day_at(sed_dt, cfg.biz_start_hour, cfg)
        post_end = post_day + timedelta(hours=4)
        events.append({
            "uid": f"T-POSTFIN-{idx}-{int(post_day.timestamp())}@fimd",
            "start": post_day,
            "end": post_end,
            "summary": f"Operación — Muestreo final + Descube + Captura de aromas (Exp #{rank})",
            "location": cfg.location_operational,
            "description": "Bloque 09:00–13:00 para muestreo final, descube y captura de aromas.",
        })
    return events


def build_oper_nutr_events(df: pd.DataFrame, starts: dict, cfg: CalendarConfig) -> list[dict]:
    events: list[dict] = []
    for ridx, r in df.iterrows():
        if ridx not in starts: continue
        start0 = starts[ridx]
        idx = int(r.get("idx", ridx)); rank = int(r.get("rank", 0))
        N1,tN1 = float(r.get("N1_mgL",0)), float(r.get("tN1_h",0))
        N2,tN2 = float(r.get("N2_mgL",0)), float(r.get("tN2_h",0))
        tau = float(r.get("tau_h",0))
        if tN1 > 0 and N1 > 0:
            tN1_dt = start0 + timedelta(hours=tN1)
            events.append({
                "uid": f"N-N1-{idx}-{int(tN1_dt.timestamp())}@fimd",
                "start": tN1_dt,
                "end": tN1_dt + timedelta(minutes=cfg.operational_duration_min),
                "summary": f"{cfg.title_oper_nutr} — Pulso N1 (Exp #{rank})",
                "location": cfg.location_operational,
                "description": f"Aplicar N1={N1:.0f} mg/L, τ={tau:.2f} h a t={tN1:.1f} h",
            })
        if tN2 > 0 and N2 > 0:
            tN2_dt = start0 + timedelta(hours=tN2)
            events.append({
                "uid": f"N-N2-{idx}-{int(tN2_dt.timestamp())}@fimd",
                "start": tN2_dt,
                "end": tN2_dt + timedelta(minutes=cfg.operational_duration_min),
                "summary": f"{cfg.title_oper_nutr} — Pulso N2 (Exp #{rank})",
                "location": cfg.location_operational,
                "description": f"Aplicar N2={N2:.0f} mg/L, τ={tau:.2f} h a t={tN2:.1f} h",
            })
    return events

# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--excel', default='FIMD_GLOBAL_v7_opsLunch_prePostPulse_targetN450.xlsx')
    ap.add_argument('--sheet', default='plan')
    ap.add_argument('--tz', default='America/Santiago')
    ap.add_argument('--fixed-utc-offset', default=None, help='ej. -04:00 para horario fijo (sin DST)')
    ap.add_argument('--start-date', default='2025-08-18', help='YYYY-MM-DD; si omites, próximo lunes')
    ap.add_argument('--monday-hour', type=int, default=12)
    ap.add_argument('--exp-duration-h', type=float, default=240.0)
    ap.add_argument('--max-concurrent', type=int, default=3)
    ap.add_argument('--sedimentation-after-h', type=float, default=240.0)
    ap.add_argument('--biz-start-hour', type=int, default=9)
    ap.add_argument('--biz-end-hour', type=int, default=17)
    ap.add_argument('--sampling-duration-min', type=int, default=15)
    ap.add_argument('--operational-duration-min', type=int, default=15)
    ap.add_argument('--ics-sampling', default='sampling_plan.ics')
    ap.add_argument('--ics-oper-temp', default='oper_temp_plan.ics')
    ap.add_argument('--ics-oper-nutr', default='oper_nutr_plan.ics')
    args = ap.parse_args()

    cfg = CalendarConfig(
        tz=args.tz,
        fixed_utc_offset=args.fixed_utc_offset,
        start_date=(date.fromisoformat(args.start_date) if args.start_date else None),
        monday_hour=args.monday_hour,
        sampling_duration_min=args.sampling_duration_min,
        operational_duration_min=args.operational_duration_min,
        biz_start_hour=args.biz_start_hour,
        biz_end_hour=args.biz_end_hour,
    )

    # Leer plan
    df = pd.read_excel(args.excel, sheet_name=args.sheet)

    # Programar inicios respetando capacidad y lunes HH:00
    starts = schedule_starts(df, cfg, exp_duration_h=args.exp_duration_h, max_concurrent=args.max_concurrent)

    # Crear eventos
    ev_sampling   = build_sampling_events(df, starts, cfg)
    ev_oper_temp  = build_oper_temp_events(df, starts, cfg, sedimentation_after_h=args.sedimentation_after_h)
    ev_oper_nutr  = build_oper_nutr_events(df, starts, cfg)

    # Escribir ICS
    tzid = args.fixed_utc_offset if args.fixed_utc_offset else args.tz
    ics_s = build_ics(ev_sampling,  tzid, cal_name='FIMD — Muestreo')
    ics_t = build_ics(ev_oper_temp, tzid, cal_name='FIMD — Operación Temperatura')
    ics_n = build_ics(ev_oper_nutr, tzid, cal_name='FIMD — Operación Nutriente')

    with open(args.ics_sampling, 'w', encoding='utf-8') as f:
        f.write(ics_s)
    with open(args.ics_oper_temp, 'w', encoding='utf-8') as f:
        f.write(ics_t)
    with open(args.ics_oper_nutr, 'w', encoding='utf-8') as f:
        f.write(ics_n)

    # Estadísticas de la agenda
    first_start = min(starts.values()) if starts else None
    ends = []
    for ridx, r in df.iterrows():
        if ridx not in starts: continue
        dur_h = float(r.get('dur_h', args.exp_duration_h)) if not pd.isna(r.get('dur_h', args.exp_duration_h)) else args.exp_duration_h
        ends.append(starts[ridx] + timedelta(hours=dur_h))
    last_end = max(ends) if ends else None
    if first_start and last_end:
        total_days = (last_end - first_start).days + 1
        print(f"Plan calendarizado desde {first_start} hasta {last_end} (≈{total_days} días).")
    print(f"ICS muestreo: {args.ics_sampling} ({len(ev_sampling)} eventos)")
    print(f"ICS oper. temp: {args.ics_oper_temp} ({len(ev_oper_temp)} eventos)")
    print(f"ICS oper. nutr: {args.ics_oper_nutr} ({len(ev_oper_nutr)} eventos)")

if __name__ == '__main__':
    main()
