from __future__ import annotations
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# ---------- 地理：从(lat,lon)沿方位角前进 distance_m ----------
EARTH_R = 6371000.0  # meters

def destination_point(lat_deg, lon_deg, bearing_deg, distance_m):
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg)
    d = distance_m / EARTH_R

    lat2 = math.asin(math.sin(lat1)*math.cos(d) + math.cos(lat1)*math.sin(d)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d)*math.cos(lat1),
                             math.cos(d) - math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

# ---------- 路线：随机坡度剖面（按距离采样） ----------
def generate_grade_profile(total_m, step_m=20, seed=1,
                           seg_len_m_range=(200, 1200),
                           grade_std=0.02, grade_clip=(-0.06, 0.08),
                           smooth_window=21):
    rng = np.random.default_rng(seed)
    s = np.arange(0, total_m + step_m, step_m)

    grade = np.zeros_like(s, dtype=float)
    idx = 0
    while idx < len(s):
        seg_len = int(rng.integers(seg_len_m_range[0], seg_len_m_range[1] + 1))
        seg_steps = max(1, seg_len // step_m)
        g = rng.normal(0.0, grade_std)
        grade[idx: idx + seg_steps] = g
        idx += seg_steps

    # 长周期起伏
    x = np.linspace(0, 2*np.pi, len(s))
    grade += 0.008*np.sin(1.2*x) + 0.004*np.sin(5.1*x + 0.7)

    # 平滑
    if smooth_window and smooth_window > 1:
        if smooth_window % 2 == 0:
            smooth_window += 1
        kernel = np.ones(smooth_window) / smooth_window
        grade = np.convolve(grade, kernel, mode="same")

    grade = np.clip(grade, grade_clip[0], grade_clip[1])
    return s, grade

# ---------- 分区功率策略：坡度 -> 功率 ----------
ZONE_CONFIG = {
    "Z2": {"if_low": 0.56, "if_high": 0.75, "a_up": 4.5, "a_down": 2.5, "softpedal_w": 110},
    "Z3": {"if_low": 0.76, "if_high": 0.90, "a_up": 5.5, "a_down": 3.0, "softpedal_w": 130},
    "Z2Z3": {"if_low": 0.60, "if_high": 0.85, "a_up": 5.0, "a_down": 2.8, "softpedal_w": 120},
}

def power_from_grade(grade, ftp=250, zone="Z2Z3", downhill_threshold=-0.03):
    cfg = ZONE_CONFIG[zone]
    if_low, if_high = cfg["if_low"], cfg["if_high"]
    IF = (if_low + if_high) / 2.0
    P_base = IF * ftp
    P_low = if_low * ftp
    P_high = if_high * ftp

    grade = np.asarray(grade, dtype=float)
    gain = np.where(grade >= 0, 1 + cfg["a_up"] * grade, 1 + cfg["a_down"] * grade)
    P = P_base * gain

    # 明显下坡轻踩/滑行倾向
    P = np.where(grade <= downhill_threshold, np.minimum(P, cfg["softpedal_w"]), P)
    P = np.clip(P, P_low, P_high)
    return P

# ---------- 心率：一阶滞后 + 轻微漂移 ----------
def simulate_hr(power_w, ftp, hr_rest=55, hr_max=185, dt_s=1.0, tau_s=45.0, drift_bpm_per_min=0.25):
    power_w = np.asarray(power_w, dtype=float)
    hr = np.zeros_like(power_w)
    hr[0] = hr_rest

    for i in range(1, len(power_w)):
        x = np.clip(power_w[i] / ftp, 0, 1.2)  # 强度归一化
        # 目标心率（用 sqrt 让 Z2-Z3 更贴近现实：前段上升快，后段趋缓）
        f = math.sqrt(x / 1.05) if x > 0 else 0.0
        f = float(np.clip(f, 0.0, 1.0))
        hr_target = hr_rest + (hr_max - hr_rest) * f

        hr[i] = hr[i-1] + (hr_target - hr[i-1]) * (dt_s / tau_s)
        hr[i] += (drift_bpm_per_min / 60.0) * dt_s

    return np.round(hr).astype(int)

# ---------- GPX 写出（含扩展字段） ----------
def build_gpx(points_df: pd.DataFrame, activity_name="Sim Ride"):
    """
    points_df columns required:
    time (datetime64[ns, UTC]), lat, lon, ele_m, hr_bpm, cad_rpm, power_w
    """
    def iso(t):
        # GPX 要求 UTC + Z
        return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    header = f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1"
 creator="simple-sim"
 xmlns="http://www.topografix.com/GPX/1/1"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
 xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1"
 xmlns:power="http://www.garmin.com/xmlschemas/PowerExtension/v1"
 xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">
  <metadata>
    <name>{activity_name}</name>
    <time>{iso(points_df['time'].iloc[0].to_pydatetime())}</time>
  </metadata>
  <trk>
    <name>{activity_name}</name>
    <trkseg>
"""
    tail = """    </trkseg>
  </trk>
</gpx>
"""
    lines = [header]
    for row in points_df.itertuples(index=False):
        t = row.time.to_pydatetime()
        lat = row.lat
        lon = row.lon
        ele = row.ele_m
        hr = int(row.hr_bpm)
        cad = int(row.cad_rpm)
        pwr = int(row.power_w)

        # 扩展：gpxtpx 写 HR/CAD；power 扩展写功率
        lines.append(
f"""      <trkpt lat="{lat:.8f}" lon="{lon:.8f}">
        <ele>{ele:.2f}</ele>
        <time>{iso(t)}</time>
        <extensions>
          <gpxtpx:TrackPointExtension>
            <gpxtpx:hr>{hr}</gpxtpx:hr>
            <gpxtpx:cad>{cad}</gpxtpx:cad>
          </gpxtpx:TrackPointExtension>
          <power:PowerExtension>
            <power:Watts>{pwr}</power:Watts>
          </power:PowerExtension>
        </extensions>
      </trkpt>
"""
        )
    lines.append(tail)
    return "".join(lines)

# ---------- 主流程：生成距离-功率/心率，并导出 GPX/CSV ----------
def make_demo(
    start_lat=0.0, start_lon=140.0,  # “海上”示例：赤道附近太平洋
    bearing_deg=90.0,                # 向东
    total_km=30.0,
    speed_kmh=28.0,                  # 简化为恒速
    step_s=1,
    ftp=260,
    zone="Z2Z3",
    seed=1
):
    total_m = int(total_km * 1000)
    v_mps = speed_kmh / 3.6
    step_m = v_mps * step_s

    # 用“按时间步进”生成距离点
    n = int(total_m / step_m) + 1
    s_m = np.arange(n) * step_m
    s_m = np.clip(s_m, 0, total_m)

    # 生成坡度剖面（按距离采样），然后插值到 s_m
    s_prof, grade_prof = generate_grade_profile(total_m=total_m, step_m=20, seed=seed)
    grade = np.interp(s_m, s_prof, grade_prof)

    # 海拔（从 0m 起步，积分 grade * ds）
    elev = np.zeros_like(s_m)
    for i in range(1, len(s_m)):
        ds = s_m[i] - s_m[i-1]
        elev[i] = elev[i-1] + grade[i] * ds

    # 功率（分区策略：坡度->功率）
    power_w = power_from_grade(grade, ftp=ftp, zone=zone)

    # 踏频（demo：用固定踏频 + 少量噪声；后续可做自动变速）
    rng = np.random.default_rng(seed + 10)
    cad = np.clip(85 + rng.normal(0, 2.0, size=len(s_m)), 70, 100).round().astype(int)

    # 心率（由功率驱动）
    hr = simulate_hr(power_w, ftp=ftp, hr_rest=55, hr_max=185, dt_s=step_s)

    # 坐标：沿固定方向推进
    lat = np.zeros_like(s_m, dtype=float)
    lon = np.zeros_like(s_m, dtype=float)
    for i, d in enumerate(s_m):
        lat[i], lon[i] = destination_point(start_lat, start_lon, bearing_deg, float(d))

    # 时间
    t0 = datetime.now(timezone.utc).replace(microsecond=0)
    times = [t0 + timedelta(seconds=i*step_s) for i in range(len(s_m))]

    df = pd.DataFrame({
        "time": pd.to_datetime(times, utc=True),
        "s_m": s_m,
        "dist_km": s_m / 1000.0,
        "lat": lat,
        "lon": lon,
        "grade": grade,
        "grade_pct": grade * 100,
        "ele_m": elev,
        "speed_kmh": speed_kmh,
        "power_w": np.round(power_w).astype(int),
        "hr_bpm": hr,
        "cad_rpm": cad,
    })
    return df

if __name__ == "__main__":
    df = make_demo(
        start_lat=0.0, start_lon=140.0,
        bearing_deg=90.0,
        total_km=25.0,
        speed_kmh=30.0,
        ftp=260,
        zone="Z2Z3",
        seed=2
    )

    # 导出 CSV（距离-功率/心率等）
    df.to_csv("sim_ride.csv", index=False)

    # 导出 GPX
    gpx = build_gpx(df, activity_name="Sim Z2Z3 Ride (straight line)")
    with open("sim_ride.gpx", "w", encoding="utf-8") as f:
        f.write(gpx)

    # 同时简单检查：打印前几行
    print(df[["dist_km", "grade_pct", "power_w", "hr_bpm", "lat", "lon", "ele_m"]].head(10))
    print("Wrote: sim_ride.csv, sim_ride.gpx")
