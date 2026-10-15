"""
robot_final_report.py  -  end-of-mission results packager for the zebrafish inspection robot.

Put this file next to your main Blender script and call build_final_package(...) as the LAST
step of the main script (see bottom of file for the exact snippet).

Produces, under OUTPUT_DIR:
  final_report/stations.csv            one row per station (classification + habitat + arm + camera)
  final_report/confusion_matrix.csv    7x7
  final_report/class_metrics.csv       per-class precision / recall / F1 / support
  final_report/mission_report.json     everything machine-readable
  final_report/mission_report.txt       human summary
  final_report/index.html              self-contained dashboard (open in any browser)
  final_report/*.svg                   confusion heat-map, station map, zone chart (pure python, no matplotlib)
  arm_analysis/*.png                   7 time-series graphs + 6-panel figure (if matplotlib is installed)
  scans/live_###_<class>.png           what the robot camera saw (rendered from Blender)
  scans/scan_###_<class>.jpg           dataset image used for that station
  Robot_Scan_Results_<timestamp>.zip   everything above in one download
"""
import csv, json, math, os, random, shutil, time, zipfile, html
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))      # so blender_live_bridge.py (same folder) is importable

CLASSES = ['advanced', 'stage1', 'stage2-4', 'returned', 'empty', 'holder', 'dead']
HEALTHY = {'advanced', 'stage1', 'stage2-4'}
WEIGHTS = dict(T=.20, pH=.15, O2=.20, turb=.10, NH3=.15, NO3=.05, cond=.05, light=.05, flow=.05)


# ----------------------------------------------------------------------------------------------
# 0. OPTIONAL: 5 x 5 station grid (serpentine) - use instead of random egg placement if wanted
# ----------------------------------------------------------------------------------------------
def grid_station_path(n=5, spacing=10.0, origin=(0.0, 0.0)):
    """Serpentine path over an n x n grid of stations. Returns [(x, y, heading_rad), ...].
    heading points to the next station, so the boat can turn and park relative to each target."""
    pts = []
    for row in range(n):
        cols = range(n) if row % 2 == 0 else range(n - 1, -1, -1)
        for col in cols:
            pts.append((origin[0] + col * spacing, origin[1] + row * spacing))
    out = []
    for i, (x, y) in enumerate(pts):
        nx, ny = pts[i + 1] if i + 1 < len(pts) else pts[i]
        out.append((x, y, math.atan2(ny - y, nx - x)))
    return out


# ----------------------------------------------------------------------------------------------
# 1. Render the robot camera view + copy dataset image for each station
# ----------------------------------------------------------------------------------------------
def render_station_images(scene, robot_components, scan_results, egg_data, output_dir, resolution=224):
    """Jumps the timeline to the scan frame of each station and renders what robot_cam sees."""
    import bpy
    scans = Path(output_dir) / "scans"
    scans.mkdir(parents=True, exist_ok=True)
    by_id = {e["id"]: e for e in egg_data}
    old_path, old_fmt = scene.render.filepath, scene.render.image_settings.file_format
    scene.camera = robot_components["camera"]
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.image_settings.file_format = 'PNG'
    n_ok = 0
    for i, r in enumerate(scan_results):
        egg = by_id.get(r["target_id"], {})
        label = r.get("ground_truth", "unknown")
        r["live_image"] = r["scan_image"] = ""
        src = egg.get("image_source")
        if src and src != "placeholder" and os.path.exists(src):
            dst = scans / f"scan_{i:03d}_{label}{Path(src).suffix.lower()}"
            try:
                shutil.copy2(src, dst); r["scan_image"] = dst.name
            except Exception:
                pass
        if not r.get("reachable", True):
            continue
        try:
            scene.frame_set(int(r["frame"]))
            p = scans / f"live_{i:03d}_{label}.png"
            scene.render.filepath = str(p)
            bpy.ops.render.render(write_still=True)
            r["live_image"] = p.name; n_ok += 1
        except Exception as exc:
            print("render failed for station", i, repr(exc))
    scene.render.filepath, scene.render.image_settings.file_format = old_path, old_fmt
    scene.frame_set(scene.frame_start)
    print(f"[FINAL] rendered {n_ok} live camera images -> {scans}")


# ----------------------------------------------------------------------------------------------
# 2. Water / habitat model (same weights as your standout module)
# ----------------------------------------------------------------------------------------------
def _trap(v, bad_lo, ok_lo, ok_hi, bad_hi):
    if ok_lo <= v <= ok_hi: return 1.0
    if v <= bad_lo or v >= bad_hi: return 0.0
    return (v - bad_lo) / (ok_lo - bad_lo) if v < ok_lo else (bad_hi - v) / (bad_hi - ok_hi)

def make_water(rng):
    w = dict(T=rng.gauss(28.0, 1.2), pH=rng.gauss(7.2, 0.35), O2=rng.gauss(6.8, 0.8),
             turb=abs(rng.gauss(3.0, 1.5)), NH3=abs(rng.gauss(0.01, 0.006)), NO3=abs(rng.gauss(18, 8)),
             cond=rng.gauss(650, 120), light=rng.uniform(0.3, 0.9), flow=rng.uniform(0.03, 0.35))
    if rng.random() < 0.45:  # pollution event hits 2-3 parameters
        for k in rng.sample(['NH3', 'turb', 'O2', 'pH', 'T', 'NO3'], rng.choice([2, 3])):
            w[k] = {'NH3': w['NH3'] * 12, 'turb': w['turb'] * 6 + 5, 'O2': w['O2'] - 3.5,
                    'pH': w['pH'] - 1.3, 'T': w['T'] + 5.0, 'NO3': w['NO3'] * 5}[k]
    return w

def habitat_score(w):
    s = dict(T=_trap(w['T'], 22, 26, 29.5, 34), pH=_trap(w['pH'], 5.5, 6.8, 7.8, 9.0),
             O2=_trap(w['O2'], 3, 6, 12, 20), turb=_trap(w['turb'], -1, 0, 5, 25),
             NH3=_trap(w['NH3'], -1, 0, 0.02, 0.2), NO3=_trap(w['NO3'], -1, 0, 40, 100),
             cond=_trap(w['cond'], 100, 300, 1500, 3000), light=_trap(w['light'], 0, .3, .85, 1.2),
             flow=_trap(w['flow'], -.1, .03, .30, .8))
    return 100.0 * sum(WEIGHTS[k] * s[k] for k in WEIGHTS)

def zone(score):
    return "GREEN" if score >= 75 else "YELLOW" if score >= 55 else "RED"


# ----------------------------------------------------------------------------------------------
# 3. Per-station aggregation from the joint motion log
# ----------------------------------------------------------------------------------------------
def _rms(v): return math.sqrt(sum(x * x for x in v) / len(v)) if v else 0.0
def _mean(v): return sum(v) / len(v) if v else 0.0

def station_arm_stats(motion_log):
    by = {}
    for r in motion_log:
        by.setdefault(r["target_id"], []).append(r)
    out = {}
    for tid, rows in by.items():
        scan = [r for r in rows if r["phase"] in ("scan", "rescan")] or rows
        n = len(scan)
        e_after = _rms([r["theta1_error_deg"] for r in scan] + [r["theta2_error_deg"] for r in scan])
        e_before = _rms([r["uncompensated_theta1_error_deg"] for r in scan] +
                        [r["uncompensated_theta2_error_deg"] for r in scan])
        out[tid] = dict(
            joint1_deg=_mean([r["theta1_actual_deg"] for r in scan]),
            joint2_deg=_mean([r["theta2_actual_deg"] for r in scan]),
            joint1_vel_dps=_mean([abs(a["theta1_actual_deg"] - b["theta1_actual_deg"]) * 24
                                  for a, b in zip(scan[1:], scan[:-1])]),
            joint2_vel_dps=_mean([abs(a["theta2_actual_deg"] - b["theta2_actual_deg"]) * 24
                                  for a, b in zip(scan[1:], scan[:-1])]),
            tau1_nm=_mean([r["compensation_joint1"] for r in scan]),
            tau2_nm=_mean([r["compensation_joint2"] for r in scan]),
            tau_max_nm=max(max(abs(r["compensation_joint1"]), abs(r["compensation_joint2"])) for r in scan),
            saturation_pct=100.0 * sum(1 for r in scan if abs(r["compensation_joint1"]) >= 2.499 or
                                       abs(r["compensation_joint2"]) >= 2.499) / n,
            disturbance_rms=_rms([r["disturbance_joint1"] for r in scan] + [r["disturbance_joint2"] for r in scan]),
            camera_distance_m=_mean([r["camera_target_distance"] for r in scan]),
            viewing_angle_deg=_mean([r["camera_viewing_angle"] for r in scan]),
            in_range_pct=100.0 * sum(1 for r in scan if r["inside_camera_range"]) / n,
            in_fov_pct=100.0 * sum(1 for r in scan if r["inside_camera_fov"]) / n,
            stability_pct=100.0 * sum(1 for r in scan if r["camera_stable"]) / n,
            scan_valid_pct=100.0 * sum(1 for r in scan if r["scan_valid"]) / n,
            pos_err_m=_mean([r["position_error"] for r in scan]),
            err_before_deg=e_before, err_after_deg=e_after,
            err_reduction_pct=100.0 * (e_before - e_after) / e_before if e_before > 1e-12 else 0.0,
            frames=len(rows),
        )
    return out


# ----------------------------------------------------------------------------------------------
# 4. Classification metrics
# ----------------------------------------------------------------------------------------------
def confusion_and_metrics(rows):
    idx = {c: i for i, c in enumerate(CLASSES)}
    cm = [[0] * 7 for _ in range(7)]
    for r in rows:
        if r["predicted_stage"] in idx and r["true_stage"] in idx:
            cm[idx[r["true_stage"]]][idx[r["predicted_stage"]]] += 1
    per = []
    for i, c in enumerate(CLASSES):
        tp = cm[i][i]; fp = sum(cm[j][i] for j in range(7)) - tp; fn = sum(cm[i]) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        per.append(dict(cls=c, precision=p, recall=rc, f1=f1, support=sum(cm[i])))
    total = sum(map(sum, cm)); correct = sum(cm[i][i] for i in range(7))
    present = [m for m in per if m["support"] > 0]
    macro = lambda k: _mean([m[k] for m in present])
    bin_ok = sum(1 for r in rows if (r["true_stage"] in HEALTHY) == (r["predicted_stage"] in HEALTHY))
    return cm, per, dict(total=total, correct=correct, accuracy=100.0 * correct / total if total else 0.0,
                         macro_precision=macro("precision"), macro_recall=macro("recall"), macro_f1=macro("f1"),
                         healthy_vs_unhealthy_accuracy=100.0 * bin_ok / len(rows) if rows else 0.0)


# ----------------------------------------------------------------------------------------------
# 5. Pure-python SVG charts (no matplotlib needed)
# ----------------------------------------------------------------------------------------------
def svg_confusion(cm, path):
    cell, left, top = 62, 110, 80
    W, H = left + 7 * cell + 20, top + 7 * cell + 40
    mx = max(1, max(map(max, cm)))
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Arial">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="{W/2}" y="26" text-anchor="middle" font-size="18" font-weight="bold">Confusion matrix (rows = true, cols = predicted)</text>']
    for i, c in enumerate(CLASSES):
        s.append(f'<text x="{left-8}" y="{top+i*cell+cell/2+4}" text-anchor="end" font-size="12">{c}</text>')
        s.append(f'<text x="{left+i*cell+cell/2}" y="{top-10}" text-anchor="middle" font-size="12" transform="rotate(-25 {left+i*cell+cell/2} {top-10})">{c}</text>')
        for j in range(7):
            v = cm[i][j]; a = v / mx
            fill = f"rgb({int(255-a*215)},{int(255-a*155)},255)" if i != j else f"rgb({int(255-a*215)},255,{int(255-a*215)})"
            s.append(f'<rect x="{left+j*cell}" y="{top+i*cell}" width="{cell}" height="{cell}" fill="{fill}" stroke="#999"/>')
            if v: s.append(f'<text x="{left+j*cell+cell/2}" y="{top+i*cell+cell/2+5}" text-anchor="middle" font-size="16">{v}</text>')
    s.append('</svg>')
    Path(path).write_text("\n".join(s), encoding="utf-8")

ZCOL = {"GREEN": "#2ca02c", "YELLOW": "#e6b800", "RED": "#d62728"}

def svg_station_map(rows, path):
    xs = [r["egg_x"] for r in rows]; ys = [r["egg_y"] for r in rows]
    if not xs: return
    W, H, m = 900, 700, 60
    x0, x1, y0, y1 = min(xs) - 5, max(xs) + 5, min(ys) - 5, max(ys) + 5
    sx = lambda x: m + (x - x0) / max(1e-9, x1 - x0) * (W - 2 * m)
    sy = lambda y: H - m - (y - y0) / max(1e-9, y1 - y0) * (H - 2 * m)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Arial">',
         '<rect width="100%" height="100%" fill="#eef6fb"/>',
         f'<text x="{W/2}" y="28" text-anchor="middle" font-size="18" font-weight="bold">Mission map - visit order and habitat zone</text>',
         '<polyline fill="none" stroke="#555" stroke-width="1.5" stroke-dasharray="5,4" points="' +
         " ".join(f"{sx(r['egg_x']):.1f},{sy(r['egg_y']):.1f}" for r in rows) + '"/>']
    for r in rows:
        cx, cy = sx(r["egg_x"]), sy(r["egg_y"])
        ok = "" if r["correct"] else ' stroke="black" stroke-width="3"'
        s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="15" fill="{ZCOL[r["zone"]]}"{ok}/>')
        s.append(f'<text x="{cx:.1f}" y="{cy+4:.1f}" text-anchor="middle" font-size="11" fill="white">{r["station"]}</text>')
        s.append(f'<text x="{cx:.1f}" y="{cy+30:.1f}" text-anchor="middle" font-size="10">{html.escape(r["predicted_stage"])}</text>')
    s.append(f'<text x="{m}" y="{H-14}" font-size="12">Green/Yellow/Red = habitat zone. Black ring = misclassified. Dashed line = tour order.</text></svg>')
    Path(path).write_text("\n".join(s), encoding="utf-8")

def svg_bars(labels, values, title, path, colors=None, unit=""):
    W, H, l, t = 900, 420, 70, 60
    mx = max(values + [1e-9]); bw = (W - l - 30) / max(1, len(values))
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Arial">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="{W/2}" y="30" text-anchor="middle" font-size="18" font-weight="bold">{html.escape(title)}</text>']
    for i, (lab, v) in enumerate(zip(labels, values)):
        h = v / mx * (H - t - 70); x = l + i * bw
        s.append(f'<rect x="{x+4:.1f}" y="{H-50-h:.1f}" width="{bw-8:.1f}" height="{h:.1f}" fill="{(colors or ["#1f77b4"]*len(values))[i]}"/>')
        s.append(f'<text x="{x+bw/2:.1f}" y="{H-50-h-5:.1f}" text-anchor="middle" font-size="11">{v:.1f}{unit}</text>')
        s.append(f'<text x="{x+bw/2:.1f}" y="{H-30}" text-anchor="middle" font-size="11">{html.escape(str(lab))}</text>')
    s.append('</svg>')
    Path(path).write_text("\n".join(s), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# 6. Matplotlib graphs (replaces the stub plot_arm_analysis in the main script)
# ----------------------------------------------------------------------------------------------
def plot_arm_graphs(log, out_dir):
    """Writes the 7 PNG graphs + 6-panel figure. Returns list of files. Silent no-op without matplotlib."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        print("[FINAL] matplotlib not available - skipping PNG graphs "
              "(install: <blender>/python/bin/python -m pip install matplotlib)")
        return []
    if not log: return []
    os.makedirs(out_dir, exist_ok=True)
    t = [r["time_seconds"] for r in log]
    col = lambda k: [r[k] for r in log]
    ph = col("phase")
    specs = {
        "joint1_angle_vs_time": ("Joint 1 (shoulder) angle", "deg",
            [("Desired", "theta1_desired_deg"), ("Actual (disturbed)", "theta1_actual_deg"), ("Compensated", "theta1_compensated_deg")]),
        "joint2_angle_vs_time": ("Joint 2 (elbow) angle", "deg",
            [("Desired", "theta2_desired_deg"), ("Actual (disturbed)", "theta2_actual_deg"), ("Compensated", "theta2_compensated_deg")]),
        "joint_angle_error_vs_time": ("Joint angle error (with control)", "deg",
            [("theta1 error", "theta1_error_deg"), ("theta2 error", "theta2_error_deg"),
             ("theta1 uncompensated", "uncompensated_theta1_error_deg"), ("theta2 uncompensated", "uncompensated_theta2_error_deg")]),
        "compensation_effort_vs_time": ("Joint compensation torque (limit +/-2.5 Nm)", "Nm",
            [("Joint 1", "compensation_joint1"), ("Joint 2", "compensation_joint2")]),
        "arm_position_error_vs_time": ("End-effector position error", "m", [("Error", "position_error")]),
        "camera_scan_range_vs_time": ("Camera-to-target distance", "m", [("Distance", "camera_target_distance")]),
        "camera_viewing_angle_vs_time": ("Camera viewing angle off target", "deg", [("Angle", "camera_viewing_angle")]),
    }
    hl = {"compensation_effort_vs_time": [2.5, -2.5], "camera_scan_range_vs_time": [0.25, 2.0],
          "camera_viewing_angle_vs_time": [33.0]}
    files = []
    def draw(ax, name):
        title, unit, series = specs[name]
        for lab, k in series: ax.plot(t, col(k), lw=1.1, label=lab)
        for h in hl.get(name, []): ax.axhline(h, color="k", ls="--", lw=0.8)
        ax.set_title(title); ax.set_xlabel("time (s)"); ax.set_ylabel(unit); ax.grid(alpha=.3); ax.legend(fontsize=7)
    for name in specs:
        fig, ax = plt.subplots(figsize=(10, 4.2)); draw(ax, name); fig.tight_layout()
        p = os.path.join(out_dir, name + ".png"); fig.savefig(p, dpi=130); fig.savefig(p[:-4] + ".svg"); plt.close(fig); files.append(p)
    fig, axs = plt.subplots(3, 2, figsize=(15, 11))
    for ax, name in zip(axs.flat, ["joint1_angle_vs_time", "joint2_angle_vs_time", "joint_angle_error_vs_time",
                                   "compensation_effort_vs_time", "arm_position_error_vs_time", "camera_scan_range_vs_time"]):
        draw(ax, name)
    fig.tight_layout(); p = os.path.join(out_dir, "robotic_arm_analysis.png"); fig.savefig(p, dpi=120)
    fig.savefig(p[:-4] + ".svg"); plt.close(fig); files.append(p)
    return files


# ----------------------------------------------------------------------------------------------
# 6a. LAGRANGIAN DYNAMICS of the 2-link arm  (computed from the logged motion, for dashboard + CSV)
#     tau = M(q) q'' + C(q,q') q' + G(q)          q1 = shoulder angle from horizontal, q2 = elbow (relative)
# ----------------------------------------------------------------------------------------------
LAG_DEFAULT = dict(m1=2.0, m2=1.2, g_eff=9.81 * 0.25)   # same masses / buoyancy-reduced gravity as the standout module


def lagrangian_terms(q1, q2, qd1, qd2, qdd1, qdd2, L1, L2, m1, m2, g):
    """Uniform-rod 2-link planar arm. Returns M(3 unique entries), C*qd (2), G (2), tau (2), KE, PE."""
    lc1, lc2 = L1 / 2.0, L2 / 2.0
    I1, I2 = m1 * L1 ** 2 / 12.0, m2 * L2 ** 2 / 12.0
    c2, s2 = math.cos(q2), math.sin(q2)
    M11 = m1 * lc1 ** 2 + m2 * (L1 ** 2 + lc2 ** 2 + 2 * L1 * lc2 * c2) + I1 + I2
    M12 = m2 * (lc2 ** 2 + L1 * lc2 * c2) + I2
    M22 = m2 * lc2 ** 2 + I2
    h = -m2 * L1 * lc2 * s2
    C1 = h * qd2 * qd1 + h * (qd1 + qd2) * qd2        # C = [[h qd2, h(qd1+qd2)], [-h qd1, 0]]
    C2 = -h * qd1 * qd1
    G1 = (m1 * lc1 + m2 * L1) * g * math.cos(q1) + m2 * lc2 * g * math.cos(q1 + q2)
    G2 = m2 * lc2 * g * math.cos(q1 + q2)
    tau1 = M11 * qdd1 + M12 * qdd2 + C1 + G1
    tau2 = M12 * qdd1 + M22 * qdd2 + C2 + G2
    ke = 0.5 * (M11 * qd1 * qd1 + 2 * M12 * qd1 * qd2 + M22 * qd2 * qd2)
    pe = (m1 * lc1 * math.sin(q1) + m2 * (L1 * math.sin(q1) + lc2 * math.sin(q1 + q2))) * g
    return M11, M12, M22, C1, C2, G1, G2, tau1, tau2, ke, pe


def compute_dynamics(motion_log, arm_lengths=(0.9, 0.75), params=None, dt=1.0 / 24.0, smooth=5):
    """Adds Lagrangian fields (M11.. KE, PE and joint velocities/accelerations) to every motion_log row, in place."""
    p = dict(LAG_DEFAULT); p.update(params or {})
    L1, L2 = arm_lengths
    n = len(motion_log)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and motion_log[j + 1]["target_id"] == motion_log[j]["target_id"] \
                and motion_log[j + 1]["frame"] == motion_log[j]["frame"] + 1:
            j += 1
        seg = motion_log[i:j + 1]; m = len(seg)
        q1 = [math.radians(r["theta1_actual_deg"]) for r in seg]
        q2 = [math.radians(r["theta2_actual_deg"]) for r in seg]
        def d(x):
            if len(x) < 2: return [0.0] * len(x)
            return [(x[min(k + 1, m - 1)] - x[max(k - 1, 0)]) / (dt * (min(k + 1, m - 1) - max(k - 1, 0))) for k in range(m)]
        def sm(x):
            h = smooth // 2
            return [sum(x[max(0, k - h):k + h + 1]) / len(x[max(0, k - h):k + h + 1]) for k in range(m)]
        v1, v2 = sm(d(q1)), sm(d(q2))
        a1, a2 = sm(d(v1)), sm(d(v2))
        for k, r in enumerate(seg):
            M11, M12, M22, C1, C2, G1, G2, t1, t2, ke, pe = lagrangian_terms(q1[k], q2[k], v1[k], v2[k], a1[k], a2[k],
                                                                           L1, L2, p["m1"], p["m2"], p["g_eff"])
            r.update(dyn_qd1_dps=math.degrees(v1[k]), dyn_qd2_dps=math.degrees(v2[k]),
                     dyn_qdd1=a1[k], dyn_qdd2=a2[k], dyn_M11=M11, dyn_M12=M12, dyn_M22=M22,
                     dyn_C1=C1, dyn_C2=C2, dyn_G1=G1, dyn_G2=G2, dyn_tau1=t1, dyn_tau2=t2, dyn_KE=ke, dyn_PE=pe)
        i = j + 1
    return p


def save_dynamics_csv(motion_log, path):
    keys = ["frame", "time_seconds", "target_id", "phase", "theta1_actual_deg", "theta2_actual_deg", "dyn_qd1_dps", "dyn_qd2_dps",
            "dyn_qdd1", "dyn_qdd2", "dyn_M11", "dyn_M12", "dyn_M22", "dyn_C1", "dyn_C2", "dyn_G1", "dyn_G2",
            "dyn_tau1", "dyn_tau2", "compensation_joint1", "compensation_joint2", "dyn_KE", "dyn_PE",
            "pid1_p", "pid1_i", "pid1_d", "pid2_p", "pid2_i", "pid2_d"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", restval=""); w.writeheader(); w.writerows(motion_log)


def save_boat_csv(boat_log, path):
    if not boat_log: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(boat_log[0].keys())); w.writeheader(); w.writerows(boat_log)


def summarize_navigation(boat_log, nav_log):
    """Mission-level boat / PID statistics."""
    if not boat_log:
        return {}
    sp = [b["speed"] for b in boat_log]
    nav = [v for v in (nav_log or {}).values()]
    park = [v["park_dist_m"] for v in nav]
    return dict(
        total_distance_m=round(sum((nav_log or {}).get(k, {}).get("travel_m", 0.0) for k in (nav_log or {})), 1),
        mean_speed_ms=round(_mean(sp), 3), max_speed_ms=round(max(sp), 3),
        peak_thrust_n=round(max(b["thrust"] for b in boat_log), 1),
        mean_thrust_n=round(_mean([b["thrust"] for b in boat_log]), 1),
        thrust_saturation_pct=round(100 * _mean([1.0 if b["saturated"] else 0.0 for b in boat_log]), 2),
        mean_park_error_m=round(_mean(park), 3) if park else None, max_park_error_m=round(max(park), 3) if park else None,
        mean_waypoints=round(_mean([v["n_waypoints"] for v in nav]), 1) if nav else None,
        waypoint_timeouts=sum(v.get("timeouts", 0) for v in nav),
        steps=len(boat_log), travel_time_s=round(len(boat_log) / 24.0, 1),
        rms_ff_n=round(_rms([math.hypot(b["ff_x"], b["ff_y"]) for b in boat_log]), 1),
        rms_p_n=round(_rms([math.hypot(b["p_x"], b["p_y"]) for b in boat_log]), 1),
        rms_i_n=round(_rms([math.hypot(b["i_x"], b["i_y"]) for b in boat_log]), 1),
        rms_d_n=round(_rms([math.hypot(b["d_x"], b["d_y"]) for b in boat_log]), 1))


def summarize_dynamics(motion_log):
    if not motion_log or "dyn_tau1" not in motion_log[0]:
        return {}
    r = lambda v, d=3: round(v, d)
    t1 = [m["dyn_tau1"] for m in motion_log]; t2 = [m["dyn_tau2"] for m in motion_log]
    a1 = [m["compensation_joint1"] for m in motion_log]; a2 = [m["compensation_joint2"] for m in motion_log]
    return dict(
        peak_tau_dyn_j1_nm=r(max(abs(x) for x in t1)), peak_tau_dyn_j2_nm=r(max(abs(x) for x in t2)),
        rms_tau_dyn_j1_nm=r(_rms(t1)), rms_tau_dyn_j2_nm=r(_rms(t2)),
        rms_gravity_j1_nm=r(_rms([m["dyn_G1"] for m in motion_log])), rms_gravity_j2_nm=r(_rms([m["dyn_G2"] for m in motion_log])),
        rms_coriolis_j1_nm=r(_rms([m["dyn_C1"] for m in motion_log]), 4), rms_coriolis_j2_nm=r(_rms([m["dyn_C2"] for m in motion_log]), 4),
        max_ke_j=r(max(m["dyn_KE"] for m in motion_log), 4), mean_pe_j=r(_mean([m["dyn_PE"] for m in motion_log]), 3),
        M11_range=[r(min(m["dyn_M11"] for m in motion_log)), r(max(m["dyn_M11"] for m in motion_log))],
        M12_range=[r(min(m["dyn_M12"] for m in motion_log)), r(max(m["dyn_M12"] for m in motion_log))],
        M22=r(motion_log[0]["dyn_M22"]),
        rms_pid_p_j1=r(_rms([m.get("pid1_p", 0) for m in motion_log])), rms_pid_i_j1=r(_rms([m.get("pid1_i", 0) for m in motion_log])), rms_pid_d_j1=r(_rms([m.get("pid1_d", 0) for m in motion_log])),
        rms_pid_p_j2=r(_rms([m.get("pid2_p", 0) for m in motion_log])), rms_pid_i_j2=r(_rms([m.get("pid2_i", 0) for m in motion_log])), rms_pid_d_j2=r(_rms([m.get("pid2_d", 0) for m in motion_log])),
        torque_saturation_pct=r(100 * _mean([1.0 if (abs(x) >= 2.499 or abs(y) >= 2.499) else 0.0 for x, y in zip(a1, a2)]), 2))


# ----------------------------------------------------------------------------------------------
# 6b. LIVE REPLAY DASHBOARD (single self-contained HTML, same look as autonomous_zebrafish_monitor)
# ----------------------------------------------------------------------------------------------
import base64

DEFAULT_PENS = [(0, 0), (26, 12), (-28, 15), (15, -25), (-18, -26), (45, -10)]
PHASE_IDX = {"approach": 1, "stabilization": 2, "scan": 3, "rescan": 3, "retraction": 5}


def sample_boat_trajectory(scene, robot_components, last_frame, stride=6):
    """Reads the boat position from the keyframed Blender animation. Returns [[frame,x,y],...] or None."""
    try:
        root = robot_components["master_root"]
        keep = scene.frame_current
        out = []
        for f in list(range(1, int(last_frame) + 1, stride)) + [int(last_frame)]:
            scene.frame_set(f)
            p = root.matrix_world.translation
            out.append([f, round(p.x, 3), round(p.y, 3)])
        scene.frame_set(keep)
        return out
    except Exception as exc:
        print("[FINAL] boat trajectory not sampled:", repr(exc))
        return None


def _data_uri(folder, name):
    if not name:
        return ""
    p = Path(folder) / name
    if not p.is_file():
        return ""
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def build_live_dashboard(path, rows, motion_log, scan_results, egg_data, mission, cm, per,
                         scans_dir, traj=None, pens=None, max_samples=3500, boat_log=None):
    n = len(motion_log)
    step = max(1, n // max_samples)
    keep, last_ph = [], None
    for i, m in enumerate(motion_log):
        if i % step == 0 or m["phase"] != last_ph or i == n - 1:
            keep.append(m)
        last_ph = m["phase"]
    r2 = lambda v, d=3: round(float(v), d)
    motion = [dict(f=m["frame"], id=m["target_id"], ph=PHASE_IDX.get(m["phase"], 0),
                   a1=r2(m["theta1_actual_deg"], 2), a2=r2(m["theta2_actual_deg"], 2),
                   c1=r2(m["theta1_compensated_deg"], 2), c2=r2(m["theta2_compensated_deg"], 2),
                   d1=r2(m["theta1_desired_deg"], 2), d2=r2(m["theta2_desired_deg"], 2),
                   e1=r2(m["theta1_error_deg"], 3), e2=r2(m["theta2_error_deg"], 3),
                   u1=r2(m["uncompensated_theta1_error_deg"], 3), u2=r2(m["uncompensated_theta2_error_deg"], 3),
                   pe=r2(m["position_error"], 4), t1=r2(m["compensation_joint1"], 3), t2=r2(m["compensation_joint2"], 3),
                   w1=r2(m["disturbance_joint1"], 4), w2=r2(m["disturbance_joint2"], 4),
                   cd=r2(m["camera_target_distance"], 3), va=r2(m["camera_viewing_angle"], 2),
                   st=int(bool(m["camera_stable"])), sv=int(bool(m["scan_valid"])),
                   V1=r2(m.get("dyn_qd1_dps", 0), 2), V2=r2(m.get("dyn_qd2_dps", 0), 2),
                   M11=r2(m.get("dyn_M11", 0), 4), M12=r2(m.get("dyn_M12", 0), 4), M22=r2(m.get("dyn_M22", 0), 4),
                   C1=r2(m.get("dyn_C1", 0), 5), C2=r2(m.get("dyn_C2", 0), 5), G1=r2(m.get("dyn_G1", 0), 4), G2=r2(m.get("dyn_G2", 0), 4),
                   T1=r2(m.get("dyn_tau1", 0), 4), T2=r2(m.get("dyn_tau2", 0), 4), KE=r2(m.get("dyn_KE", 0), 5), PE=r2(m.get("dyn_PE", 0), 4),
                   A1=r2(m.get("dyn_qdd1", 0), 3), A2=r2(m.get("dyn_qdd2", 0), 3),
                   P1=r2(m.get("pid1_p", 0), 4), I1=r2(m.get("pid1_i", 0), 4), D1=r2(m.get("pid1_d", 0), 4),
                   P2=r2(m.get("pid2_p", 0), 4), I2=r2(m.get("pid2_i", 0), 4), D2=r2(m.get("pid2_d", 0), 4))
              for m in keep]

    boat = []
    if boat_log:
        bstep = max(1, len(boat_log) // 2500)
        for i, b in enumerate(boat_log):
            if i % bstep == 0 or i == len(boat_log) - 1 or i > 0 and b["wp_idx"] != boat_log[i - 1]["wp_idx"]:
                boat.append(dict(f=b["frame"], id=b["target_id"], wi=b["wp_idx"], nw=b["n_wp"], x=r2(b["x"], 3), y=r2(b["y"], 3),
                                 sp=r2(b["speed"], 3), we=r2(b["wp_err"], 3), ee=r2(b["egg_err"], 3), th=r2(b["thrust"], 1),
                                 tx=r2(b["thrust_x"], 1), ty=r2(b["thrust_y"], 1),
                                 P=r2(math.hypot(b["p_x"], b["p_y"]), 1), I=r2(math.hypot(b["i_x"], b["i_y"]), 2),
                                 Dm=r2(math.hypot(b["d_x"], b["d_y"]), 1), FF=r2(math.hypot(b["ff_x"], b["ff_y"]), 1),
                                 W=r2(math.hypot(b["dist_x"], b["dist_y"]), 1), sat=int(bool(b["saturated"]))))

    span = {}
    for m in motion_log:
        s = span.setdefault(m["target_id"], [m["frame"], m["frame"]])
        s[0] = min(s[0], m["frame"]); s[1] = max(s[1], m["frame"])
    last_frame = max([m["frame"] for m in motion_log] + [r["frame"] for r in scan_results] + [1])

    stations, prev_end = [], 1
    for r, sr in zip(rows, scan_results):
        a, b = span.get(r["target_id"], [sr["frame"], sr["frame"]])
        s = {k: r.get(k) for k in ("station", "target_id", "egg_x", "egg_y", "true_stage", "predicted_stage",
             "confidence_pct", "correct", "reachable", "rescanned", "zone", "habitat_score", "temperature_C", "pH",
             "oxygen_mgL", "turbidity_NTU", "ammonia_mgL", "nitrate_mgL", "conductivity_uScm", "light", "flow_ms",
             "joint1_deg", "joint2_deg", "tau1_nm", "tau2_nm", "camera_distance_m", "viewing_angle_deg",
             "stability_pct", "scan_valid_pct", "err_before_deg", "err_after_deg", "err_reduction_pct",
             "nav_n_waypoints", "nav_path_length_m", "nav_travel_m", "nav_travel_frames", "nav_timeouts", "nav_park_dist_m",
             "nav_reach_d_m", "nav_reach_min_m", "nav_reach_max_m", "nav_ik_yaw_deg", "nav_ik_shoulder_deg", "nav_ik_elbow_deg",
             "nav_ik_reachable", "nav_park_x", "nav_park_y")}
        s.update(id=r["target_id"], start=a, scan=min(max(sr["frame"], a), b) if r["reachable"] else sr["frame"],
                 end=b, live=_data_uri(scans_dir, r.get("live_image")), ds=_data_uri(scans_dir, r.get("scan_image")))
        stations.append(s)

    if not traj and boat_log:   # rebuild from the boat log: moving frames + parked while the arm works
        pts = [[b["frame"], round(b["x"], 3), round(b["y"], 3)] for b in boat_log[::max(1, len(boat_log) // 1500)]]
        for s in stations:
            if s.get("nav_park_x") is not None:
                pts += [[s["start"], s["nav_park_x"], s["nav_park_y"]], [s["end"], s["nav_park_x"], s["nav_park_y"]]]
        traj = sorted(pts, key=lambda p: p[0]) or None
    if not traj:  # last resort: boat glides between station positions
        traj = [[1, 0.0, 0.0]] + [[s["start"], s["egg_x"], s["egg_y"]] for s in stations] + [[last_frame, stations[-1]["egg_x"], stations[-1]["egg_y"]]]

    data = dict(classes=CLASSES, stations=stations, motion=motion, boat=boat, traj=traj, last_frame=last_frame,
                pens=[list(p) for p in (pens or DEFAULT_PENS)], pen_r=12.0, fps=24,
                mission=mission, confusion=cm, per_class=per)
    blob = json.dumps(data, default=str).replace("</", "<\\/")
    Path(path).write_text(LIVE_TEMPLATE.replace("__DATA__", blob), encoding="utf-8")
    return path


LIVE_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aquatic Zebrafish Egg Monitoring Robot</title>
<style>
:root{--bg:#080f17;--p:#0f1923;--b:#1d3145;--t:#c9d8e6;--m:#7f97ad;--a:#38bdf8;--g:#34d399;--y:#fbbf24;--r:#f87171}
*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--t);font:12px/1.45 "Segoe UI",system-ui,sans-serif}
header{height:36px;display:flex;align-items:center;gap:14px;padding:0 14px;background:#0b141e;border-bottom:1px solid var(--b)}
header h1{margin:0;font-size:13px;font-weight:600;letter-spacing:.14em;text-transform:uppercase}
header span{color:var(--m);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#main{display:grid;grid-template-columns:210px 1fr 330px;height:calc(100vh - 36px - 78px);min-height:520px}
.side{background:var(--p);overflow-y:auto;padding:10px 12px}
#left{border-right:1px solid var(--b)}#right{border-left:1px solid var(--b)}
h3{margin:12px 0 5px;font-size:10px;letter-spacing:.16em;color:var(--a);text-transform:uppercase;font-weight:600;border-bottom:1px solid var(--b);padding-bottom:3px}
h3:first-child{margin-top:0}
.kv{display:flex;justify-content:space-between;gap:8px;padding:1px 0}.kv span{color:var(--m)}.kv b{font-weight:600;font-variant-numeric:tabular-nums;text-align:right}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#33475a;margin-right:7px}.dot.on{background:var(--g);box-shadow:0 0 6px var(--g)}.dot.warn.on{background:var(--y);box-shadow:0 0 6px var(--y)}.dot.cam.on{background:var(--a);box-shadow:0 0 6px var(--a)}
.st{padding:2px 0;color:var(--m)}.st.on{color:var(--t)}
.wf{padding:2px 6px;margin:1px 0;border-left:2px solid var(--b);color:var(--m);font-size:11px}.wf.on{border-color:var(--a);color:var(--t);background:#122638}.wf.done{color:#5f7a92}
#vp{position:relative;overflow:hidden;background:#07111b}#c{width:100%;height:100%;display:block}
#badge{position:absolute;left:12px;top:10px;background:rgba(8,15,23,.78);border:1px solid var(--b);padding:6px 10px;border-radius:4px}
#badge big{display:block;font-size:17px;font-weight:600;letter-spacing:.06em}
#legend{position:absolute;right:10px;top:10px;background:rgba(8,15,23,.78);border:1px solid var(--b);padding:6px 10px;border-radius:4px;color:var(--m);font-size:11px}
#legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 8px}
#msg{position:absolute;left:0;right:0;bottom:14px;text-align:center;font-size:15px;letter-spacing:.2em;color:var(--a);text-shadow:0 0 8px #000;pointer-events:none}
#banner{position:absolute;inset:0;display:none;align-items:center;justify-content:center;font-size:34px;letter-spacing:.3em;color:var(--g);background:rgba(5,12,18,.45);pointer-events:none}
.ov{position:absolute;background:rgba(8,15,23,.86);border:1px solid var(--b);border-radius:4px;padding:5px}
.ov small{display:block;color:var(--m);font-size:9px;letter-spacing:.12em;margin-bottom:3px}
#armBox{right:10px;bottom:44px}#camBox{left:10px;bottom:44px;width:150px}
#camBox img{width:138px;height:138px;display:block;border-radius:3px;background:#0b141e;object-fit:cover}
#flash{position:absolute;inset:0;background:#fff;opacity:0;pointer-events:none}
button{background:#122638;color:var(--t);border:1px solid #27425b;padding:5px 11px;border-radius:3px;font:600 11px "Segoe UI",system-ui,sans-serif;letter-spacing:.06em;cursor:pointer}
button:hover{background:#1a3650}button.act{background:#0e4f6e;border-color:var(--a)}button:disabled{opacity:.4;cursor:default}
#ctl{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;padding:8px 14px;background:#0b141e;border-top:1px solid var(--b);border-bottom:1px solid var(--b);min-height:78px}
#ctl .g{display:flex;gap:5px;align-items:center}#ctl label{color:var(--m);font-size:10px;letter-spacing:.12em;margin-right:3px}
#prog{flex:1 1 100%;height:6px;background:#15293b;border-radius:3px;overflow:hidden;cursor:pointer}#prog i{display:block;height:100%;width:0;background:var(--a)}
details{margin:0;border-bottom:1px solid var(--b);background:var(--p)}summary{padding:8px 14px;cursor:pointer;letter-spacing:.16em;color:var(--a);font-weight:600;font-size:11px}
.g4{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:10px;padding:10px 14px}.cw{position:relative;height:215px;background:#0b141e;border:1px solid var(--b);border-radius:4px;padding:6px}
.cw h4{margin:0 0 2px;font-size:10px;letter-spacing:.12em;color:var(--m);font-weight:600}.cw div{position:relative;height:calc(100% - 16px)}
#sum{padding:16px 20px;background:var(--p)}#sum h2{margin:0 0 10px;letter-spacing:.2em;color:var(--g);font-size:16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px;margin-bottom:14px}.card{background:#0b141e;border:1px solid var(--b);padding:8px 10px;border-radius:4px}.card small{display:block;color:var(--m);font-size:10px;letter-spacing:.08em}.card b{font-size:18px}
table{border-collapse:collapse;margin:6px 0}th,td{border:1px solid var(--b);padding:4px 9px;text-align:center;font-variant-numeric:tabular-nums}th{background:#0b141e;color:var(--m);font-weight:600}
td img{width:46px;height:46px;object-fit:cover;border-radius:3px;vertical-align:middle;cursor:pointer}
.ok{color:var(--g)}.bad{color:var(--r)}.zt{padding:1px 8px;border-radius:9px;color:#06121c;font-weight:600}
.tw{overflow-x:auto}
.eqs{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:10px;padding:10px 14px}
.eq{background:#0b141e;border:1px solid var(--b);border-radius:4px;padding:8px 10px}
.eq h4{margin:0 0 4px;font-size:10px;letter-spacing:.12em;color:var(--a);font-weight:600}
.eq .f{font:italic 12px/1.5 Georgia,'Times New Roman',serif;color:#e2edf7;background:#0f1923;border-left:2px solid var(--a);padding:4px 8px;margin-bottom:5px}
.eq pre{margin:0;font:11px/1.5 Consolas,'Courier New',monospace;color:#b9ccdc;white-space:pre-wrap}
.eq pre b{color:var(--g)}.eq pre u{color:var(--y);text-decoration:none}
.sub{color:var(--m);font-size:10px;margin:6px 0 2px;letter-spacing:.1em}
.lk{margin-left:auto;font-size:11px;font-weight:600;letter-spacing:.1em;color:var(--m);white-space:nowrap}.lk.on{color:var(--g)}.lk.on.pl{color:var(--a)}
.note{color:var(--m);font-size:10px;margin:2px 0 4px}
</style></head><body>
<header><h1>Aquatic Zebrafish Egg Monitoring Robot</h1><span>Blender simulation replay · autonomous surface vehicle · 2-DOF underwater arm · AI classification · water-quality &amp; habitat monitoring</span><b id="lk" class="lk">○ STANDALONE REPLAY</b></header>
<div id="main">
<div id="left" class="side"></div>
<div id="vp"><canvas id="c"></canvas><div id="badge"></div>
<div id="legend"><i style="background:#34d399"></i>GREEN<i style="background:#fbbf24"></i>YELLOW<i style="background:#f87171"></i>RED<i style="background:#456"></i>unscanned</div>
<div class="ov" id="camBox" hidden><small>ROBOT CAMERA</small><img id="camImg" alt=""><small id="camCap" style="margin:3px 0 0"></small></div>
<div class="ov" id="armBox"><small>ARM SIDE VIEW (underwater)</small><canvas id="arm" width="230" height="170"></canvas></div>
<div id="msg"></div><div id="flash"></div><div id="banner">MISSION COMPLETE</div></div>
<div id="right" class="side"></div>
</div>
<div id="ctl">
<div class="g"><button id="bStart">START REPLAY</button><button id="bPause">PAUSE</button><button id="bReset">RESET</button><button id="bNext">NEXT STATION</button></div>
<div class="g"><label>SPEED</label><button id="sp0">SLOW</button><button id="sp1" class="act">NORMAL</button><button id="sp2">FAST</button><button id="sp3">MAX</button></div>
<div class="g"><label>UNCOMPENSATED OVERLAY</label><button id="bComp" class="act">ON</button></div>
<div class="g"><button id="bCsv">DOWNLOAD CSV</button><button id="bZip">DOWNLOAD IMAGES ZIP</button><button id="bAll">DOWNLOAD ALL RESULTS</button></div>
<div id="prog" title="click to seek"><i></i></div>
</div>
<details open><summary>ANALYTICS (live)</summary>
<div class="g4">
<div class="cw"><h4>ARM JOINT ANGLES (deg) — ACTUAL vs DESIRED</h4><div><canvas id="ch1"></canvas></div></div>
<div class="cw"><h4>JOINT ANGLE ERROR (deg RMS) — BEFORE vs AFTER COMPENSATION</h4><div><canvas id="ch2"></canvas></div></div>
<div class="cw"><h4>END-EFFECTOR POSITION ERROR (m) — TOLERANCE 0.12 m</h4><div><canvas id="ch3"></canvas></div></div>
<div class="cw"><h4>JOINT 1 (SHOULDER) TORQUE (Nm) — DISTURBANCE vs COMPENSATION</h4><div><canvas id="ch4"></canvas></div></div>
<div class="cw"><h4>JOINT 2 (ELBOW) TORQUE (Nm) — DISTURBANCE vs COMPENSATION</h4><div><canvas id="ch5"></canvas></div></div>
<div class="cw"><h4>CAMERA-TO-TARGET DISTANCE (m) — VALID 0.25–2.0 m</h4><div><canvas id="ch6"></canvas></div></div>
<div class="cw"><h4>CLASSIFICATION CONFIDENCE (%) PER STATION</h4><div><canvas id="ch7"></canvas></div></div>
<div class="cw"><h4>HABITAT SCORE PER STATION</h4><div><canvas id="ch8"></canvas></div></div>
</div></details>
<details open><summary>CONTROL LAWS &amp; DYNAMICS (live equations and signals)</summary>
<div class="eqs">
<div class="eq"><h4>BOAT NAVIGATION — A* PATH + POSITION PID + CURRENT FEEDFORWARD</h4>
<div class="f">F = K<sub>p</sub>·e + K<sub>i</sub>·∫e dt + K<sub>d</sub>·de/dt + F<sub>ff</sub> &nbsp;,&nbsp; F<sub>ff</sub> = −k·F<sub>current</sub><br>m·a = F + F<sub>current</sub> + F<sub>wobble</sub> − c·|v|·v</div><pre id="eqBoat"></pre></div>
<div class="eq"><h4>ARM JOINT CONTROL — PID WITH ANTI-WINDUP</h4>
<div class="f">τ<sub>j</sub> = sat( K<sub>p</sub>·e<sub>j</sub> + K<sub>i</sub>·∫e<sub>j</sub> dt + K<sub>d</sub>·de<sub>j</sub>/dt ) &nbsp;,&nbsp; e = q<sub>des</sub> − q<br>q̈ = ( τ + τ<sub>dist</sub> − b·q̇ − k·(q − q<sub>des</sub>) ) / J</div><pre id="eqArm"></pre></div>
<div class="eq"><h4>INVERSE KINEMATICS — 2-LINK ARM</h4>
<div class="f">ψ = atan2(y,x) , &nbsp; d = √(r² + z²) , &nbsp; |L₁−L₂| ≤ d ≤ L₁+L₂<br>cos q₂ = (d² − L₁² − L₂²)/(2 L₁ L₂) , &nbsp; q₁ = atan2(z,r) + acos((L₁² + d² − L₂²)/(2 L₁ d))</div><pre id="eqIK"></pre></div>
<div class="eq"><h4>LAGRANGIAN DYNAMICS — INVERSE DYNAMICS OF THE MEASURED MOTION</h4>
<div class="f">τ = M(q)·q̈ + C(q,q̇)·q̇ + G(q) &nbsp;,&nbsp; L = T − V , &nbsp; T = ½ q̇ᵀ M q̇<br>M₁₁ = m₁lc₁² + m₂(L₁² + lc₂² + 2L₁lc₂cos q₂) + I₁ + I₂ , &nbsp; M₁₂ = m₂(lc₂² + L₁lc₂cos q₂) + I₂ , &nbsp; M₂₂ = m₂lc₂² + I₂<br>h = −m₂L₁lc₂ sin q₂ , &nbsp; C = [[h q̇₂ , h(q̇₁+q̇₂)],[−h q̇₁ , 0]] , &nbsp; G = g<sub>eff</sub>·[(m₁lc₁+m₂L₁)cos q₁ + m₂lc₂cos(q₁+q₂) , m₂lc₂cos(q₁+q₂)]</div><pre id="eqLag"></pre></div>
<div class="eq"><h4>LAGRANGE–EULER WITH HYDRODYNAMICS — FULL FORM (matches the Blender arm model)</h4>
<div class="f">M(q)·q̈ + C(q,q̇)·q̇ + G(q) + D·q̇ = τ + τ<sub>d</sub> &nbsp;,&nbsp; L = T − V<br>T = ½ q̇ᵀ M(q) q̇ &nbsp;,&nbsp; V = g<sub>net</sub> Σ mᵢ·y<sub>ci</sub> &nbsp;,&nbsp; q̈ = M⁻¹( τ + τ<sub>d</sub> − C·q̇ − G − D·q̇ )<br>Blender joint sim: J·q̈ = τ + τ<sub>d</sub> − b·q̇ − k·(q − q<sub>des</sub>) &nbsp;,&nbsp; D = b = joint damping<br>τ<sub>d1</sub> = A·sin(0.75t) + c·sin(0.45t+0.4) + n(t) &nbsp;,&nbsp; τ<sub>d2</sub> = 0.8A·cos(0.95t+0.7) + c·cos(0.55t) + n(t) &nbsp;(water current + turbulence)</div><pre id="eqLE"></pre></div>
<div class="eq"><h4>ADAPTIVE NEURAL COMPENSATOR + LYAPUNOV — DESIGN REFERENCE (v5 simulator)</h4>
<div class="f">τ = τ<sub>PD</sub> + τ<sub>nn</sub> &nbsp;,&nbsp; τ<sub>nn</sub> = Wᵀ·tanh(Vᵀ x) &nbsp;,&nbsp; x = [sin θ, cos θ, sin 2θ, cos 2θ, cos q₁, cos(q₁+q₂), sin(q₁+q₂), 1]<br>filtered error &nbsp;r = ė + Λ·e &nbsp;,&nbsp; e = q<sub>des</sub> − q<br>Ẇ = F·σ·rᵀ − κ‖r‖·F·W &nbsp;,&nbsp; V̇ = G·x·(σ′ᵒ W r)ᵀ − κ‖r‖·G·V &nbsp;(e-modification keeps the weights bounded)<br>Lyapunov candidate &nbsp;L = ½ rᵀ M r + ½ tr(W̃ᵀF⁻¹W̃) + ½ tr(Ṽᵀ G⁻¹ Ṽ)</div>
<div class="note">Network 8 → 14 → 3 (tanh), F = 0.5, G = 0.17, κ = 0.05, Λ = 4 s⁻¹. The Blender run compensates with the PID above (no NN) — r and ½rᵀMr below are evaluated on the logged Blender motion while the arm holds position.</div><pre id="eqNN"></pre></div>
</div>
<div class="g4">
<div class="cw"><h4>BOAT — DISTANCE TO WAYPOINT / TO EGG (m)</h4><div><canvas id="cb1"></canvas></div></div>
<div class="cw"><h4>BOAT PID — THRUST TERMS (N)</h4><div><canvas id="cb2"></canvas></div></div>
<div class="cw"><h4>BOAT — SPEED (m/s)</h4><div><canvas id="cb3"></canvas></div></div>
<div class="cw"><h4>JOINT 1 PID TERMS (Nm): P · I · D</h4><div><canvas id="ca1"></canvas></div></div>
<div class="cw"><h4>JOINT 2 PID TERMS (Nm): P · I · D</h4><div><canvas id="ca2"></canvas></div></div>
<div class="cw"><h4>JOINT 1 — LAGRANGIAN TORQUE vs APPLIED (Nm)</h4><div><canvas id="ca3"></canvas></div></div>
<div class="cw"><h4>JOINT 2 — LAGRANGIAN TORQUE vs APPLIED (Nm)</h4><div><canvas id="ca4"></canvas></div></div>
<div class="cw"><h4>ARM ENERGY (J): KINETIC T · POTENTIAL V</h4><div><canvas id="ca5"></canvas></div></div>
<div class="cw"><h4>INERTIA MATRIX M(q) (kg·m²)</h4><div><canvas id="ca6"></canvas></div></div>
</div></details>
<section id="sum" hidden></section>

<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
"use strict";
const D=__DATA__;
const ST=D.stations,M=D.motion,B=D.boat||[],TR=D.traj,CL=D.classes,NS=ST.length,FPS=D.fps||24;
const ZC={GREEN:'#34d399',YELLOW:'#fbbf24',RED:'#f87171'};
const LABEL=['MOVING','ARM UNFOLDING','STABILIZING','SCANNING','CAPTURED · ANALYZING','ARM FOLDING'];
const WFN=['Robot moves to station (A* + PID)','Arm unfolds underwater · camera targeting','Disturbance · compensation · stabilization','Camera valid · scanning','Image capture · AI classification · habitat','Arm folds up → next station'];
const WK=[['temperature_C','Temperature','°C',1],['pH','pH','',2],['oxygen_mgL','Oxygen','mg/L',2],['turbidity_NTU','Turbidity','NTU',2],['ammonia_mgL','Ammonia','mg/L',4],['nitrate_mgL','Nitrate','mg/L',1],['conductivity_uScm','Conductivity','µS/cm',0],['light','Light','',2],['flow_ms','Flow','m/s',3]];
const L1=.9,L2=.75;
const $=id=>document.getElementById(id),set=(id,v)=>{const e=$(id);if(e&&e.textContent!==v)e.textContent=v};
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v)),f=(v,n)=>(v==null||isNaN(v))?'—':(+v).toFixed(n),sg=(v,n)=>(v>=0?'+':'')+v.toFixed(n);
const kv=(l,id)=>`<div class=kv><span>${l}</span><b id=${id}>—</b></div>`,kf=(l,v)=>`<div class=kv><span>${l}</span><b>${v}</b></div>`;
const mis=D.mission||{},ar=mis.arm||{},cam=mis.camera||{},nv=mis.navigation||{},dy=mis.dynamics||{},PR=mis.params||{},BG=PR.boat_gains||[],AG=PR.arm_gains||[],AL=PR.arm_lengths||[.9,.75],LG=PR.lagrangian||{m1:2,m2:1.2,g_eff:2.4525};
$('left').innerHTML=`<h3>System status</h3><div class=st id=s0><i class=dot></i>SYSTEM READY</div><div class=st id=s1><i class=dot></i>AUTONOMOUS SCANNING</div><div class=st id=s2><i class="dot cam"></i>UNDERWATER CAMERA ACTIVE</div><div class=st id=s3><i class="dot warn"></i>DISTURBANCE ACTIVE</div><div class=st id=s4><i class=dot></i>COMPENSATION ACTIVE</div>
<h3>Scan workflow</h3>${WFN.map((w,i)=>`<div class=wf id=wf${i}>${w}</div>`).join('')}
<h3>Platform</h3>${kf('Vehicle','Surface catamaran')}${kf('Arm','2-DOF, always submerged')}${kf('L1 / L2','0.90 / 0.75 m')}${kf('Camera FOV','60°')}${kf('Stations',NS)}${kf('Mission time',f(D.last_frame/FPS,1)+' s')}${kf('Re-scans',mis.rescans??0)}
<h3>Robot</h3>${kv('X','rx')}${kv('Y','ry')}${kv('Speed','rv')}${kv('Replay time','rt')}${kv('Blender frame','bfr')}${kv('Blender link','lnk')}`;
$('right').innerHTML=`<h3>Current station</h3>${kv('Station','stn')}${kv('Egg position','epos')}${kv('State','st')}
<h3>Latest classification</h3>${kv('Result from station','lat')}${kv('Egg stage (predicted)','pred')}${kv('True stage','tru')}${kv('Confidence','conf')}${kv('Classification','res')}${kv('Zone','zone')}${kv('Habitat score','hab')}
<h3>Camera</h3>${kf('Valid range','0.25–2.00 m')}${kv('Current distance','cdist')}${kv('Viewing angle','cang')}${kv('Camera stable','cstab')}${kv('Scan valid','cval')}
<h3>Underwater disturbance</h3>${kv('Disturbance','dis')}${kv('Compensation','cmp')}${kv('No compensation · error','ebv')}${kv('With compensation · error','eav')}${kv('Stability improvement','imp')}
<h3>Arm — joint 1 (shoulder)</h3>${kv('Angle q1','j1a')}${kv('Velocity','j1v')}${kv('Applied torque','j1t')}${kv('Disturbance torque','j1d')}
<h3>Arm — joint 2 (elbow)</h3>${kv('Angle q2','j2a')}${kv('Velocity','j2v')}${kv('Applied torque','j2t')}${kv('Disturbance torque','j2d')}
<h3>Water parameters</h3>${WK.map(w=>kv(w[1],'w_'+w[0])).join('')}`;

/* ---------- charts ---------- */
const gc={animation:false,responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#9fb4c8',boxWidth:10,font:{size:10}}}},scales:{x:{ticks:{color:'#7f97ad',maxTicksLimit:8},grid:{color:'#162636'}},y:{ticks:{color:'#7f97ad'},grid:{color:'#162636'}}}};
const ds=(l,c,dash)=>({label:l,data:[],borderColor:c,backgroundColor:c,borderWidth:1.5,pointRadius:0,tension:.2,borderDash:dash||[]});
const mk=(id,type,d,o)=>new Chart($(id),{type,data:{labels:[],datasets:d},options:Object.assign({},gc,o||{})});
const barOpt={scales:{x:gc.scales.x,y:{min:0,max:100,ticks:gc.scales.y.ticks,grid:gc.scales.y.grid}}};
const bar=l=>({label:l,data:[],backgroundColor:[]});
const CH=[mk('ch1','line',[ds('q1 actual','#38bdf8'),ds('q2 actual','#fbbf24'),ds('q1 desired','#38bdf8',[4,3]),ds('q2 desired','#fbbf24',[4,3])]),
 mk('ch2','line',[ds('Before compensation','#fb923c'),ds('After compensation','#34d399')]),
 mk('ch3','line',[ds('Position error','#38bdf8'),ds('Tolerance','#f87171',[5,4])]),
 mk('ch4','line',[ds('Disturbance','#fb923c'),ds('Compensation torque','#34d399')]),
 mk('ch5','line',[ds('Disturbance','#fb923c'),ds('Compensation torque','#34d399')]),
 mk('ch6','line',[ds('Distance','#38bdf8'),ds('Min 0.25','#f87171',[5,4]),ds('Max 2.0','#f87171',[5,4])]),
 mk('ch7','bar',[bar('Confidence %')],barOpt),mk('ch8','bar',[bar('Habitat score')],barOpt)];
const CHB=[mk('cb1','line',[ds('To waypoint','#38bdf8'),ds('To egg','#fbbf24'),ds('Final tolerance 0.5 m','#f87171',[5,4])]),
 mk('cb2','line',[ds('P = Kp·e','#38bdf8'),ds('I = Ki·∫e','#a78bfa'),ds('D = Kd·de/dt','#fbbf24'),ds('Feedforward','#fb923c'),ds('Total (clamped)','#34d399')]),
 mk('cb3','line',[ds('Speed','#34d399'),ds('Current force /100','#fb923c',[4,3])])];
const CHA=[mk('ca1','line',[ds('P','#38bdf8'),ds('I','#a78bfa'),ds('D','#fbbf24')]),mk('ca2','line',[ds('P','#38bdf8'),ds('I','#a78bfa'),ds('D','#fbbf24')]),
 mk('ca3','line',[ds('τ Lagrangian (required)','#fb923c'),ds('τ applied (PID)','#34d399'),ds('Gravity+buoyancy G','#7f97ad',[4,3])]),
 mk('ca4','line',[ds('τ Lagrangian (required)','#fb923c'),ds('τ applied (PID)','#34d399'),ds('Gravity+buoyancy G','#7f97ad',[4,3])]),
 mk('ca5','line',[ds('Kinetic T','#38bdf8'),ds('Potential V','#fbbf24')]),mk('ca6','line',[ds('M11','#38bdf8'),ds('M12','#a78bfa'),ds('M22','#fbbf24')])];
const LINES=[CH[0],CH[1],CH[2],CH[3],CH[4],CH[5]].concat(CHB,CHA),WIN=300,ALLC=CH.concat(CHB,CHA);
function pushSample(m){const t=(m.f/FPS).toFixed(1);
 const v=[[m.a1,m.a2,m.d1,m.d2],[Math.sqrt((m.u1*m.u1+m.u2*m.u2)/2),Math.sqrt((m.e1*m.e1+m.e2*m.e2)/2)],[m.pe,.12],[m.w1,m.t1],[m.w2,m.t2],[m.cd,.25,2]];
 const va=[[m.P1,m.I1,m.D1],[m.P2,m.I2,m.D2],[m.T1,m.t1,m.G1],[m.T2,m.t2,m.G2],[m.KE,m.PE],[m.M11,m.M12,m.M22]];
 const push=(c,vals)=>{c.data.labels.push(t);c.data.datasets.forEach((d,k)=>d.data.push(+(+vals[k]).toFixed(5)));if(c.data.labels.length>WIN){c.data.labels.shift();c.data.datasets.forEach(d=>d.data.shift())}};
 for(let i=0;i<6;i++)push(LINES[i],v[i]);CHA.forEach((c,i)=>push(c,va[i]))}
function pushBoat(b){const t=(b.f/FPS).toFixed(1),vals=[[b.we,b.ee,.5],[b.P,b.I,b.Dm,b.FF,b.th],[b.sp,b.W/100]];
 CHB.forEach((c,i)=>{c.data.labels.push(t);c.data.datasets.forEach((d,k)=>d.data.push(+(+vals[i][k]).toFixed(4)));if(c.data.labels.length>WIN){c.data.labels.shift();c.data.datasets.forEach(d=>d.data.shift())}})}
function clearCharts(){ALLC.forEach(c=>{c.data.labels.length=0;c.data.datasets.forEach(d=>{d.data.length=0;if(Array.isArray(d.backgroundColor))d.backgroundColor=[]});c.update('none')})}

/* ---------- state ---------- */
let fc=0,mi=0,ri=0,running=false,done=false,speed=4,showU=true,lastUi=0,lastChart=0,lastT=performance.now(),flash=0;
let bi=0,cb=null,lastF=0,cur=null,win=[],prevS=null,boat={x:TR[0][1],y:TR[0][2],h:0,v:0},trail=[];
const dtS=1/FPS;
function stationAt(f){for(let i=0;i<NS;i++)if(f<=ST[i].end)return i;return NS-1}
function boatAt(f){let lo=0,hi=TR.length-1;while(hi-lo>1){const mid=(lo+hi)>>1;if(TR[mid][0]<=f)lo=mid;else hi=mid}
 const a=TR[lo],b=TR[hi],k=b[0]>a[0]?clamp((f-a[0])/(b[0]-a[0]),0,1):0;return[a[1]+(b[1]-a[1])*k,a[2]+(b[2]-a[2])*k]}
function reveal(i){const s=ST[i];if(s.reachable===false){CH[6].data.labels.push(s.station);CH[6].data.datasets[0].data.push(0);CH[6].data.datasets[0].backgroundColor.push('#456');CH[7].data.labels.push(s.station);CH[7].data.datasets[0].data.push(+s.habitat_score);CH[7].data.datasets[0].backgroundColor.push(ZC[s.zone]);CH[6].update('none');CH[7].update('none');return}
 CH[6].data.labels.push(s.station);CH[6].data.datasets[0].data.push(+s.confidence_pct);CH[6].data.datasets[0].backgroundColor.push(s.correct?'#38bdf8':'#f87171');
 CH[7].data.labels.push(s.station);CH[7].data.datasets[0].data.push(+s.habitat_score);CH[7].data.datasets[0].backgroundColor.push(ZC[s.zone]);CH[6].update('none');CH[7].update('none');flash=1}
function advanceTo(F){fc=F;
 while(mi<M.length&&M[mi].f<=F){const m=M[mi++];cur=m;pushSample(m);win.push([m.f,Math.hypot(m.u1,m.u2),Math.hypot(m.e1,m.e2)]);if(prevS&&prevS.id===m.id&&m.f>prevS.f)m.v1=(m.a1-prevS.a1)/((m.f-prevS.f)/FPS),m.v2=(m.a2-prevS.a2)/((m.f-prevS.f)/FPS);prevS=m}
 while(bi<B.length&&B[bi].f<=F){cb=B[bi++];pushBoat(cb)}
 while(win.length&&win[0][0]<F-FPS*.6)win.shift();
 while(ri<NS&&ST[ri].scan<=F)reveal(ri++);
 const p=boatAt(F),dx=p[0]-boat.x,dy=p[1]-boat.y,sp=Math.hypot(dx,dy);if(sp>.002)boat.h=Math.atan2(dy,dx);const df=F-lastF;lastF=F;boat.v=df>.01?Math.min(6,sp/(df/FPS)):0;boat.x=p[0];boat.y=p[1];
 if(!trail.length||Math.hypot(p[0]-trail[trail.length-1][0],p[1]-trail[trail.length-1][1])>.4)trail.push(p)}
function reset(){fc=0;lastF=0;bi=0;cb=null;mi=0;ri=0;done=false;running=false;cur=null;win=[];prevS=null;trail=[];boat={x:TR[0][1],y:TR[0][2],h:0,v:0};clearCharts();$('sum').hidden=true;$('banner').style.display='none';$('camBox').hidden=true;advanceTo(0)}
function phaseNow(){const i=stationAt(fc),s=ST[i];if(done)return{i,ph:5};if(fc<s.start||!cur||cur.id!==s.id)return{i,ph:0};if(fc>=s.scan&&fc<s.scan+22&&s.reachable!==false)return{i,ph:4};return{i,ph:cur.ph>=1?cur.ph:0}}

/* ---------- map ---------- */
const cv=$('c'),g=cv.getContext('2d'),VPD=$('vp');
const xs=ST.map(s=>s.egg_x).concat(D.pens.map(p=>p[0]-D.pen_r),D.pens.map(p=>p[0]+D.pen_r),[0]),ys=ST.map(s=>s.egg_y).concat(D.pens.map(p=>p[1]-D.pen_r),D.pens.map(p=>p[1]+D.pen_r),[0]);
const BX=[Math.min(...xs)-6,Math.max(...xs)+6],BY=[Math.min(...ys)-6,Math.max(...ys)+6];
let W=1,H=1,dpr=1;function fit(){dpr=Math.min(devicePixelRatio||1,2);W=VPD.clientWidth;H=VPD.clientHeight;cv.width=W*dpr;cv.height=H*dpr}
new ResizeObserver(fit).observe(VPD);fit();
function mapT(){const pad=46,sx=(W-2*pad)/(BX[1]-BX[0]),sy=(H-2*pad)/(BY[1]-BY[0]),s=Math.min(sx,sy),ox=(W-s*(BX[1]-BX[0]))/2,oy=(H-s*(BY[1]-BY[0]))/2;
 return{s,X:x=>ox+(x-BX[0])*s,Y:y=>H-oy-(y-BY[0])*s}}
function drawMap(t){g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,W,H);
 const bgd=g.createLinearGradient(0,0,0,H);bgd.addColorStop(0,'#0a2236');bgd.addColorStop(1,'#07111b');g.fillStyle=bgd;g.fillRect(0,0,W,H);
 const T=mapT();g.lineWidth=1;g.strokeStyle='rgba(90,170,210,.10)';
 for(let x=Math.ceil(BX[0]/10)*10;x<=BX[1];x+=10){g.beginPath();g.moveTo(T.X(x),T.Y(BY[0]));g.lineTo(T.X(x),T.Y(BY[1]));g.stroke()}
 for(let y=Math.ceil(BY[0]/10)*10;y<=BY[1];y+=10){g.beginPath();g.moveTo(T.X(BX[0]),T.Y(y));g.lineTo(T.X(BX[1]),T.Y(y));g.stroke()}
 g.fillStyle='rgba(127,151,173,.7)';g.font='10px Segoe UI';g.fillText('10 m grid · top-down mission map',10,H-10);
 D.pens.forEach((p,i)=>{g.beginPath();g.arc(T.X(p[0]),T.Y(p[1]),D.pen_r*T.s,0,6.283);g.fillStyle='rgba(30,100,150,.12)';g.fill();g.strokeStyle='rgba(95,180,232,.5)';g.lineWidth=2;g.stroke();g.fillStyle='rgba(127,151,173,.8)';g.fillText('PEN '+(i+1),T.X(p[0])-14,T.Y(p[1])-D.pen_r*T.s-5)});
 g.setLineDash([6,5]);g.strokeStyle='rgba(120,160,190,.35)';g.lineWidth=1;g.beginPath();ST.forEach((s,i)=>{const x=T.X(s.egg_x),y=T.Y(s.egg_y);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();g.setLineDash([]);
 if(trail.length>1){g.strokeStyle='#fbbf24';g.lineWidth=2;g.beginPath();trail.forEach((p,i)=>i?g.lineTo(T.X(p[0]),T.Y(p[1])):g.moveTo(T.X(p[0]),T.Y(p[1])));g.lineTo(T.X(boat.x),T.Y(boat.y));g.stroke()}
 const ci=phaseNow().i;
 ST.forEach((s,i)=>{const x=T.X(s.egg_x),y=T.Y(s.egg_y),rv=i<ri,c=rv?ZC[s.zone]:'#456';
  if(i===ci&&!done){g.beginPath();g.arc(x,y,12+3*Math.sin(t*5),0,6.283);g.strokeStyle='#7dd3fc';g.lineWidth=2;g.stroke()}
  g.beginPath();g.arc(x,y,8,0,6.283);g.fillStyle=c;g.fill();if(rv&&!s.correct&&s.reachable!==false){g.strokeStyle='#fff';g.lineWidth=2;g.stroke()}
  g.fillStyle='#06121c';g.font='bold 9px Segoe UI';g.textAlign='center';g.fillText(s.station,x,y+3);
  if(rv){g.fillStyle='#c9d8e6';g.font='10px Segoe UI';g.fillText(s.predicted_stage,x,y+22)}g.textAlign='left'});
 const bx=T.X(boat.x),by=T.Y(boat.y);g.save();g.translate(bx,by);g.rotate(-boat.h);
 g.fillStyle='#f2a93a';g.strokeStyle='#fff';g.lineWidth=1.5;g.beginPath();g.moveTo(15,0);g.lineTo(-10,-8);g.lineTo(-6,0);g.lineTo(-10,8);g.closePath();g.fill();g.stroke();g.restore();
 const ph=phaseNow().ph;if(ph>=1&&ph<=4&&!done){g.beginPath();g.arc(bx,by,18+2*Math.sin(t*6),0,6.283);g.strokeStyle='rgba(56,189,248,.8)';g.lineWidth=2;g.stroke()}}

/* ---------- arm side view ---------- */
const ac=$('arm'),ag=ac.getContext('2d');
function drawArm(){const W2=230,H2=170;ag.clearRect(0,0,W2,H2);ag.fillStyle='#07111b';ag.fillRect(0,0,W2,H2);
 const ox=48,oy=30,sc=44;const P=(r,z)=>[ox+r*sc,oy-z*sc];ag.strokeStyle='#1d3145';ag.lineWidth=1;
 for(let i=0;i<=3;i++){ag.beginPath();ag.moveTo(0,oy+i*sc*1);ag.lineTo(W2,oy+i*sc*1);ag.stroke()}
 ag.strokeStyle='rgba(56,189,248,.35)';ag.beginPath();ag.moveTo(0,oy-sc*.2);ag.lineTo(W2,oy-sc*.2);ag.stroke();
 ag.fillStyle='#7f97ad';ag.font='9px Segoe UI';ag.fillText('boat hull',4,oy-sc*.2-3);
 ag.fillStyle='#f2a93a';ag.fillRect(ox-22,oy-12,44,10);
 const m=cur||{c1:85,c2:50,d1:85,d2:50},rad=Math.PI/180,fk=(a,b)=>{const t1=a*rad,t2=b*rad;const r1=L1*Math.cos(t1),z1=L1*Math.sin(t1);return[[r1,z1],[r1+L2*Math.cos(t1+t2),z1+L2*Math.sin(t1+t2)]]};
 const tgt=fk(m.d1,m.d2)[1],pt=fk(m.c1,m.c2),j=P(0,0),e=P(pt[0][0],pt[0][1]),t=P(pt[1][0],pt[1][1]),tg=P(tgt[0],tgt[1]);
 ag.strokeStyle='#34d399';ag.lineWidth=1.5;ag.beginPath();ag.arc(tg[0],tg[1],7,0,6.283);ag.stroke();ag.beginPath();ag.moveTo(tg[0]-11,tg[1]);ag.lineTo(tg[0]+11,tg[1]);ag.moveTo(tg[0],tg[1]-11);ag.lineTo(tg[0],tg[1]+11);ag.stroke();
 ag.lineCap='round';ag.lineWidth=6;ag.strokeStyle='#b8c4d0';ag.beginPath();ag.moveTo(j[0],j[1]);ag.lineTo(e[0],e[1]);ag.stroke();
 ag.lineWidth=5;ag.strokeStyle='#8fa3b8';ag.beginPath();ag.moveTo(e[0],e[1]);ag.lineTo(t[0],t[1]);ag.stroke();
 [j,e].forEach(p=>{ag.fillStyle='#f2a93a';ag.beginPath();ag.arc(p[0],p[1],5,0,6.283);ag.fill()});
 ag.fillStyle='#38bdf8';ag.beginPath();ag.arc(t[0],t[1],5,0,6.283);ag.fill();
 if(cur&&cur.ph>=2&&cur.ph<=3){ag.fillStyle='rgba(125,211,252,.12)';ag.beginPath();ag.moveTo(t[0],t[1]);ag.lineTo(t[0]-40,t[1]+90);ag.lineTo(t[0]+40,t[1]+90);ag.closePath();ag.fill()}
 ag.fillStyle='#7f97ad';ag.fillText('● target  ● camera',4,H2-5)}

/* ---------- UI ---------- */
const dotS=(id,on)=>{const e=$(id);e.classList.toggle('on',on);e.firstChild.classList.toggle('on',on)};
function ui(){const{i,ph}=phaseNow(),s=ST[i],u=ph>=1&&ph<=4&&!done,reachable=s.reachable!==false;
 dotS('s0',!running||done);$('s0').lastChild.textContent=done?'MISSION COMPLETE':running?(LK.on?'BLENDER PLAYING · LIVE':'REPLAY RUNNING'):(LK.on?'LINKED · BLENDER PAUSED':'SYSTEM READY');dotS('s1',running);
 dotS('s2',u&&cur&&cur.cd>=.25&&cur.cd<=2);dotS('s3',u&&reachable);dotS('s4',u&&reachable);
 for(let k=0;k<6;k++)$('wf'+k).className='wf'+(k===ph&&!done?' on':(k<ph||done)?' done':'');
 set('rx',f(boat.x,2)+' m');set('ry',f(boat.y,2)+' m');set('rv',f(boat.v,2)+' m/s');set('rt',f(fc/FPS,1)+' / '+f(D.last_frame/FPS,1)+' s');set('bfr',LK.on?Math.round(LK.bf)+' / '+LK.end:'—');set('lnk',LK.on?(LK.playing?'LIVE · playing':'LIVE · paused'):'offline');
 $('badge').innerHTML=`<big>STATION ${String(s.station).padStart(2,'0')} / ${NS}</big>${s.true_stage} · <b>${done?'COMPLETE':running||fc>0?(reachable?LABEL[ph]:'UNREACHABLE — SKIPPED'):'READY'}</b>`;
 set('stn',`${s.station} / ${NS}`);set('epos',`${f(s.egg_x,1)}, ${f(s.egg_y,1)} m`);set('st',done?'COMPLETE':LABEL[ph]);
 const L=ri>0?ST[ri-1]:null;
 if(L){set('lat','#'+L.station);set('pred',L.predicted_stage);set('tru',L.true_stage);set('conf',reachableOf(L)?f(L.confidence_pct,1)+'%':'—');const ok=L.correct;set('res',reachableOf(L)?(ok?'CORRECT':'INCORRECT'):'UNREACHABLE');$('res').className=ok?'ok':'bad';
  set('zone',L.zone);$('zone').style.color=ZC[L.zone];set('hab',f(L.habitat_score,0));WK.forEach(w=>set('w_'+w[0],f(L[w[0]],w[3])+' '+w[2]));
  const im=L.live||L.ds;if(im){$('camBox').hidden=false;if($('camImg').getAttribute('src')!==im)$('camImg').src=im;set('camCap','#'+L.station+' · '+L.true_stage)}}
 if(cur&&u&&reachable){set('cdist',f(cur.cd,2)+' m'+(cur.cd>=.25&&cur.cd<=2?' ✓':''));set('cang',f(cur.va,1)+'°');set('cstab',cur.st?'YES':'NO');set('cval',cur.sv?'YES':'NO');
  set('dis','ACTIVE');set('cmp','ACTIVE');
  let b=0,a=0;win.forEach(x=>{b+=x[1]*x[1];a+=x[2]*x[2]});const n=win.length||1,rb=Math.sqrt(b/n),ra=Math.sqrt(a/n);set('ebv',f(rb,3)+'°');set('eav',f(ra,3)+'°');set('imp',rb>1e-6?f(Math.max(0,(rb-ra)/rb*100),1)+'%':'—');
  set('j1a',f(cur.a1,1)+'°');set('j2a',f(cur.a2,1)+'°');set('j1v',sg(cur.v1||0,1)+' °/s');set('j2v',sg(cur.v2||0,1)+' °/s');set('j1t',sg(cur.t1,3)+' Nm');set('j2t',sg(cur.t2,3)+' Nm');set('j1d',sg(cur.w1,4)+' Nm');set('j2d',sg(cur.w2,4)+' Nm')}
 else{['cdist','cang','cstab','cval','ebv','eav','imp'].forEach(k=>set(k,'—'));set('dis','IDLE');set('cmp','STANDBY')}
 const msg=done?'':!reachable?'TARGET OUT OF ARM REACH':ph===1?'ARM UNFOLDING · TARGETING':ph===2?'STABILIZING · COMPENSATION ACTIVE':ph===3?'SCANNING':ph===4?'IMAGE CAPTURED · '+s.predicted_stage.toUpperCase():ph===5?'ARM FOLDING':'';set('msg',msg);
 eqPanels(i,ph,s,u,reachable);
 $('prog').firstChild.style.width=(fc/D.last_frame*100)+'%'}
const rj=(x,n,w)=>(x==null||isNaN(x)?'—':(+x).toFixed(n)).padStart(w||0),R=Math.PI/180;
function eqPanels(i,ph,s,u,reachable){
 // --- boat PID
 let tx='';const kp=BG[0],ki=BG[1],kd=BG[2];
 if(!B.length||!cb)tx=B.length?'waiting for the boat to start moving…':'boat log not available in this run';
 else{const mv=ph===0&&!done,sat=cb.sat?'  <u>⚠ SATURATED</u>':'';
  tx=`state          : ${mv?'<b>NAVIGATING</b>':'<b>LOCKED</b> (holding position, arm working)'}
target         : station ${s.station}   A* waypoint ${cb.wi+1}/${cb.nw}
gains          : Kp=${kp} N/m   Ki=${ki} N/(m·s)   Kd=${kd} N·s/m
|e| to waypoint: ${rj(cb.we,2,6)} m     to egg: ${rj(cb.ee,2,6)} m
P = Kp·e       : ${rj(cb.P,1,7)} N
I = Ki·∫e      : ${rj(cb.I,2,7)} N   (clamped ±3 m·s)
D = Kd·de/dt   : ${rj(cb.Dm,1,7)} N
F_ff (current) : ${rj(cb.FF,1,7)} N   (cancels the 1.5/−0.8 m/s current)
F_cmd = Σ      : ${rj(cb.th,1,7)} N   (limit 400 N)${sat}
F_disturbance  : ${rj(cb.W,1,7)} N      speed: ${rj(cb.sp,2,5)} m/s`}
 $('eqBoat').innerHTML=tx;
 // --- arm PID
 let ta='';
 if(!cur||!reachable)ta=reachable?'arm folded — waiting':'target out of reach — arm not deployed';
 else{const ex=j=>j?cur.d2-cur.a2:cur.d1-cur.a1,lim=(PR.torque_limit||2.5);
  const row=(j)=>{const e=ex(j),P=j?cur.P2:cur.P1,I=j?cur.I2:cur.I1,Dd=j?cur.D2:cur.D1,raw=P+I+Dd,out=j?cur.t2:cur.t1,w=j?cur.w2:cur.w1;
   return`joint ${j+1}:  e=${rj(e,3,7)}° (${rj(e*R,4,7)} rad)
  P=Kp·e   = ${rj(P,3,7)} Nm   (Kp=${AG[0]})
  I=Ki·∫e  = ${rj(I,3,7)} Nm   (Ki=${AG[1]})
  D=Kd·ė   = ${rj(Dd,3,7)} Nm   (Kd=${AG[2]})
  Σ = ${rj(raw,3,7)} → τ_applied = <b>${rj(out,3,7)} Nm</b>${Math.abs(raw)>lim+1e-6?'  <u>⚠ clamped ±'+lim+'</u>':''}
  τ_disturbance = ${rj(w,4,8)} Nm`};
  ta=row(0)+'\n\n'+row(1)+`\n\nphase: ${LABEL[cur.ph]||'—'}   stable: ${cur.st?'yes':'no'}`}
 $('eqArm').innerHTML=ta;
 // --- IK
 const L1=AL[0],L2=AL[1];let tk='';
 if(s.nav_reach_d_m==null)tk='no navigation data for this station';
 else{const d=s.nav_reach_d_m,c2=(d*d-L1*L1-L2*L2)/(2*L1*L2),ok=s.nav_ik_reachable;
  tk=`station ${s.station}:  boat parked ${rj(s.nav_park_dist_m,2)} m from the egg (horizontal)
d = √(r²+z²)         = ${rj(d,3)} m
reach window         = [${rj(s.nav_reach_min_m,2)} , ${rj(s.nav_reach_max_m,2)}] m  →  ${ok?'<b>REACHABLE ✓</b>':'<u>OUT OF REACH ✗</u>'}
cos q₂ = (d²−L₁²−L₂²)/2L₁L₂ = ${rj(c2,4)}
yaw ψ                = ${rj(s.nav_ik_yaw_deg,1)}°
shoulder q₁ (IK)     = ${rj(s.nav_ik_shoulder_deg,1)}°
elbow q₂ (IK)        = ${rj(s.nav_ik_elbow_deg,1)}°
L₁ / L₂              = ${L1} / ${L2} m
A* waypoints         = ${s.nav_n_waypoints}   path ${rj(s.nav_path_length_m,1)} m   driven ${rj(s.nav_travel_m,1)} m   timeouts ${s.nav_timeouts}`}
 $('eqIK').innerHTML=tk;
 // --- Lagrangian
 let tl='';
 if(!cur||!reachable)tl='arm folded — no motion yet';
 else{const q1=cur.a1*R,q2=cur.a2*R;
  tl=`m₁=${LG.m1} kg  m₂=${LG.m2} kg  g_eff=${rj(LG.g_eff,3)} m/s² (buoyancy-reduced)
q  = [${rj(cur.a1,2,7)}°, ${rj(cur.a2,2,7)}°]   q̇ = [${rj(cur.V1,1,7)}, ${rj(cur.V2,1,7)}] °/s   q̈ = [${rj(cur.A1,2,6)}, ${rj(cur.A2,2,6)}] rad/s²
M(q) = [ ${rj(cur.M11,4,8)}  ${rj(cur.M12,4,8)} ]  kg·m²
       [ ${rj(cur.M12,4,8)}  ${rj(cur.M22,4,8)} ]
C·q̇ = [ ${rj(cur.C1,5,9)} , ${rj(cur.C2,5,9)} ] Nm   (Coriolis / centrifugal)
G(q) = [ ${rj(cur.G1,4,8)} , ${rj(cur.G2,4,8)} ] Nm   (gravity − buoyancy)
τ_required = M·q̈ + C·q̇ + G = [ <b>${rj(cur.T1,4,8)}</b> , <b>${rj(cur.T2,4,8)}</b> ] Nm
τ_applied (PID)             = [ ${rj(cur.t1,4,8)} , ${rj(cur.t2,4,8)} ] Nm
kinetic T = ${rj(cur.KE,5)} J     potential V = ${rj(cur.PE,4)} J     L = T − V = ${rj(cur.KE-cur.PE,4)} J`}
 $('eqLag').innerHTML=tl;eqExtra(reachable)}
function eqExtra(reachable){
 const am=PR.arm_model||{damping:2.5,stiffness:8,inertia:1},Lam=4;let a='',b='';
 if(!cur||!reachable){a=b=reachable?'arm folded — waiting':'target out of reach — arm not deployed'}
 else{
  const qd=[(cur.V1||0)*R,(cur.V2||0)*R],Dq=[am.damping*qd[0],am.damping*qd[1]],tauReq=[cur.T1,cur.T2],tau=[cur.t1,cur.t2],td=[cur.w1,cur.w2];
  const res=[0,1].map(j=>(tau[j]+td[j])-(tauReq[j]+Dq[j]));
  a=`b (damping) = ${am.damping} Nm·s/rad   k (stiffness) = ${am.stiffness} Nm/rad   J = ${am.inertia} kg·m²
M·q̈ + C·q̇ + G   = [ ${rj(cur.T1,4,8)} , ${rj(cur.T2,4,8)} ] Nm     (inverse Lagrangian)
D·q̇ = b·q̇       = [ ${rj(Dq[0],4,8)} , ${rj(Dq[1],4,8)} ] Nm     (viscous drag)
τ + τ_d          = [ ${rj(tau[0]+td[0],4,8)} , ${rj(tau[1]+td[1],4,8)} ] Nm     (PID applied + water disturbance)
residual (τ+τ_d) − (Mq̈+Cq̇+G+Dq̇) = [ ${rj(res[0],4,8)} , ${rj(res[1],4,8)} ] Nm
  (lumped Blender joint model vs rigid-body 2-link Lagrangian)`;
  const hold=cur.ph===2||cur.ph===3;
  if(!hold)b='evaluated while the arm HOLDS position (stabilization / scan).\nphase now: '+(LABEL[cur.ph]||'—');
  else{
   const e=[(cur.d1-cur.a1)*R,(cur.d2-cur.a2)*R],ed=[-qd[0],-qd[1]],r=[ed[0]+Lam*e[0],ed[1]+Lam*e[1]],
    VL=.5*(cur.M11*r[0]*r[0]+2*cur.M12*r[0]*r[1]+cur.M22*r[1]*r[1]);
   b=`e   = [ ${rj(e[0],5,9)} , ${rj(e[1],5,9)} ] rad
ė   = −q̇ (q̇des≈0) = [ ${rj(ed[0],5,9)} , ${rj(ed[1],5,9)} ] rad/s
r = ė + Λe (Λ=${Lam})  = [ ${rj(r[0],5,9)} , ${rj(r[1],5,9)} ] rad/s     |r| = ${rj(Math.hypot(r[0],r[1]),5,8)}
½ rᵀ M r  (Lyapunov, arm part) = <b>${VL.toExponential(3)}</b> J
M(q) = [ ${rj(cur.M11,4,7)} ${rj(cur.M12,4,7)} ; ${rj(cur.M12,4,7)} ${rj(cur.M22,4,7)} ] kg·m²`}}
 $('eqLE').innerHTML=a;$('eqNN').innerHTML=b}
const reachableOf=s=>s.reachable!==false;

/* ---------- summary ---------- */
function summary(){const cm=D.confusion,pc=D.per_class,c=mis.classification||{},n=ST.filter(reachableOf).length;
 const card=(l,v)=>`<div class=card><small>${l}</small><b>${v}</b></div>`,z=k=>ST.filter(s=>s.zone===k).length;
 let h=`<h2>MISSION COMPLETE</h2><div class=cards>${card('Total stations',NS)}${card('Images captured',ST.filter(s=>s.live||s.ds).length)}${card('Classification accuracy',f(c.accuracy,1)+'%')}${card('Healthy vs unhealthy',f(c.healthy_vs_unhealthy_accuracy,1)+'%')}${card('Average confidence',f(mis.avg_confidence_pct,1)+'%')}${card('Joint RMS error J1 / J2',f(ar.j1_rms_err_deg,3)+' / '+f(ar.j2_rms_err_deg,3)+'°')}${card('Position RMS error',f(ar.position_rms_m,4)+' m')}${card('Max compensation torque',f(ar.max_torque_nm,2)+' Nm')}${card('Error reduction',f(ar.mean_error_reduction_pct,1)+'%')}${card('Camera stable',f(cam.stable_pct,1)+'%')}${card('Valid scans',f(cam.valid_scan_pct,1)+'%')}${card('Avg camera distance',f(cam.avg_distance_m,2)+' m')}${card('GREEN zones',z('GREEN'))}${card('YELLOW zones',z('YELLOW'))}${card('RED zones',z('RED'))}${card('Mean habitat',f(mis.mean_habitat,1))}</div>
 <div class=sub>NAVIGATION · BOAT PID (Kp=${BG[0]} Ki=${BG[1]} Kd=${BG[2]}, current feedforward)</div><div class=cards>${card('Total distance driven',f(nv.total_distance_m,1)+' m')}${card('Mean / max speed',f(nv.mean_speed_ms,2)+' / '+f(nv.max_speed_ms,2)+' m/s')}${card('Mean park error',f(nv.mean_park_error_m,2)+' m')}${card('Max park error',f(nv.max_park_error_m,2)+' m')}${card('Peak / mean thrust',f(nv.peak_thrust_n,0)+' / '+f(nv.mean_thrust_n,0)+' N')}${card('Thrust saturation',f(nv.thrust_saturation_pct,2)+'%')}${card('Waypoint timeouts',nv.waypoint_timeouts??'—')}${card('RMS P / I / D / FF',[nv.rms_p_n,nv.rms_i_n,nv.rms_d_n,nv.rms_ff_n].map(x=>f(x,0)).join(' / ')+' N')}</div>
 <div class=sub>ARM PID (Kp=${AG[0]} Ki=${AG[1]} Kd=${AG[2]}) · LAGRANGIAN DYNAMICS</div><div class=cards>${card('RMS P / I / D · joint 1',[dy.rms_pid_p_j1,dy.rms_pid_i_j1,dy.rms_pid_d_j1].map(x=>f(x,2)).join(' / ')+' Nm')}${card('RMS P / I / D · joint 2',[dy.rms_pid_p_j2,dy.rms_pid_i_j2,dy.rms_pid_d_j2].map(x=>f(x,2)).join(' / ')+' Nm')}${card('Torque saturation',f(dy.torque_saturation_pct,1)+'%')}${card('Peak τ required J1 / J2',f(dy.peak_tau_dyn_j1_nm,2)+' / '+f(dy.peak_tau_dyn_j2_nm,2)+' Nm')}${card('RMS gravity G J1 / J2',f(dy.rms_gravity_j1_nm,2)+' / '+f(dy.rms_gravity_j2_nm,2)+' Nm')}${card('RMS Coriolis J1 / J2',f(dy.rms_coriolis_j1_nm,4)+' / '+f(dy.rms_coriolis_j2_nm,4)+' Nm')}${card('Max kinetic energy',f(dy.max_ke_j,4)+' J')}${card('M11 range',(dy.M11_range||[]).map(x=>f(x,2)).join(' – ')+' kg·m²')}</div>
 <div class=cards>${card('Macro precision',f(c.macro_precision*100,1)+'%')}${card('Macro recall',f(c.macro_recall*100,1)+'%')}${card('Macro F1-score',f(c.macro_f1*100,1)+'%')}</div>
 <b>Confusion matrix</b> <span style="color:var(--m)">(rows = TRUE class, columns = PREDICTED class)</span><div class=tw><table><tr><th></th>${CL.map(x=>`<th>${x}</th>`).join('')}<th>Recall</th></tr>`;
 cm.forEach((r,i)=>{h+=`<tr><th>${CL[i]}</th>${r.map((v,j)=>`<td style="background:rgba(${i===j?'52,211,153':'248,113,113'},${v?Math.min(.65,.15+v*.15):0})">${v}</td>`).join('')}<td>${f(pc[i].recall*100,0)}%</td></tr>`});
 h+=`<tr><th>Precision</th>${pc.map(p=>`<td>${f(p.precision*100,0)}%</td>`).join('')}<td></td></tr><tr><th>F1-score</th>${pc.map(p=>`<td>${f(p.f1*100,0)}%</td>`).join('')}<td></td></tr></table></div>
 <p><b>Stations</b></p><div class=tw><table><tr><th>#</th><th>Robot cam</th><th>Dataset</th><th>True</th><th>Predicted</th><th>Conf</th><th>OK</th><th>Zone</th><th>Habitat</th><th>Cam dist m</th><th>Stable %</th><th>Err before°</th><th>Err after°</th><th>Reduction</th><th>A* wps</th><th>Driven m</th><th>Park err m</th><th>Reach d m</th><th>IK q1°</th><th>IK q2°</th></tr>`;
 ST.forEach(s=>{h+=`<tr><td>${s.station}</td><td>${s.live?`<img src="${s.live}" onclick="openImg(this.src)">`:'—'}</td><td>${s.ds?`<img src="${s.ds}" onclick="openImg(this.src)">`:'—'}</td><td>${s.true_stage}</td><td>${s.predicted_stage}</td><td>${f(s.confidence_pct,1)}%</td><td class="${s.correct?'ok':'bad'}">${s.correct?'✓':'✗'}</td><td><span class=zt style="background:${ZC[s.zone]}">${s.zone}</span></td><td>${f(s.habitat_score,0)}</td><td>${f(s.camera_distance_m,2)}</td><td>${f(s.stability_pct,0)}</td><td>${f(s.err_before_deg,3)}</td><td>${f(s.err_after_deg,3)}</td><td>${f(s.err_reduction_pct,0)}%</td><td>${s.nav_n_waypoints??'—'}</td><td>${f(s.nav_travel_m,1)}</td><td>${f(s.nav_park_dist_m,2)}</td><td>${f(s.nav_reach_d_m,2)}</td><td>${f(s.nav_ik_shoulder_deg,1)}</td><td>${f(s.nav_ik_elbow_deg,1)}</td></tr>`});
 $('sum').innerHTML=h+'</table></div>';$('sum').hidden=false;$('banner').style.display='flex';setTimeout(()=>$('banner').style.display='none',2500)}
window.openImg=src=>{const w=window.open();if(w)w.document.write(`<body style="margin:0;background:#000"><img src="${src}" style="max-width:100%;display:block;margin:auto">`)};

/* ---------- exports ---------- */
const COLS=['station','true_stage','predicted_stage','confidence_pct','correct','reachable','rescanned','zone','habitat_score','temperature_C','pH','oxygen_mgL','turbidity_NTU','ammonia_mgL','nitrate_mgL','conductivity_uScm','light','flow_ms','egg_x','egg_y','joint1_deg','joint2_deg','tau1_nm','tau2_nm','camera_distance_m','viewing_angle_deg','stability_pct','scan_valid_pct','err_before_deg','err_after_deg','err_reduction_pct','nav_n_waypoints','nav_path_length_m','nav_travel_m','nav_travel_frames','nav_timeouts','nav_park_dist_m','nav_reach_d_m','nav_ik_yaw_deg','nav_ik_shoulder_deg','nav_ik_elbow_deg','nav_ik_reachable'];
const csvRes=()=>[COLS.join(',')].concat(ST.map(s=>COLS.map(c=>s[c]==null?'':s[c]).join(','))).join('\n');
const csvCM=()=>['true\\pred,'+CL.join(',')].concat(D.confusion.map((r,i)=>CL[i]+','+r.join(','))).join('\n');
const csvBoat=()=>{if(!B.length)return'';const k=Object.keys(B[0]);return[k.join(',')].concat(B.map(m=>k.map(c=>m[c]).join(','))).join('\n')};
const csvArm=()=>{const k=Object.keys(M[0]);return[k.join(',')].concat(M.map(m=>k.map(c=>m[c]).join(','))).join('\n')};
function saveBlob(b,n){const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=n;document.body.appendChild(a);a.click();setTimeout(()=>{URL.revokeObjectURL(a.href);a.remove()},2000)}
async function zip(all){if(typeof JSZip==='undefined')return alert('JSZip could not load (needs internet). CSV download still works.');const z=new JSZip(),fo=all?z.folder('captured_images'):z;let n=0;
 ST.forEach(s=>{if(s.live){fo.file(`live_${String(s.station).padStart(3,'0')}_${s.true_stage}.png`,s.live.split(',')[1],{base64:true});n++}if(s.ds){fo.file(`scan_${String(s.station).padStart(3,'0')}_${s.true_stage}.${s.ds.includes('png')?'png':'jpg'}`,s.ds.split(',')[1],{base64:true});n++}});
 if(!n&&!all)return alert('No images were embedded in this report.');if(all){z.file('scan_results.csv',csvRes());z.file('confusion_matrix.csv',csvCM());z.file('joint_motion_dynamics_samples.csv',csvArm());if(B.length)z.file('boat_pid_samples.csv',csvBoat())}
 saveBlob(await z.generateAsync({type:'blob'}),all?'results.zip':'captured_images.zip')}

/* ---------- controls ---------- */
const SP=[1,4,12,40],LSP=[.5,1,2,4];   // site speed buttons: standalone replay multiplier | Blender playback speed when linked
const SPsel=i=>SP.forEach((_,j)=>$('sp'+j).classList.toggle('act',j===i));
SP.forEach((v,i)=>$('sp'+i).onclick=()=>{speed=v;SPsel(i);lkSend('speed',{value:LSP[i]})});

/* ---------- Blender live link: two-way sync (Blender timeline <-> this page) ---------- */
const LK={on:false,base:null,bf:0,bt:0,playing:false,speed:1,end:D.last_frame,pend:0,pendF:0,es:null};
const lkCands=()=>{const c=[];if(location.protocol.indexOf('http')===0)c.push(location.origin);c.push('http://127.0.0.1:8765','http://localhost:8765');return c.filter((v,i)=>c.indexOf(v)===i)};
function lkSend(cmd,extra){if(!LK.on)return;fetch(LK.base+'/cmd',{method:'POST',headers:{'Content-Type':'text/plain'},body:JSON.stringify(Object.assign({cmd:cmd},extra||{}))}).catch(()=>{})}
function lkUi(){const e=$('lk');if(!e)return;e.className='lk'+(LK.on?' on':'')+(LK.on&&LK.playing?' pl':'');e.textContent=LK.on?(LK.playing?'● LIVE · BLENDER PLAYING':'● LINKED · BLENDER PAUSED'):'○ STANDALONE REPLAY (no Blender link)'}
async function lkConnect(){for(const b of lkCands()){try{const r=await fetch(b+'/state',{cache:'no-store'});if(r.ok){LK.base=b;return lkOpen()}}catch(e){}}setTimeout(lkConnect,3000)}
function lkOpen(){const es=new EventSource(LK.base+'/events');LK.es=es;
 es.onmessage=ev=>{try{lkRecv(JSON.parse(ev.data))}catch(e){}};
 es.onerror=()=>{es.close();if(LK.on){LK.on=false;LK.playing=false;running=false}lkUi();setTimeout(lkConnect,2000)}}
function lkRecv(s){const now=performance.now();LK.on=true;LK.end=s.frame_end;LK.speed=s.speed||1;
 if(now>=LK.pendF){LK.bf=s.frame;LK.bt=now}                       // frame: Blender is the master (ignored briefly right after a local seek)
 if(now>=LK.pend){LK.playing=!!s.playing;running=LK.playing}        // play state: same, ignored briefly right after a local play/pause
 const k=LSP.findIndex(v=>Math.abs(v-LK.speed)<.01);if(k>=0)SPsel(k);lkUi()}
function lkLocal(playing,frame){const n=performance.now();if(playing!=null){LK.playing=playing;running=playing;LK.pend=n+600}if(frame!=null){LK.bf=frame;LK.bt=n;LK.pendF=n+300}lkUi()}
function linkTick(now){let F=LK.bf;if(LK.playing)F+=Math.min(.25,(now-LK.bt)/1000)*FPS*LK.speed;   // smooth extrapolation between Blender messages
 F=clamp(F,0,D.last_frame);
 if(F<fc-.5){const r=running;reset();advanceTo(F);running=r}       // Blender went backwards (scrub / rewind / replay)
 else if(F>fc+1e-6)advanceTo(F);
 if(!done&&F>=D.last_frame-.5&&LK.bf>=Math.min(D.last_frame,LK.end)-.5)finishNow()}

$('bStart').onclick=()=>{if(LK.on){lkLocal(true);lkSend('play');return}if(done)reset();running=true};
$('bPause').onclick=()=>{if(LK.on){lkLocal(false);lkSend('pause');return}running=false};
$('bReset').onclick=()=>{if(LK.on){lkLocal(false,1);lkSend('reset')}reset()};
$('bNext').onclick=()=>{if(done)return;const i=stationAt(fc),F=Math.min(D.last_frame,ST[i].end+1);if(LK.on){lkLocal(null,F);lkSend('seek',{frame:Math.round(F)})}advanceTo(F);if(fc>=D.last_frame)finishNow()};
$('bComp').onclick=()=>{showU=!showU;$('bComp').textContent=showU?'ON':'OFF';$('bComp').classList.toggle('act',showU);CH[1].getDatasetMeta(0).hidden=!showU;CH[1].update('none')};
$('bCsv').onclick=()=>saveBlob(new Blob([csvRes()],{type:'text/csv'}),'scan_results.csv');$('bZip').onclick=()=>zip(false);$('bAll').onclick=()=>zip(true);
$('prog').onclick=e=>{const r=$('prog').getBoundingClientRect(),F=clamp((e.clientX-r.left)/r.width,0,1)*D.last_frame,was=running;if(LK.on){lkLocal(null,F);lkSend('seek',{frame:Math.round(F)})}reset();advanceTo(F);running=was};
function finishNow(){done=true;if(!LK.on)running=false;summary()}

/* ---------- loop ---------- */
function loop(now){requestAnimationFrame(loop);const dt=Math.min(.1,(now-lastT)/1000);lastT=now;
 if(LK.on)linkTick(now);else if(running&&!done){advanceTo(Math.min(D.last_frame,fc+dt*FPS*speed));if(fc>=D.last_frame)finishNow()}
 drawMap(now/1000);drawArm();if(flash>0){$('flash').style.opacity=flash*.35;flash=Math.max(0,flash-dt*4)}
 if(now-lastUi>100){lastUi=now;ui()}if(now-lastChart>250){lastChart=now;LINES.forEach(c=>c.update('none'))}}
reset();lkUi();lkConnect();requestAnimationFrame(loop);
</script></body></html>
'''


# ----------------------------------------------------------------------------------------------
# 7. MAIN ENTRY POINT
# ----------------------------------------------------------------------------------------------
def build_final_package(output_dir, scan_results, motion_log, egg_data, events=None,
                        scene=None, robot_components=None, render_images=True, seed=7, pens=None,
                        nav_log=None, boat_log=None, arm_lengths=(0.9, 0.75), boat_gains=None,
                        arm_gains=None, torque_limit=2.5, arm_model=None, live_link=True, port=8765):
    t0 = time.time()
    out = Path(output_dir); fin = out / "final_report"; fin.mkdir(parents=True, exist_ok=True)
    arm_dir = out / "arm_analysis"; arm_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    by_id = {e["id"]: e for e in egg_data}

    # camera images (needs Blender)
    if render_images and scene is not None and robot_components is not None:
        try: render_station_images(scene, robot_components, scan_results, egg_data, out)
        except Exception as exc: print("[FINAL] image rendering skipped:", repr(exc))

    graph_files = plot_arm_graphs(motion_log, str(arm_dir))
    arm = station_arm_stats(motion_log)
    lag_params = compute_dynamics(motion_log, arm_lengths)
    save_dynamics_csv(motion_log, fin / "joint_dynamics_lagrangian.csv")
    save_boat_csv(boat_log, fin / "boat_pid_log.csv")
    nav_summary, dyn_summary = summarize_navigation(boat_log, nav_log), summarize_dynamics(motion_log)

    # ---- station table
    rows = []
    for i, r in enumerate(scan_results):
        egg = by_id.get(r["target_id"], {}); a = arm.get(r["target_id"], {})
        w = make_water(rng); hs = habitat_score(w)
        rows.append(dict(
            station=i + 1, target_id=r["target_id"], egg_x=egg.get("x", 0.0), egg_y=egg.get("y", 0.0),
            true_stage=r["ground_truth"], true_bin=r["ground_truth_bin"],
            predicted_stage=r["predicted"], predicted_bin=r["predicted_bin"],
            confidence_pct=round(100 * r["confidence"], 1), correct=bool(r["correct"]),
            reachable=r.get("reachable", True), rescanned=r["confidence"] < 0.70 and r.get("reachable", True),
            zone=zone(hs), habitat_score=round(hs, 1),
            temperature_C=round(w['T'], 2), pH=round(w['pH'], 2), oxygen_mgL=round(w['O2'], 2),
            turbidity_NTU=round(w['turb'], 2), ammonia_mgL=round(w['NH3'], 3), nitrate_mgL=round(w['NO3'], 1),
            conductivity_uScm=round(w['cond'], 0), light=round(w['light'], 2), flow_ms=round(w['flow'], 3),
            **{k: round(v, 4) for k, v in a.items()},
            **{"nav_"+k: (round(v, 3) if isinstance(v, float) else v) for k, v in (nav_log or {}).get(r["target_id"], {}).items()},
            live_image=r.get("live_image", ""), scan_image=r.get("scan_image", "")))
    if not rows:
        print("[FINAL] no scan results - nothing to package"); return None
    # union of columns over ALL rows (unreachable stations have no arm stats)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(fin / "stations.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys, restval=""); wr.writeheader(); wr.writerows(rows)

    # ---- classification
    cm, per, summ = confusion_and_metrics([r for r in rows if r["reachable"]])
    with open(fin / "confusion_matrix.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["true\\pred"] + CLASSES)
        for c, row in zip(CLASSES, cm): wr.writerow([c] + row)
    with open(fin / "class_metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["cls", "precision", "recall", "f1", "support"]); wr.writeheader(); wr.writerows(per)

    # ---- mission / arm / camera / habitat summary
    reach = [r for r in rows if r["reachable"]]
    zones = {z: sum(1 for r in rows if r["zone"] == z) for z in ("GREEN", "YELLOW", "RED")}
    last_frame = max((m["frame"] for m in motion_log), default=0)
    mission = dict(
        stations=len(rows), unreachable=len(rows) - len(reach),
        rescans=sum(1 for r in rows if r["rescanned"]),
        mission_time_s=round(last_frame / 24.0, 1), frames=last_frame,
        avg_confidence_pct=round(_mean([r["confidence_pct"] for r in reach]), 1),
        zones=zones, mean_habitat=round(_mean([r["habitat_score"] for r in rows]), 1),
        arm=dict(
            j1_rms_err_deg=round(_rms([m["theta1_error_deg"] for m in motion_log]), 4),
            j2_rms_err_deg=round(_rms([m["theta2_error_deg"] for m in motion_log]), 4),
            position_rms_m=round(_rms([m["position_error"] for m in motion_log]), 4),
            max_torque_nm=round(max((max(abs(m["compensation_joint1"]), abs(m["compensation_joint2"])) for m in motion_log), default=0), 3),
            mean_error_reduction_pct=round(_mean([r.get("err_reduction_pct", 0) for r in reach]), 1)),
        camera=dict(
            avg_distance_m=round(_mean([m["camera_target_distance"] for m in motion_log]), 3),
            avg_view_angle_deg=round(_mean([m["camera_viewing_angle"] for m in motion_log]), 2),
            stable_pct=round(100 * _mean([1.0 if m["camera_stable"] else 0.0 for m in motion_log]), 1),
            valid_scan_pct=round(100 * _mean([1.0 if m["scan_valid"] else 0.0 for m in motion_log]), 1)),
        classification=summ, navigation=nav_summary, dynamics=dyn_summary,
        params=dict(boat_gains=list(boat_gains or []), arm_gains=list(arm_gains or []), torque_limit=torque_limit,
                    arm_lengths=list(arm_lengths), lagrangian=lag_params,
                    arm_model=dict(arm_model or dict(damping=2.5, stiffness=8.0, inertia=1.0))))
    report = dict(mission=mission, per_class=per, confusion=dict(classes=CLASSES, matrix=cm),
                  events=[(round(t, 2), e) for t, e in (events or [])], stations=rows)
    (fin / "mission_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # ---- charts
    svg_confusion(cm, fin / "confusion_matrix.svg")
    svg_station_map(rows, fin / "station_map.svg")
    svg_bars(["GREEN", "YELLOW", "RED"], [zones["GREEN"], zones["YELLOW"], zones["RED"]],
             "Stations per habitat zone", fin / "zone_counts.svg", [ZCOL["GREEN"], ZCOL["YELLOW"], ZCOL["RED"]])
    svg_bars([f"#{r['station']}" for r in rows], [r["habitat_score"] for r in rows],
             "Habitat score per station", fin / "habitat_scores.svg", [ZCOL[r["zone"]] for r in rows])
    svg_bars([f"#{r['station']}" for r in reach], [r.get("err_before_deg", 0) for r in reach],
             "Joint error WITHOUT compensation (deg RMS)", fin / "error_before.svg", ["#d62728"] * len(reach))
    svg_bars([f"#{r['station']}" for r in reach], [r.get("err_after_deg", 0) for r in reach],
             "Joint error WITH PID compensation (deg RMS)", fin / "error_after.svg", ["#2ca02c"] * len(reach))

    # ---- text report
    c, a, m = mission["classification"], mission["arm"], mission["camera"]
    L = ["ZEBRAFISH EGG INSPECTION - MISSION REPORT", "=" * 60,
         f"Stations: {mission['stations']}  (unreachable {mission['unreachable']}, re-scans {mission['rescans']})",
         f"Mission time: {mission['mission_time_s']} s  ({mission['frames']} frames)", "",
         "CLASSIFICATION", f"  accuracy {c['accuracy']:.1f}%   healthy/unhealthy {c['healthy_vs_unhealthy_accuracy']:.1f}%",
         f"  macro P {c['macro_precision']:.3f}  R {c['macro_recall']:.3f}  F1 {c['macro_f1']:.3f}   avg confidence {mission['avg_confidence_pct']}%", "",
         "ARM CONTROL", f"  RMS joint error  J1 {a['j1_rms_err_deg']} deg   J2 {a['j2_rms_err_deg']} deg",
         f"  position RMS {a['position_rms_m']} m   max torque {a['max_torque_nm']} Nm   error reduction {a['mean_error_reduction_pct']}%", "",
         "CAMERA", f"  avg distance {m['avg_distance_m']} m   avg view angle {m['avg_view_angle_deg']} deg",
         f"  stable {m['stable_pct']}%   valid scan {m['valid_scan_pct']}%", "",
         "HABITAT", f"  mean score {mission['mean_habitat']}   GREEN {zones['GREEN']}  YELLOW {zones['YELLOW']}  RED {zones['RED']}", "",
         "PER STATION"]
    for r in rows:
        L.append(f"  #{r['station']:02d} {r['true_stage']:>8} -> {r['predicted_stage']:<8} {r['confidence_pct']:5.1f}% "
                 f"{'OK ' if r['correct'] else 'ERR'} {r['zone']:<6} hab={r['habitat_score']:5.1f}")
    (fin / "mission_report.txt").write_text("\n".join(L), encoding="utf-8")

    # ---- LIVE replay dashboard (same look as the reference monitor page)
    live_path = out / "live_dashboard.html"
    try:
        last_f = max([mm["frame"] for mm in motion_log] + [1])
        traj = sample_boat_trajectory(scene, robot_components, last_f) if (scene is not None and robot_components is not None) else None
        build_live_dashboard(live_path, rows, motion_log, scan_results, egg_data, mission, cm, per,
                             out / "scans", traj=traj, pens=pens, boat_log=boat_log)
        print("[FINAL] live dashboard:", live_path)
    except Exception as exc:
        import traceback; traceback.print_exc(); print("[FINAL] live dashboard failed:", repr(exc))

    # ---- HTML dashboard
    def tbl(head, body):
        return "<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + \
               "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in body) + "</table>"
    card = lambda k, v: f'<div class="card"><div class="v">{v}</div><div class="k">{k}</div></div>'
    st_rows = []
    for r in rows:
        img = f'<img src="../scans/{r["live_image"]}" width="72">' if r["live_image"] else ""
        st_rows.append([r["station"], r["true_stage"], r["predicted_stage"], f'{r["confidence_pct"]}%',
                        "yes" if r["correct"] else "<b>NO</b>", f'<span class="z {r["zone"]}">{r["zone"]}</span>',
                        r["habitat_score"], r["temperature_C"], r["pH"], r["oxygen_mgL"],
                        round(r.get("camera_distance_m", 0), 2), round(r.get("stability_pct", 0), 0),
                        round(r.get("err_before_deg", 0), 3), round(r.get("err_after_deg", 0), 3), img])
    graphs = "".join(f'<img class="g" src="../arm_analysis/{Path(p).name}">' for p in graph_files)
    doc = f"""<!doctype html><meta charset="utf-8"><title>Zebrafish inspection report</title>
<style>body{{font-family:Arial;margin:24px;background:#f6f8fa}}h2{{margin-top:34px}}
.cards{{display:flex;flex-wrap:wrap;gap:12px}}.card{{background:#fff;border-radius:8px;padding:14px 20px;box-shadow:0 1px 3px #0002}}
.v{{font-size:26px;font-weight:bold}}.k{{color:#666;font-size:12px}}table{{border-collapse:collapse;background:#fff;font-size:13px}}
td,th{{border:1px solid #ddd;padding:5px 8px;text-align:center}}th{{background:#eaeef2}}
.z{{padding:2px 8px;border-radius:10px;color:#fff}}.GREEN{{background:#2ca02c}}.YELLOW{{background:#e6b800}}.RED{{background:#d62728}}
img.g{{width:48%;margin:4px}}embed,img.s{{max-width:48%}}</style>
<h1>Autonomous zebrafish egg inspection - results</h1>
<div class="cards">{card("stations", mission["stations"])}{card("accuracy", f'{c["accuracy"]:.1f}%')}{card("macro F1", f'{c["macro_f1"]:.2f}')}
{card("mean habitat", mission["mean_habitat"])}{card("camera stable", f'{m["stable_pct"]}%')}{card("valid scans", f'{m["valid_scan_pct"]}%')}
{card("error reduction", f'{a["mean_error_reduction_pct"]}%')}{card("max torque", f'{a["max_torque_nm"]} Nm')}{card("mission time", f'{mission["mission_time_s"]} s')}</div>
<h2>Stations</h2>{tbl(["#","true","predicted","conf","correct","zone","habitat","T C","pH","O2","cam dist m","stable %","err before deg","err after deg","camera"], st_rows)}
<h2>Mission map &amp; classification</h2><img class="s" src="station_map.svg"> <img class="s" src="confusion_matrix.svg">
<h2>Per-class metrics</h2>{tbl(["class","precision","recall","F1","support"], [[p["cls"], f'{p["precision"]:.2f}', f'{p["recall"]:.2f}', f'{p["f1"]:.2f}', p["support"]] for p in per])}
<h2>Habitat</h2><img class="s" src="zone_counts.svg"> <img class="s" src="habitat_scores.svg">
<h2>Arm compensation (before vs after)</h2><img class="s" src="error_before.svg"> <img class="s" src="error_after.svg">
<h2>Arm &amp; camera time series</h2>{graphs or "<p>Install matplotlib in Blender's Python to embed the time-series graphs.</p>"}
<h2>Events</h2><pre>{html.escape(chr(10).join(f"{t:7.2f}s  {e}" for t, e in (events or [])))}</pre>"""
    (fin / "index.html").write_text(doc, encoding="utf-8")

    # ---- ZIP
    stamp = time.strftime("%Y%m%d_%H%M%S")
    zpath = out / f"Robot_Scan_Results_{stamp}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        if live_path.is_file(): z.write(live_path, live_path.name)
        for sub in ("final_report", "arm_analysis", "scans", "reports", "animation"):
            for p in (out / sub).rglob("*"):
                if p.is_file(): z.write(p, p.relative_to(out))
    print("\n" + "\n".join(L[:22]))
    print(f"\n[FINAL] dashboard : {fin / 'index.html'}")
    print(f"[FINAL] zip       : {zpath}  ({time.time() - t0:.1f}s)")
    # ---- LIVE LINK: serve the dashboard from inside Blender and keep timeline <-> site in two-way sync
    link_url = None
    if live_link and live_path.is_file() and scene is not None:
        try:
            import importlib, blender_live_bridge as BLB
            importlib.reload(BLB)
            link_url = BLB.start(live_path, scene=scene, port=port, open_browser=True).url
        except Exception as exc:
            import traceback; traceback.print_exc()
            print("[FINAL] live link not started (falling back to the static file):", repr(exc))
    if link_url is None:
        try:
            import webbrowser; webbrowser.open((live_path if live_path.is_file() else fin / "index.html").as_uri())
        except Exception:
            pass
    return dict(report=report, zip=str(zpath), dashboard=str(live_path), live_url=link_url)


# ----------------------------------------------------------------------------------------------
# PASTE THIS AT THE VERY END OF YOUR MAIN SCRIPT (after ST.run_standout):
#
#   import sys
#   sys.path.insert(0, r"C:\Users\pantu\OneDrive\Pictures\Desktop\airoboticspro")   # folder with this file
#   import importlib, robot_final_report as RFR; importlib.reload(RFR)
#   RFR.build_final_package(OUTPUT_DIR, scan_results, joint_motion_log, egg_data, arm_events,
#                           scene=scene, robot_components=robot_components, render_images=True)
# ----------------------------------------------------------------------------------------------
