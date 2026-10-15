"""
blender_live_bridge.py  -  two-way live link between the Blender timeline and live_dashboard.html

Put this file next to robot_final_report.py (same folder as robot_code.py's helpers).

HOW IT WORKS
  * A tiny HTTP server runs in a background thread INSIDE Blender (127.0.0.1 only, stdlib only):
        GET  /             -> live_dashboard.html (re-read from disk on every request)
        GET  /events       -> Server-Sent-Events stream of Blender state {frame, playing, speed, ...}
        GET  /state        -> same state, one-shot JSON
        POST /cmd          -> {"cmd": "play|pause|toggle|reset|seek|speed", ...}
  * Everything that touches `bpy` happens on Blender's MAIN thread through bpy.app.timers
    (bpy is not thread-safe). The server thread only reads a shared state dict and queues commands.
  * Blender -> site : press Space / drag the timeline / scrub in Blender  => the site follows (same frame).
  * Site -> Blender : PLAY / PAUSE / RESET / NEXT STATION / SEEK / SPEED on the site => Blender obeys.
  * Both run in parallel because the dashboard has no clock of its own while linked: it renders
    whatever frame Blender reports, so the two can never drift apart.

USAGE (already wired into robot_final_report.build_final_package):
    import blender_live_bridge as BLB
    BLB.start(r".../live_dashboard.html", scene=bpy.context.scene)   # opens the browser
    BLB.stop()
"""
import json
import queue
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

try:
    import bpy
    from bpy.app.handlers import persistent
except Exception:                      # allows importing / unit-testing outside Blender
    bpy = None

    def persistent(fn):
        return fn

NS_KEY = "zebrafish_live_bridge"       # slot in bpy.app.driver_namespace -> survives script re-runs
TICK = 1.0 / 30.0                      # main-thread poll interval (s)
DEFAULT_PORT = 8765


class Bridge:
    def __init__(self, html_path, host="127.0.0.1", port=DEFAULT_PORT, loop_playback=False):
        self.html_path = Path(html_path)
        self.host, self.port = host, port
        self.loop_playback = loop_playback
        self.cv = threading.Condition()
        self.cmds = queue.Queue()
        self.state = dict(seq=0, frame=1.0, playing=False, speed=1.0, frame_start=1, frame_end=1,
                          fps=24, t=time.time())
        self.running = False
        self.httpd = None
        self.thread = None
        self._saved = {}               # scene settings we change (restored in stop())
        self._speed = 1.0
        self._last_sig = None
        self._last_beat = 0.0

    # ------------------------------------------------------------------ Blender side (MAIN thread)
    def _scene(self):
        return bpy.context.scene

    def _is_playing(self):
        try:
            for w in bpy.context.window_manager.windows:
                if w.screen.is_animation_playing:
                    return True
        except Exception:
            pass
        return False

    def _override(self):
        """Context override so screen.animation_play works from a timer."""
        wm = bpy.context.window_manager
        for w in wm.windows:
            area = next((a for a in w.screen.areas if a.type in ("VIEW_3D", "DOPESHEET_EDITOR", "TIMELINE")), None)
            if area is None and w.screen.areas:
                area = w.screen.areas[0]
            if area is not None:
                return dict(window=w, screen=w.screen, area=area)
        return None

    def _toggle_play(self):
        ov = self._override()
        if ov is None:
            bpy.ops.screen.animation_play()
        elif hasattr(bpy.context, "temp_override"):          # Blender 3.2+
            with bpy.context.temp_override(**ov):
                bpy.ops.screen.animation_play()
        else:                                                # older Blender
            bpy.ops.screen.animation_play(ov)

    def _set_playing(self, want):
        if want == self._is_playing():
            return
        sc = self._scene()
        if want and sc.frame_current >= sc.frame_end:        # finished -> start again from the beginning
            sc.frame_set(sc.frame_start)
        self._toggle_play()

    def _apply_speed(self, v):
        """Blender plays at fps / fps_base, so speed v  ->  fps_base = 1/v (restored in stop())."""
        v = max(0.1, min(16.0, float(v)))
        sc = self._scene()
        sc.render.fps_base = 1.0 / v
        try:
            sc.sync_mode = "FRAME_DROP" if v > 1.0 else "NONE"   # keep true speed if the viewport is slow
        except Exception:
            pass
        self._speed = v

    def _apply(self, c):
        sc = self._scene()
        cmd = str(c.get("cmd", "")).lower()
        if cmd == "play":
            self._set_playing(True)
        elif cmd == "pause":
            self._set_playing(False)
        elif cmd == "toggle":
            self._set_playing(not self._is_playing())
        elif cmd == "reset":                                  # stop + rewind
            self._set_playing(False)
            sc.frame_set(sc.frame_start)
        elif cmd == "seek":
            f = max(sc.frame_start, min(sc.frame_end, int(round(float(c.get("frame", sc.frame_start))))))
            sc.frame_set(f)
        elif cmd == "speed":
            self._apply_speed(c.get("value", 1.0))

    def _tick(self):
        if not self.running:
            return None                                       # unregisters the timer
        try:
            while True:
                try:
                    c = self.cmds.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._apply(c)
                except Exception:
                    traceback.print_exc()
            sc = self._scene()
            playing = self._is_playing()
            if playing and not self.loop_playback and sc.frame_current >= sc.frame_end:
                self._set_playing(False)                      # stop on the last frame (site shows MISSION COMPLETE)
                playing = False
            self._publish(sc, playing)
        except Exception:
            traceback.print_exc()
        return TICK

    def _publish(self, sc=None, playing=None, force=False):
        sc = sc or self._scene()
        playing = self._is_playing() if playing is None else playing
        sig = (sc.frame_current, playing, round(self._speed, 3), sc.frame_start, sc.frame_end)
        now = time.time()
        if sig == self._last_sig and not force and now - self._last_beat < 1.0:
            return
        self._last_sig, self._last_beat = sig, now
        with self.cv:
            s = self.state
            s.update(seq=s["seq"] + 1, frame=float(sc.frame_current) + float(getattr(sc, "frame_subframe", 0.0)),
                     playing=bool(playing), speed=self._speed, frame_start=int(sc.frame_start),
                     frame_end=int(sc.frame_end), fps=int(sc.render.fps), t=now)
            self.cv.notify_all()

    # ------------------------------------------------------------------ HTTP side (server threads)
    def _make_handler(bridge):
        class H(BaseHTTPRequestHandler):
            server_version = "ZebrafishBridge/1.0"

            def log_message(self, *a):
                pass

            def _origin_ok(self):
                o = self.headers.get("Origin")
                return (not o) or o == "null" or o.startswith("http://127.0.0.1") or o.startswith("http://localhost")

            def _cors(self):
                o = self.headers.get("Origin")
                self.send_header("Access-Control-Allow-Origin", o if o else "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Vary", "Origin")

            def _json(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_OPTIONS(self):
                self.send_response(204)
                self._cors()
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                path = urlparse(self.path).path
                if not self._origin_ok():
                    return self._json(403, {"ok": False})
                if path in ("/", "/index.html", "/live_dashboard.html"):
                    try:
                        b = bridge.html_path.read_bytes()
                    except Exception as exc:
                        return self._json(404, {"ok": False, "error": repr(exc)})
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    self.wfile.write(b)
                elif path == "/state":
                    with bridge.cv:
                        self._json(200, dict(bridge.state))
                elif path == "/events":
                    self.send_response(200)
                    self._cors()
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    last = -1
                    try:
                        while bridge.running:
                            with bridge.cv:
                                if bridge.state["seq"] == last:
                                    bridge.cv.wait(1.0)
                                s = dict(bridge.state)
                            if s["seq"] != last:
                                last = s["seq"]
                                self.wfile.write(("data: " + json.dumps(s) + "\n\n").encode())
                            else:
                                self.wfile.write(b": hb\n\n")        # heartbeat, also detects closed tabs
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                        pass
                else:
                    self._json(404, {"ok": False})

            def do_POST(self):
                if not self._origin_ok():
                    return self._json(403, {"ok": False})
                if urlparse(self.path).path != "/cmd":
                    return self._json(404, {"ok": False})
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    c = json.loads(self.rfile.read(n).decode() or "{}")
                    if not isinstance(c, dict):
                        raise ValueError("body must be an object")
                except Exception as exc:
                    return self._json(400, {"ok": False, "error": repr(exc)})
                bridge.cmds.put(c)
                self._json(200, {"ok": True})
        return H

    # ------------------------------------------------------------------ lifecycle
    def start(self, scene=None, open_browser=True):
        sc = scene or self._scene()
        self._saved = dict(fps_base=sc.render.fps_base, sync_mode=getattr(sc, "sync_mode", None))
        self._speed = 1.0 / (sc.render.fps_base or 1.0)
        handler = self._make_handler()
        last_exc = None
        for p in range(self.port, self.port + 20):               # first free port
            try:
                self.httpd = ThreadingHTTPServer((self.host, p), handler)
                self.port = p
                break
            except OSError as exc:
                last_exc = exc
        if self.httpd is None:
            raise RuntimeError(f"no free port near {self.port}: {last_exc!r}")
        self.httpd.daemon_threads = True
        self.running = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="zebrafish-bridge", daemon=True)
        self.thread.start()
        self._publish(sc, force=True)
        bpy.app.timers.register(self._tick, first_interval=TICK, persistent=True)
        bpy.app.handlers.frame_change_post.append(self._on_frame)
        bpy.app.handlers.load_pre.append(self._on_load)
        url = self.url
        print(f"[LIVE LINK] dashboard  : {url}")
        print("[LIVE LINK] Blender <-> site are now in sync (Space in Blender = play on site; site buttons drive Blender)")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        return url

    @property
    def url(self):
        return f"http://{self.host}:{self.port}/"

    # frame handler: pushes EVERY frame Blender plays (timer alone could miss frames at high speed)
    @persistent
    def _on_frame(self, *a):
        try:
            if self.running:
                self._publish(force=False)
        except Exception:
            pass

    @persistent
    def _on_load(self, *a):
        self.stop()

    def stop(self):
        self.running = False
        with self.cv:
            self.cv.notify_all()
        try:
            if self.httpd:
                threading.Thread(target=self.httpd.shutdown, daemon=True).start()
                self.httpd.server_close()
        except Exception:
            pass
        self.httpd = None
        for lst, fn in ((bpy.app.handlers.frame_change_post, self._on_frame), (bpy.app.handlers.load_pre, self._on_load)):
            try:
                lst.remove(fn)
            except ValueError:
                pass
        try:
            if bpy.app.timers.is_registered(self._tick):
                bpy.app.timers.unregister(self._tick)
        except Exception:
            pass
        try:                                                     # put the scene back the way we found it
            sc = self._scene()
            if "fps_base" in self._saved:
                sc.render.fps_base = self._saved["fps_base"]
            if self._saved.get("sync_mode"):
                sc.sync_mode = self._saved["sync_mode"]
        except Exception:
            pass
        print("[LIVE LINK] stopped")


def start(html_path, scene=None, port=DEFAULT_PORT, open_browser=True, loop_playback=False):
    """Start (or restart) the bridge. Safe to call again on every script run."""
    ns = bpy.app.driver_namespace
    old = ns.get(NS_KEY)
    if old is not None:
        try:
            old.stop()
        except Exception:
            pass
    b = Bridge(html_path, port=port, loop_playback=loop_playback)
    ns[NS_KEY] = b
    b.start(scene=scene, open_browser=open_browser)
    return b


def stop():
    b = bpy.app.driver_namespace.pop(NS_KEY, None)
    if b is not None:
        b.stop()
