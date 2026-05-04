"""
Interactive 3D OBB Segment Viewer
==================================
Load dense PLY + obb_results.json, isolate detected segments,
and drag them around in 3D with your mouse.

Controls:
  Left-click + drag on a segment  →  move it (XY plane)
  Ctrl + drag                     →  move along Z (depth) axis
  Right-click + drag              →  rotate scene camera
  Scroll wheel                    →  zoom
  R                               →  reset all segments
  S                               →  save positions to moved_obbs.json
  Q / Escape                      →  quit

Usage:
  python interactive_viewer.py --ply dense_reconstruction/fuse.ply --json obb_results.json
"""

import argparse
import json
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from pathlib import Path

# ─── Colour palette ───────────────────────────────────────────────────────────
LABEL_COLORS = {
    "vga":  [1.0, 0.20, 0.20],
    "lan":  [0.20, 1.0, 0.20],
    "usb":  [0.30, 0.60, 1.0],
    "hdmi": [1.0, 0.85, 0.10],
    "dp":   [1.0, 0.20, 1.0],
}
DEFAULT_COLOR  = [1.0, 0.55, 0.10]
SCENE_GRAY     = [0.30, 0.30, 0.30]
POINT_SIZE_BG  = 1.5
POINT_SIZE_SEG = 4.0


# ─── Helpers ──────────────────────────────────────────────────────────────────

def base_label(label: str) -> str:
    parts = label.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else label


def color_for(label):
    return LABEL_COLORS.get(base_label(label), DEFAULT_COLOR)


def obb_corners(center, extent, rotation_rows):
    axes  = np.array(rotation_rows).T
    half  = np.array(extent) / 2.0
    signs = np.array([
        [-1,-1,-1],[+1,-1,-1],[+1,+1,-1],[-1,+1,-1],
        [-1,-1,+1],[+1,-1,+1],[+1,+1,+1],[-1,+1,+1],
    ], dtype=float)
    return np.array(center) + (signs * half) @ axes.T


def make_wire(center, extent, rotation_rows, color):
    corners = obb_corners(center, extent, rotation_rows)
    edges   = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
                [0,4],[1,5],[2,6],[3,7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines  = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector([color] * len(edges))
    return ls


def extract_segment(all_pts, all_cols, center, extent, rotation_rows, pad=1.4):
    axes  = np.array(rotation_rows).T
    half  = np.array(extent) / 2.0 * pad
    local = (all_pts - np.array(center)) @ axes
    mask  = np.all(np.abs(local) <= half, axis=1)
    cols  = all_cols[mask] if all_cols is not None else None
    return all_pts[mask], cols


def camera_axes(view_mat):
    """
    Derive camera world-space position, right, up, forward from
    Open3D's view matrix (row-major 4x4, world→camera).
    """
    V       = np.array(view_mat)
    R       = V[:3, :3]
    t       = V[:3,  3]
    right   =  R[0]
    up      =  R[1]
    forward = -R[2]
    cam_pos = -R.T @ t
    return cam_pos, right, up, forward


# ─── Viewer ───────────────────────────────────────────────────────────────────

class OBBViewer:

    def __init__(self, ply_path: Path, json_path: Path):
        # ── Load data ────────────────────────────────────────────────────────
        print("Loading point cloud …")
        pcd           = o3d.io.read_point_cloud(str(ply_path))
        self.all_pts  = np.asarray(pcd.points)
        self.all_cols = np.asarray(pcd.colors) if pcd.has_colors() else None
        print(f"  {len(self.all_pts):,} points")

        print("Loading OBBs …")
        with open(json_path) as f:
            obb_data = json.load(f)

        # ── Build segment objects ─────────────────────────────────────────────
        self.segments = []
        for entry in obb_data:
            label    = entry["entity"]
            obb      = entry["obb"]
            center   = np.array(obb["center"], dtype=float)
            extent   = obb["extent"]
            rotation = obb["rotation"]
            color    = color_for(label)

            seg_pts, seg_cols = extract_segment(
                self.all_pts, self.all_cols, center, extent, rotation)

            pcd_obj = o3d.geometry.PointCloud()
            pcd_obj.points = o3d.utility.Vector3dVector(seg_pts)
            if seg_cols is not None:
                pcd_obj.colors = o3d.utility.Vector3dVector(seg_cols)
            else:
                pcd_obj.paint_uniform_color(color)

            self.segments.append(dict(
                label       = label,
                color       = color,
                center      = center.copy(),
                orig_center = center.copy(),
                extent      = extent,
                rotation    = rotation,
                pcd_obj     = pcd_obj,
                wire_obj    = make_wire(center.tolist(), extent, rotation, color),
                offset      = np.zeros(3),
            ))

        print(f"  {len(self.segments)} segments ready")

        # ── GUI ──────────────────────────────────────────────────────────────
        app = gui.Application.instance
        app.initialize()

        self.win = app.create_window(
            "OBB Interactive Viewer  —  left-drag segments to move", 1440, 900)
        em = self.win.theme.font_size

        self.sw = gui.SceneWidget()
        self.sw.scene = rendering.Open3DScene(self.win.renderer)
        self.sw.scene.set_background([0.07, 0.07, 0.09, 1.0])

        self.panel = gui.Vert(int(0.5 * em), gui.Margins(em, em, em, em))
        self._build_panel(em)

        self.win.add_child(self.sw)
        self.win.add_child(self.panel)
        self.win.set_on_layout(self._on_layout)

        # Input state
        self._selected    = None
        self._drag_active = False
        self._last_xy     = None
        self._ctrl        = False

        self.sw.set_on_mouse(self._on_mouse)
        self.sw.set_on_key(self._on_key)

        self._populate_scene()
        self._fit_camera()

    # ── Scene helpers ─────────────────────────────────────────────────────────

    def _mat_pts(self, size=POINT_SIZE_BG):
        m = rendering.MaterialRecord()
        m.shader = "defaultUnlit"
        m.point_size = size
        return m

    def _mat_wire(self):
        m = rendering.MaterialRecord()
        m.shader = "unlitLine"
        m.line_width = 2.5
        return m

    def _populate_scene(self):
        sc = self.sw.scene
        sc.clear_geometry()

        bg = o3d.geometry.PointCloud()
        bg.points = o3d.utility.Vector3dVector(self.all_pts)
        bg.paint_uniform_color(SCENE_GRAY)
        sc.add_geometry("__bg__", bg, self._mat_pts(POINT_SIZE_BG))

        for i, seg in enumerate(self.segments):
            sc.add_geometry(f"seg_{i}",  seg["pcd_obj"],  self._mat_pts(POINT_SIZE_SEG))
            sc.add_geometry(f"wire_{i}", seg["wire_obj"], self._mat_wire())

    def _refresh(self, i):
        sc = self.sw.scene
        sc.remove_geometry(f"seg_{i}")
        sc.remove_geometry(f"wire_{i}")
        sc.add_geometry(f"seg_{i}",  self.segments[i]["pcd_obj"],  self._mat_pts(POINT_SIZE_SEG))
        sc.add_geometry(f"wire_{i}", self.segments[i]["wire_obj"], self._mat_wire())

    def _fit_camera(self):
        bb = self.sw.scene.bounding_box
        self.sw.setup_camera(60.0, bb, bb.get_center())

    # ── Panel ─────────────────────────────────────────────────────────────────

    def _build_panel(self, em):
        def lbl(text, r=0.8, g=0.8, b=0.8):
            l = gui.Label(text)
            l.text_color = gui.Color(r, g, b)
            return l

        self.panel.add_child(lbl("OBB Viewer", 0.95, 0.95, 0.95))
        self.panel.add_child(lbl("─" * 26, 0.35, 0.35, 0.35))
        self.status = lbl("Click a segment to select", 0.5, 0.9, 1.0)
        self.panel.add_child(self.status)
        self.panel.add_child(lbl(" "))
        self.panel.add_child(lbl("Segments:", 0.65, 0.65, 0.65))

        self.seg_labels = []
        for seg in self.segments:
            c = seg["color"]
            l = lbl(f"  {seg['label']}", c[0], c[1], c[2])
            self.panel.add_child(l)
            self.seg_labels.append(l)

        self.panel.add_child(lbl(" "))
        self.panel.add_child(lbl("─" * 26, 0.35, 0.35, 0.35))

        for line in [
            "Left-drag   move XY",
            "Ctrl+drag   move Z",
            "Right-drag  rotate view",
            "Scroll      zoom",
            "R           reset all",
            "S           save JSON",
            "Q / Esc     quit",
        ]:
            self.panel.add_child(lbl(line, 0.5, 0.5, 0.5))

        self.panel.add_child(lbl(" "))

        b1 = gui.Button("Reset All  [R]")
        b1.set_on_clicked(self._reset_all)
        self.panel.add_child(b1)

        b2 = gui.Button("Save JSON  [S]")
        b2.set_on_clicked(self._save)
        self.panel.add_child(b2)

    def _on_layout(self, _ctx):
        r  = self.win.content_rect
        pw = 210
        self.sw.frame    = gui.Rect(r.x, r.y, r.width - pw, r.height)
        self.panel.frame = gui.Rect(r.x + r.width - pw, r.y, pw, r.height)

    # ── Ray–OBB picking ───────────────────────────────────────────────────────

    def _pick(self, fx, fy):
        cam   = self.sw.scene.camera
        V     = cam.get_view_matrix()
        cam_pos, right, up, forward = camera_axes(V)

        fw  = self.sw.frame.width
        fh  = self.sw.frame.height
        fov = np.radians(cam.get_field_of_view())
        hh  = np.tan(fov / 2.0)
        hw  = hh * fw / fh

        ndc_x = (fx / fw) * 2.0 - 1.0
        ndc_y = 1.0 - (fy / fh) * 2.0

        ray_d  = forward + right * (ndc_x * hw) + up * (ndc_y * hh)
        ray_d /= np.linalg.norm(ray_d) + 1e-12

        best_t, best_i = np.inf, None

        for i, seg in enumerate(self.segments):
            axes = np.array(seg["rotation"]).T
            half = np.array(seg["extent"]) / 2.0 * 1.6
            oc   = cam_pos - seg["center"]

            t_min, t_max = -np.inf, np.inf
            hit = True
            for a in range(3):
                ax = axes[:, a]
                e  = float(np.dot(ax, oc))
                f  = float(np.dot(ax, ray_d))
                if abs(f) > 1e-9:
                    t1 = (-e - half[a]) / f
                    t2 = (-e + half[a]) / f
                    if t1 > t2: t1, t2 = t2, t1
                    t_min = max(t_min, t1)
                    t_max = min(t_max, t2)
                    if t_max < t_min:
                        hit = False; break
                elif abs(e) > half[a]:
                    hit = False; break

            if hit and t_max >= 0 and t_min < best_t:
                best_t, best_i = t_min, i

        return best_i

    # ── Drag ─────────────────────────────────────────────────────────────────

    def _world_delta(self, dx, dy, ref_pt):
        cam   = self.sw.scene.camera
        V     = cam.get_view_matrix()
        cam_pos, right, up, forward = camera_axes(V)

        dist  = float(np.linalg.norm(ref_pt - cam_pos))
        fov   = np.radians(cam.get_field_of_view())
        fh    = self.sw.frame.height
        scale = dist * 2.0 * np.tan(fov / 2.0) / fh

        if self._ctrl:
            return forward * (-dy * scale)
        return right * (dx * scale) + up * (-dy * scale)

    def _translate(self, i, delta):
        seg = self.segments[i]
        seg["offset"] += delta
        seg["center"] += delta

        pts = np.asarray(seg["pcd_obj"].points) + delta
        seg["pcd_obj"].points = o3d.utility.Vector3dVector(pts)

        wpts = np.asarray(seg["wire_obj"].points) + delta
        seg["wire_obj"].points = o3d.utility.Vector3dVector(wpts)

        self._refresh(i)

    # ── Mouse / Key ───────────────────────────────────────────────────────────

    def _on_mouse(self, event):
        ET = gui.MouseEvent.Type
        MB = gui.MouseButton
        CR = gui.SceneWidget.EventCallbackResult

        if event.type == ET.BUTTON_DOWN and event.is_button_down(MB.LEFT):
            fx = event.x - self.sw.frame.x
            fy = event.y - self.sw.frame.y
            idx = self._pick(fx, fy)
            if idx is not None:
                self._selected    = idx
                self._drag_active = True
                self._last_xy     = (event.x, event.y)
                self._highlight(idx)
                self.status.text  = f"Dragging: {self.segments[idx]['label']}"
                return CR.CONSUMED

        elif event.type == ET.DRAG and self._drag_active and self._selected is not None:
            dx = event.x - self._last_xy[0]
            dy = event.y - self._last_xy[1]
            self._last_xy = (event.x, event.y)
            ref = self.segments[self._selected]["center"].copy()
            self._translate(self._selected, self._world_delta(dx, dy, ref))
            return CR.CONSUMED

        elif event.type == ET.BUTTON_UP and self._drag_active:
            self._drag_active = False
            if self._selected is not None:
                self.status.text = f"Placed: {self.segments[self._selected]['label']}"
            return CR.CONSUMED

        return CR.IGNORED

    def _on_key(self, event):
        KT = gui.KeyEvent.Type
        KN = gui.KeyName
        CR = gui.SceneWidget.EventCallbackResult

        if event.type == KT.DOWN:
            if event.key in (KN.LEFT_CONTROL, KN.RIGHT_CONTROL):
                self._ctrl = True
                self.status.text = "Z-axis mode (Ctrl held)"
            elif event.key == KN.R:
                self._reset_all()
            elif event.key == KN.S:
                self._save()
            elif event.key in (KN.Q, KN.ESCAPE):
                gui.Application.instance.quit()
        elif event.type == KT.UP:
            if event.key in (KN.LEFT_CONTROL, KN.RIGHT_CONTROL):
                self._ctrl = False
                self.status.text = "XY mode"

        return CR.IGNORED

    # ── Highlight ─────────────────────────────────────────────────────────────

    def _highlight(self, idx):
        for i, lbl in enumerate(self.seg_labels):
            seg = self.segments[i]
            c   = seg["color"]
            if i == idx:
                lbl.text_color = gui.Color(1.0, 1.0, 1.0)
                lbl.text       = f"▶ {seg['label']}"
            else:
                lbl.text_color = gui.Color(c[0], c[1], c[2])
                lbl.text       = f"  {seg['label']}"

    # ── Reset / Save ──────────────────────────────────────────────────────────

    def _reset_all(self):
        for i, seg in enumerate(self.segments):
            if np.any(seg["offset"] != 0):
                self._translate(i, -seg["offset"].copy())
                seg["offset"] = np.zeros(3)
        self._selected = None
        for i, lbl in enumerate(self.seg_labels):
            c = self.segments[i]["color"]
            lbl.text_color = gui.Color(c[0], c[1], c[2])
            lbl.text       = f"  {self.segments[i]['label']}"
        self.status.text = "All segments reset"

    def _save(self):
        out = [{"entity": s["label"],
                "obb": {"center":   s["center"].tolist(),
                        "extent":   s["extent"],
                        "rotation": s["rotation"]}}
               for s in self.segments]
        p = Path("moved_obbs.json")
        with open(p, "w") as f:
            json.dump(out, f, indent=2)
        self.status.text = f"Saved → {p}"
        print(f"Saved → {p}")

    def run(self):
        gui.Application.instance.run()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Interactive 3D OBB Segment Viewer")
    ap.add_argument("--ply",  default="dense_reconstruction/fuse.ply",
                    help="Path to dense PLY point cloud")
    ap.add_argument("--json", default="obb_results.json",
                    help="Path to OBB results JSON")
    args = ap.parse_args()

    ply  = Path(args.ply)
    jsn  = Path(args.json)

    if not ply.exists():
        print(f"[ERROR] PLY not found:  {ply}"); return
    if not jsn.exists():
        print(f"[ERROR] JSON not found: {jsn}"); return

    OBBViewer(ply, jsn).run()


if __name__ == "__main__":
    main()