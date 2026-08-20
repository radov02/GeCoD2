"""Three-panel viewer for IOCfish5k: RGB and segmap on top, depth below.
Bboxes come from the matching XML, orange = fallback square. With
--GECO2_train / --GECO2_inference (plus --tag) it browses the pre-rendered
*_pred.png visuals instead. Keys: A/D = prev/next, Q/Esc = quit,
ctrl+scroll = zoom, right-drag = pan, left-click = mirror, hover = info."""

import argparse
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, TextBox

BASE_DIR    = Path(__file__).resolve().parent
RGB_DIR     = BASE_DIR / "latest" / "IOCfish5k" / "images"
DEPTH_DIR   = BASE_DIR / "dataset" / "IOCfish5k-D" / "color"
SEGMAP_DIR  = BASE_DIR / "latest" / "IOCfish5k" / "segment_maps"
BBOX_DIR    = BASE_DIR / "latest" / "IOCfish5k" / "bbox_annotations"
# Default GECO2 model output dir (sibling of IOCfish5kDataset under GECO2/).
MODELS_DIR  = BASE_DIR.parent / "models"

BBOX_COLOR          = (0, 1, 0) # green
BBOX_SEL_COLOR      = (1, 1, 0) # yellow, clicked/mirrored
BBOX_FALLBACK_COLOR = (1, 0.55, 0) # orange, cascade=fallback_square
POINT_COLOR         = (1, 0, 0)
BBOX_LW             = 1.0
ZOOM_FACTOR         = 1.2
HOVER_MIN_INTERVAL  = 1.0 / 30.0 # cap hover redraws at 30 Hz


def parse_objects(xml_path: Path):
    """One dict per <object> with optional 'bbox' / 'point' / 'source' keys."""
    if not xml_path.exists():
        return []
    objects = []
    root = ET.parse(xml_path).getroot()
    for obj in root.findall("object"):
        item = {}
        b = obj.find("bndbox")
        if b is not None:
            try:
                item["bbox"] = (
                    int(float(b.findtext("xmin"))),
                    int(float(b.findtext("ymin"))),
                    int(float(b.findtext("xmax"))),
                    int(float(b.findtext("ymax"))),
                )
            except (TypeError, ValueError):
                pass
        p = obj.find("point")
        if p is not None:
            try:
                item["point"] = (
                    int(float(p.findtext("x"))),
                    int(float(p.findtext("y"))),
                )
            except (TypeError, ValueError):
                pass
        s = obj.find("bbox_source")
        if s is not None:
            src = {
                "cascade": (s.findtext("cascade") or "").strip(),
                "modality": (s.findtext("modality") or "").strip(),
                "sharpened": (s.findtext("sharpened") or "").strip().lower() == "true",
            }
            # Near-miss diagnostics (SAM3_hpc_annotation, fallback_square points).
            for key in ("near_miss_modality", "near_miss_cascade",
                        "near_miss_reason"):
                val = s.findtext(key)
                if val is not None and val.strip() != "":
                    src[key] = val.strip()
            for key in ("near_miss_score", "near_miss_floor"):
                val = s.findtext(key)
                if val is not None and val.strip() != "":
                    try:
                        src[key] = float(val)
                    except ValueError:
                        pass
            item["source"] = src
        if item:
            objects.append(item)
    return objects


def load_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_segmap(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_to(img, target_hw):
    if img is None:
        return None
    th, tw = target_hw
    if img.shape[:2] == (th, tw):
        return img
    interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_AREA
    return cv2.resize(img, (tw, th), interpolation=interp)


def _has_ctrl(event):
    key = getattr(event, "key", None) or ""
    # mpl gives 'control' for ctrl alone, 'ctrl+...' for combos
    return "control" in key or "ctrl" in key


class Viewer:
    def __init__(self, ids, visuals_dir=None, mode=None):
        """With visuals_dir set the RGB panel comes from <visuals_dir>/<id>_pred.png
        and XML overlays are skipped; mode only affects the window title."""
        self.ids = ids
        self._id_to_idx = {sid: i for i, sid in enumerate(ids)}
        self.idx = 0
        self.show_all = False
        self.selected = set() # object indices mirrored onto depth/segmap
        self.objects = []
        self._pan_state = None # active right-click pan, if any
        self.visuals_dir = visuals_dir
        self.mode = mode

        self.rgb_artists = []
        self.depth_artists = []
        self.segmap_artists = []
        self.rgb_rect_to_idx = {} # Rectangle artist -> object index
        self.any_rect_to_idx = {} # rect on any panel -> object index
        self._tooltip = None # matplotlib Annotation, lazily created
        self._hovered_idx = None
        self._last_hover_t = 0.0
        # hover hit-test cache per axes: (N, 5) of (x0, y0, x1, y1, area)
        self._rect_extents = {}
        self._rect_idx_arrays = {}

        # Layout: RGB | Segmap on top row, Depth full-width on bottom row.
        self.fig = plt.figure(figsize=(18, 12))
        gs = self.fig.add_gridspec(2, 2)
        ax_rgb   = self.fig.add_subplot(gs[0, 0])
        ax_seg   = self.fig.add_subplot(gs[0, 1], sharex=ax_rgb, sharey=ax_rgb)
        ax_depth = self.fig.add_subplot(gs[1, :], sharex=ax_rgb, sharey=ax_rgb)
        # Convention: self.axes[0]=RGB, [1]=Depth, [2]=Segmap.
        self.axes = [ax_rgb, ax_depth, ax_seg]

        self.fig.subplots_adjust(
            left=0.005, right=0.995, top=0.95, bottom=0.07,
            wspace=0.01, hspace=0.06,
        )

        # Bottom-of-figure controls.
        ax_prev = self.fig.add_axes([0.22, 0.015, 0.08, 0.04])
        ax_next = self.fig.add_axes([0.32, 0.015, 0.08, 0.04])
        ax_idlabel = self.fig.add_axes([0.43, 0.015, 0.04, 0.04])
        ax_id   = self.fig.add_axes([0.47, 0.015, 0.10, 0.04])
        ax_go   = self.fig.add_axes([0.58, 0.015, 0.05, 0.04])
        ax_all  = self.fig.add_axes([0.66, 0.015, 0.14, 0.04])
        self.btn_prev = Button(ax_prev, "Prev (A)")
        self.btn_next = Button(ax_next, "Next (D)")
        ax_idlabel.set_axis_off()
        ax_idlabel.text(1.0, 0.5, "ID:", ha="right", va="center", fontsize=10)
        self.txt_id   = TextBox(ax_id, "", initial="")
        self.btn_go   = Button(ax_go, "Go")
        self.btn_all  = Button(ax_all, "Show on all: OFF")
        self.btn_prev.on_clicked(lambda _e: self._step(-1))
        self.btn_next.on_clicked(lambda _e: self._step(+1))
        self.btn_all.on_clicked(lambda _e: self._toggle_show_all())
        self.txt_id.on_submit(self._jump_to_id)
        self.btn_go.on_clicked(lambda _e: self._jump_to_id(self.txt_id.text))

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.fig.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.fig.canvas.mpl_connect("button_release_event", self.on_button_release)

        self.render()

    # navigation
    def _step(self, delta):
        self.idx = (self.idx + delta) % len(self.ids)
        self.render()

    def _jump_to_id(self, text):
        """Jump to the image with this id, also trying common zero-pad widths.
        Unknown input just flashes the textbox."""
        wanted = (text or "").strip()
        if not wanted:
            return
        candidates = [wanted]
        if wanted.isdigit():
            for width in (4, 5, 6, 7, 8):
                padded = wanted.zfill(width)
                if padded != wanted:
                    candidates.append(padded)
        target = None
        for c in candidates:
            if c in self._id_to_idx:
                target = self._id_to_idx[c]
                break
        if target is None:
            # tint the textbox on a failed lookup
            try:
                orig = self.txt_id.color
                self.txt_id.color = (1.0, 0.85, 0.85)
                self.fig.canvas.draw_idle()

                def _restore():
                    self.txt_id.color = orig
                    self.fig.canvas.draw_idle()

                timer = self.fig.canvas.new_timer(interval=400)
                timer.single_shot = True
                timer.add_callback(_restore)
                timer.start()
            except Exception:
                pass
            return
        if target == self.idx:
            return
        self.idx = target
        self.render()

    def on_key(self, event):
        if event.key in ("right", "d"):
            self._step(+1)
        elif event.key in ("left", "a"):
            self._step(-1)
        elif event.key in ("q", "escape"):
            plt.close(self.fig)

    # zoom (ctrl+scroll, cursor-anchored)
    def on_scroll(self, event):
        if not _has_ctrl(event):
            return
        if event.inaxes not in self.axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        factor = 1.0 / ZOOM_FACTOR if event.button == "up" else ZOOM_FACTOR
        ax = event.inaxes
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        relx = (cx - x0) / (x1 - x0)
        rely = (cy - y0) / (y1 - y0)
        new_w = (x1 - x0) * factor
        new_h = (y1 - y0) * factor
        ax.set_xlim(cx - new_w * relx, cx + new_w * (1 - relx))
        ax.set_ylim(cy - new_h * rely, cy + new_h * (1 - rely))
        self.fig.canvas.draw_idle()

    # pan (right-drag, panels sync via shared axes)
    def on_button_press(self, event):
        if event.button != 3 or event.inaxes not in self.axes:
            return
        ax = event.inaxes
        ext = ax.get_window_extent()
        if ext.width <= 0 or ext.height <= 0:
            return
        self._pan_state = {
            "ax": ax,
            "press_xy_pixel": (event.x, event.y),
            "xlim": ax.get_xlim(),
            "ylim": ax.get_ylim(),
            "ax_w_px": ext.width,
            "ax_h_px": ext.height,
        }

    def on_motion(self, event):
        st = self._pan_state
        if st is None or event.x is None or event.y is None:
            return
        px, py = st["press_xy_pixel"]
        x0, x1 = st["xlim"]
        y0, y1 = st["ylim"]
        dx_data = (event.x - px) * (x1 - x0) / st["ax_w_px"]
        dy_data = (event.y - py) * (y1 - y0) / st["ax_h_px"]
        st["ax"].set_xlim(x0 - dx_data, x1 - dx_data)
        st["ax"].set_ylim(y0 - dy_data, y1 - dy_data)
        self.fig.canvas.draw_idle()

    def on_button_release(self, event):
        if event.button == 3:
            self._pan_state = None

    # hover tooltip with bbox_source info
    def _format_source(self, idx):
        if idx is None or idx >= len(self.objects):
            return None
        obj = self.objects[idx]
        src = obj.get("source")
        if not src:
            return f"#{idx}: (no <bbox_source> in XML)"
        lines = [
            f"#{idx}",
            f"cascade: {src.get('cascade', '?')}",
            f"modality: {src.get('modality', '?')}",
            f"sharpened: {'yes' if src.get('sharpened') else 'no'}",
        ]
        # For fallback_square entries, also show why the cascade gave up.
        cascade = src.get("cascade", "")
        if cascade.startswith("fallback_square"):
            reason = src.get("near_miss_reason")
            if reason is None:
                lines.append("near miss: (no diagnostic)")
            elif reason == "no_attempts":
                lines.append("near miss: no attempts (skipped by routing)")
            else:
                nm_mod = src.get("near_miss_modality", "?")
                nm_score = src.get("near_miss_score")
                nm_floor = src.get("near_miss_floor")
                nm_cascade = src.get("near_miss_cascade", "?")
                score_str = (
                    f"{nm_score:.3f}" if isinstance(nm_score, float) else "?"
                )
                floor_str = (
                    f"{nm_floor:.3f}" if isinstance(nm_floor, float) else "?"
                )
                lines.append(
                    f"near miss: {nm_mod} @ {nm_cascade}"
                )
                lines.append(
                    f"  score {score_str} / floor {floor_str}"
                )
                lines.append(f"  reason: {reason}")
        return "\n".join(lines)

    def _ensure_tooltip(self, ax):
        # recreate after an ax.clear(), or when it sits on another axes
        if self._tooltip is None or self._tooltip.axes is not ax:
            if self._tooltip is not None:
                try:
                    self._tooltip.remove()
                except (NotImplementedError, ValueError):
                    pass
            self._tooltip = ax.annotate(
                "",
                xy=(0, 0), xycoords="data",
                xytext=(12, -12), textcoords="offset points",
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                          ec="black", lw=0.5, alpha=0.92),
                annotation_clip=False, zorder=20,
            )
            self._tooltip.set_visible(False)
        return self._tooltip

    def on_hover(self, event):
        # don't fight the pan handler
        if self._pan_state is not None:
            return
        now = time.monotonic()
        if now - self._last_hover_t < HOVER_MIN_INTERVAL:
            return
        self._last_hover_t = now

        ax = event.inaxes
        if ax not in self.axes or event.xdata is None or event.ydata is None:
            self._hide_tooltip_if_visible()
            return

        extents = self._rect_extents.get(ax)
        idx_arr = self._rect_idx_arrays.get(ax)
        if extents is None or idx_arr is None or len(extents) == 0:
            self._hide_tooltip_if_visible()
            return

        x, y = event.xdata, event.ydata
        inside = (
            (extents[:, 0] <= x) & (x <= extents[:, 2])
            & (extents[:, 1] <= y) & (y <= extents[:, 3])
        )
        if not inside.any():
            self._hide_tooltip_if_visible()
            return
        # Smallest containing rect (so nested bboxes resolve to the inner one).
        candidate_pos = np.flatnonzero(inside)
        smallest = candidate_pos[np.argmin(extents[candidate_pos, 4])]
        hit_idx = int(idx_arr[smallest])

        tooltip = self._ensure_tooltip(ax)
        tooltip.xy = (x, y)
        changed = False
        if hit_idx != self._hovered_idx:
            tooltip.set_text(self._format_source(hit_idx) or "")
            self._hovered_idx = hit_idx
            changed = True
        if not tooltip.get_visible():
            tooltip.set_visible(True)
            changed = True
        if changed:
            self.fig.canvas.draw_idle()

    def _hide_tooltip_if_visible(self):
        if self._tooltip is not None and self._tooltip.get_visible():
            self._tooltip.set_visible(False)
            self._hovered_idx = None
            self.fig.canvas.draw_idle()

    # bbox selection
    def on_pick(self, event):
        # left-click toggles selection, right-click is pan
        if getattr(event, "mouseevent", None) is not None and event.mouseevent.button != 1:
            return
        rect = event.artist
        idx = self.rgb_rect_to_idx.get(rect)
        if idx is None:
            return
        if idx in self.selected:
            self.selected.remove(idx)
        else:
            self.selected.add(idx)
        self._refresh_overlays()

    def _toggle_show_all(self):
        self.show_all = not self.show_all
        self.btn_all.label.set_text(
            f"Show on all: {'ON' if self.show_all else 'OFF'}"
        )
        self._refresh_overlays()

    # rendering
    def _clear_overlays(self):
        for art_list in (self.rgb_artists, self.depth_artists, self.segmap_artists):
            for a in art_list:
                try:
                    a.remove()
                except (NotImplementedError, ValueError):
                    pass
            art_list.clear()
        self.rgb_rect_to_idx.clear()
        self.any_rect_to_idx.clear()
        self._rect_extents.clear()
        self._rect_idx_arrays.clear()
        self._hovered_idx = None
        if self._tooltip is not None:
            self._tooltip.set_visible(False)

    def _draw_bbox_with_line(self, ax, art_list, bbox, point, color, pickable=False,
                             obj_idx=None, line_color=None):
        """Draw a bbox plus a connector line from its top-right corner to point."""
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=BBOX_LW, edgecolor=color, facecolor="none",
            picker=pickable,
        )
        ax.add_patch(rect)
        art_list.append(rect)
        if obj_idx is not None:
            self.any_rect_to_idx[rect] = obj_idx
        if point is not None:
            cx, cy = point
            line, = ax.plot(
                [x2, cx], [y1, cy],
                color=line_color if line_color is not None else color,
                linewidth=BBOX_LW,
            )
            art_list.append(line)
        return rect

    def _refresh_overlays(self):
        """Rebuild bbox/point overlays based on current selection state."""
        self._clear_overlays()
        rgb_ax, depth_ax, seg_ax = self.axes

        # Center points are always shown on all three panels.
        all_points = [obj["point"] for obj in self.objects if "point" in obj]
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            for ax, art_list in (
                (rgb_ax,   self.rgb_artists),
                (depth_ax, self.depth_artists),
                (seg_ax,   self.segmap_artists),
            ):
                art_list.append(
                    ax.scatter(xs, ys, s=8, c=[POINT_COLOR], marker="o")
                )

        # Bboxes (with their connector lines): always on RGB, conditional elsewhere.
        for i, obj in enumerate(self.objects):
            if "bbox" not in obj:
                continue
            mirrored = self.show_all or (i in self.selected)
            cascade = (obj.get("source") or {}).get("cascade", "")
            is_fallback = cascade.startswith("fallback_square")
            if mirrored:
                color = BBOX_SEL_COLOR
            elif is_fallback:
                color = BBOX_FALLBACK_COLOR
            else:
                color = BBOX_COLOR
            point = obj.get("point")

            rect = self._draw_bbox_with_line(
                rgb_ax, self.rgb_artists, obj["bbox"], point, color, pickable=True,
                obj_idx=i,
            )
            self.rgb_rect_to_idx[rect] = i

            if mirrored:
                self._draw_bbox_with_line(
                    depth_ax, self.depth_artists, obj["bbox"], point, "black",
                    obj_idx=i, line_color=color,
                )
                self._draw_bbox_with_line(
                    seg_ax, self.segmap_artists, obj["bbox"], point, color,
                    obj_idx=i,
                )

        self._rebuild_hover_index()
        self.fig.canvas.draw_idle()

    def _rebuild_hover_index(self):
        """Cache the data-coord extent of every rect per axis for hover hit-tests."""
        self._rect_extents.clear()
        self._rect_idx_arrays.clear()
        for ax in self.axes:
            ext_rows = []
            idx_rows = []
            for art in ax.patches:
                idx = self.any_rect_to_idx.get(art)
                if idx is None:
                    continue
                x0, y0 = art.get_xy()
                w = art.get_width()
                h = art.get_height()
                x1 = x0 + w
                y1 = y0 + h
                # Normalise so x0<=x1, y0<=y1 (rect height can be negative).
                xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
                ya, yb = (y0, y1) if y0 <= y1 else (y1, y0)
                ext_rows.append((xa, ya, xb, yb, abs(w * h)))
                idx_rows.append(idx)
            if ext_rows:
                self._rect_extents[ax] = np.asarray(ext_rows, dtype=np.float64)
                self._rect_idx_arrays[ax] = np.asarray(idx_rows, dtype=np.int64)

    def render(self):
        sid = self.ids[self.idx]
        if self.visuals_dir is not None:
            # visuals mode: pre-rendered PNG, no XML overlay
            rgb = load_rgb(self.visuals_dir / f"{sid}_pred.png")
            objects = []
        else:
            rgb = load_rgb(RGB_DIR / f"{sid}.jpg")
            objects = parse_objects(BBOX_DIR / f"{sid}.xml")
        depth = load_rgb(DEPTH_DIR / f"{sid}.jpg")
        segmap = load_segmap(SEGMAP_DIR / f"{sid}_segmap.png")

        target_hw = rgb.shape[:2] if rgb is not None else (1, 1)
        depth = resize_to(depth, target_hw)
        segmap = resize_to(segmap, target_hw)

        # Reset per-image state.
        self.objects = objects
        self.selected = set()
        # Tear down everything (axes.clear wipes our cached artists too).
        for ax in self.axes:
            ax.clear()
            ax.set_xticks([]); ax.set_yticks([])
        self.rgb_artists.clear()
        self.depth_artists.clear()
        self.segmap_artists.clear()
        self.rgb_rect_to_idx.clear()
        self.any_rect_to_idx.clear()
        self._rect_extents.clear()
        self._rect_idx_arrays.clear()
        # ax.clear() removed the tooltip artist, drop the stale reference
        self._tooltip = None
        self._hovered_idx = None
        self._last_hover_t = 0.0

        if self.mode is None:
            rgb_title = f"RGB  {sid}.jpg"
        else:
            mode_label = (
                "Train visuals" if self.mode == "GECO2_train" else "Inference visuals"
            )
            rgb_title = f"{mode_label}  {sid}_pred.png"
        titles = [
            rgb_title,
            f"Depth  {sid}.jpg",
            f"Segmap  {sid}_segmap.png",
        ]
        imgs = [rgb, depth, segmap]
        cmaps = [
            None, None,
            "gray" if segmap is not None and segmap.ndim == 2 else None,
        ]

        for ax, img, title, cmap in zip(self.axes, imgs, titles, cmaps):
            if img is not None:
                ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=10)
            ax.set_aspect("equal", adjustable="box")

        h, w = target_hw
        self.axes[0].set_xlim(0, w)
        self.axes[0].set_ylim(h, 0)

        n_bboxes = sum(1 for o in objects if "bbox" in o)
        n_points = sum(1 for o in objects if "point" in o)
        self.fig.suptitle(
            f"[{self.idx + 1}/{len(self.ids)}]  id={sid}   "
            f"bboxes={n_bboxes}   points={n_points}     "
            f"(A/D = nav, Ctrl+scroll = zoom, right-drag = pan, "
            f"click bbox = mirror, hover = source info, Q/Esc = quit)",
            fontsize=11,
        )

        self._refresh_overlays()


def collect_ids():
    ids = sorted(p.stem for p in RGB_DIR.glob("*.jpg"))
    if not ids:
        raise SystemExit(f"No images found in {RGB_DIR}")
    return ids


def collect_visuals_ids(visuals_dir: Path):
    """Scan visuals_dir for *_pred.png and return the bare image ids."""
    if not visuals_dir.is_dir():
        raise SystemExit(f"Visuals dir not found: {visuals_dir}")
    ids = sorted(p.name[:-len("_pred.png")]
                 for p in visuals_dir.glob("*_pred.png"))
    if not ids:
        raise SystemExit(f"No *_pred.png files in {visuals_dir}")
    return ids


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="IOCfish5k bbox / prediction viewer.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--GECO2_train", action="store_true",
        help="View post-training visuals from <models>/GECO2_IOCfish_<tag>_<mode>_TRAIN_test_whole_visuals/.",
    )
    grp.add_argument(
        "--GECO2_inference", action="store_true",
        help="View inference visuals from <models>/GECO2_IOCfish_<tag>_<mode>_TEST_whole_visuals/.",
    )
    p.add_argument(
        "--tag", type=str, default=None,
        help="Experiment tag (one of: geco2, conv_depth_add, conv_hiera, depth_dim, sep_hiera). "
             "Required when --GECO2_train or --GECO2_inference is set.",
    )
    p.add_argument(
        "--mode", type=str, default="few", choices=("few", "zero"),
        help="Training regime: 'few' or 'zero'. Part of the visuals dir name.",
    )
    p.add_argument(
        "--model_name", type=str, default="GECO2_IOCfish",
        help="Base model name (default: GECO2_IOCfish).",
    )
    p.add_argument(
        "--models_dir", type=str, default=None,
        help=f"Override models dir (defaults to {MODELS_DIR}).",
    )
    return p


def main():
    args = _build_arg_parser().parse_args()

    if args.GECO2_train or args.GECO2_inference:
        if not args.tag:
            raise SystemExit(
                "--tag is required with --GECO2_train / --GECO2_inference")
        suffix = "_TRAIN_test_whole_visuals" if args.GECO2_train else "_TEST_whole_visuals"
        models_dir = Path(args.models_dir) if args.models_dir else MODELS_DIR
        visuals_dir = models_dir / f"{args.model_name}_{args.tag}_{args.mode}{suffix}"
        ids = collect_visuals_ids(visuals_dir)
        mode = "GECO2_train" if args.GECO2_train else "GECO2_inference"
        Viewer(ids, visuals_dir=visuals_dir, mode=mode)
    else:
        ids = collect_ids()
        Viewer(ids)
    plt.show()


if __name__ == "__main__":
    main()
