import json
import numpy as np
import open3d as o3d
import pycolmap
from pathlib import Path

# ─────────────────────────────────────────────
# INPUTS — adapt these
# ─────────────────────────────────────────────

ply_path          = Path("dense_reconstruction/fuse.ply")
input_sparse_path = Path("input_poses")

# Map class_id → label name (match your YOLO classes.txt order)
class_names = {
    0: "vga",
    1: "lan",
    2: "usb",
    3: "hdmi",
    4: "dp",
}

label_colors = {
    "vga":      [1.0, 0.0, 0.0],   # red
    "lan": [0.0, 1.0, 0.0],   # green
    "usb":    [0.0, 0.0, 1.0],   # blue
    "hdmi":      [1.0, 1.0, 0.0],   # yellow
    "dp":     [1.0, 0.0, 1.0],   # magenta
}

# YOLO raw output: { "image_name": "raw yolo string" }
# Paste your YOLO output as a multi-line string per image
yolo_raw = {
    "img07.png": """
0 0.502317 0.498234 0.028469 0.117702
1 0.542042 0.816620 0.028469 0.062382
2 0.502861 0.797706 0.015130 0.052083
2 0.519848 0.804431 0.013342 0.051839
2 0.535275 0.378998 0.017992 0.062710
2 0.555871 0.380892 0.017519 0.064815
3 0.502131 0.626894 0.018466 0.067761
4 0.527817 0.634891 0.018703 0.071128
""",
}

output_json    = Path("obb_results.json")
output_ply_dir = Path("obb_segments")
output_ply_dir.mkdir(exist_ok=True)

OUTLIER_NB  = 20
OUTLIER_STD = 2.0
MIN_POINTS  = 10


# ─────────────────────────────────────────────
# 1. Parse YOLO output
# ─────────────────────────────────────────────

def parse_yolo_raw(raw_str, img_w, img_h, class_names):
    """
    Parse raw YOLO HBB output string into list of
    {"label": str, "bbox": [x1,y1,x2,y2]} in pixel coords.
    Multiple detections of the same class are suffixed _0, _1 ...
    """
    entries      = []
    label_counts = {}

    for line in raw_str.strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id        = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])

        # Denormalize
        xc *= img_w;  bw *= img_w
        yc *= img_h;  bh *= img_h
        bbox = [xc - bw/2, yc - bh/2, xc + bw/2, yc + bh/2]

        base_label = class_names.get(class_id, f"class_{class_id}")
        count      = label_counts.get(base_label, 0)

        # Unique label per instance (vga_socket, power_socket_0, power_socket_1 ...)
        label = base_label if count == 0 else f"{base_label}_{count}"
        label_counts[base_label] = count + 1

        entries.append({"label": label, "base_label": base_label, "bbox": bbox})

    return entries


# ─────────────────────────────────────────────
# 2. Load assets
# ─────────────────────────────────────────────

print("Loading dense point cloud...")
pcd       = o3d.io.read_point_cloud(str(ply_path))
points_3d = np.asarray(pcd.points)
colors_3d = np.asarray(pcd.colors) if pcd.has_colors() else None
print(f"  {len(points_3d):,} points loaded")

print("Loading reconstruction...")
reconstruction = pycolmap.Reconstruction(str(input_sparse_path))
name_to_image  = {img.name: img for img in reconstruction.images.values()}
print(f"  {len(name_to_image)} images: {list(name_to_image.keys())}")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def project_points(points_3d, image, camera):
    try:
        cwf = image.cam_from_world
        if callable(cwf):
            cwf = cwf()
        R = cwf.rotation.matrix()
        t = np.array(cwf.translation)
    except AttributeError:
        R = np.array(image.rotation_matrix())
        t = np.array(image.tvec)

    pts_cam = (R @ points_3d.T).T + t
    depths  = pts_cam[:, 2]

    fx, fy, cx, cy = camera.params
    denom = np.where(depths != 0, depths, 1e-10)
    u = (pts_cam[:, 0] / denom) * fx + cx
    v = (pts_cam[:, 1] / denom) * fy + cy
    uvs = np.stack([u, v], axis=1)

    return uvs, depths


def mask_inside_bbox(uvs, depths, bbox, w, h):
    x1, y1, x2, y2 = bbox
    u, v = uvs[:, 0], uvs[:, 1]
    return (
        (depths > 0) &
        (u >= 0) & (u < w) &
        (v >= 0) & (v < h) &
        (u >= x1) & (u <= x2) &
        (v >= y1) & (v <= y2)
    )

def mask_inside_bbox_nearest(uvs, depths, bbox, w, h, abs_tolerance=0.02):
    """
    Keep only points at the NEAREST depth inside the bbox.
    abs_tolerance: meters from the closest point to keep.
                   e.g. 0.02 = keep points within 2cm of the closest point.
    """
    x1, y1, x2, y2 = bbox
    u, v = uvs[:, 0], uvs[:, 1]

    in_bbox = (
        (depths > 0) &
        (u >= 0) & (u < w) &
        (v >= 0) & (v < h) &
        (u >= x1) & (u <= x2) &
        (v >= y1) & (v <= y2)
    )

    if in_bbox.sum() < 5:
        return in_bbox

    # Absolute nearest depth — no percentile, no fraction
    min_depth = depths[in_bbox].min()

    # Keep only points within abs_tolerance meters of that nearest surface
    near_mask = in_bbox & (depths <= min_depth + abs_tolerance)

    return near_mask

def fit_obb(pts):
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov      = np.cov(centered.T)
    _, eigenvectors = np.linalg.eigh(cov)
    axes     = eigenvectors[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    projected    = centered @ axes
    min_proj     = projected.min(axis=0)
    max_proj     = projected.max(axis=0)
    extent       = max_proj - min_proj
    local_center = (min_proj + max_proj) / 2.0
    center       = centroid + axes @ local_center
    return {
        "center":   center.tolist(),
        "extent":   extent.tolist(),
        "rotation": axes.T.tolist(),
    }


def save_segmented_ply(all_pts, all_colors, seg_masks, labels,
                       base_labels, obbs, out_dir):
    """
    Save one full scene PLY with each object highlighted in its label color,
    plus one OBB wireframe PLY per object.
    """
    n = len(all_pts)

    if all_colors is not None and len(all_colors) == n:
        final_colors = all_colors.copy()
    else:
        final_colors = np.full((n, 3), 0.6)

    for mask, base_label in zip(seg_masks, base_labels):
        color = np.array(label_colors.get(base_label, [1.0, 0.5, 0.0]))
        final_colors[mask] = color

    full_pcd        = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(all_pts)
    full_pcd.colors = o3d.utility.Vector3dVector(final_colors)
    full_path       = out_dir / "full_scene_segmented.ply"
    o3d.io.write_point_cloud(str(full_path), full_pcd)
    print(f"\n  Full segmented scene → {full_path}")

    # OBB wireframe per object
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ]
    signs = np.array([
        [-1,-1,-1],[+1,-1,-1],[+1,+1,-1],[-1,+1,-1],
        [-1,-1,+1],[+1,-1,+1],[+1,+1,+1],[-1,+1,+1],
    ])
    for label, base_label, obb in zip(labels, base_labels, obbs):
        center  = np.array(obb["center"])
        extent  = np.array(obb["extent"])
        axes    = np.array(obb["rotation"]).T
        corners = center + (signs * extent / 2.0) @ axes.T

        ls        = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners)
        ls.lines  = o3d.utility.Vector2iVector(edges)
        color     = label_colors.get(base_label, [1.0, 0.5, 0.0])
        ls.colors = o3d.utility.Vector3dVector([color] * len(edges))

        obb_path  = out_dir / f"{label}_obb.ply"
        o3d.io.write_line_set(str(obb_path), ls)
        print(f"  [{label}] OBB wireframe → {obb_path}")


# ─────────────────────────────────────────────
# 3. Main loop
# ─────────────────────────────────────────────

output = []

for img_name, raw_str in yolo_raw.items():

    if img_name not in name_to_image:
        print(f"\n[ERROR] '{img_name}' not in reconstruction.")
        print(f"  Available: {list(name_to_image.keys())}")
        continue

    image  = name_to_image[img_name]
    camera = reconstruction.cameras[image.camera_id]
    w, h   = camera.width, camera.height

    detections = parse_yolo_raw(raw_str, w, h, class_names)
    print(f"\nImage '{img_name}' | {w}x{h} | {len(detections)} detections")

    uvs, depths = project_points(points_3d, image, camera)

    in_front = depths > 0
    in_frame = in_front & (uvs[:,0] >= 0) & (uvs[:,0] < w) & \
                          (uvs[:,1] >= 0) & (uvs[:,1] < h)
    print(f"  Points in front : {in_front.sum():,}")
    print(f"  Points in frame : {in_frame.sum():,}")

    all_masks      = []
    all_labels     = []
    all_base_labels= []
    all_obbs       = []

    for det in detections:
        label      = det["label"]
        base_label = det["base_label"]
        bbox       = det["bbox"]
        x1,y1,x2,y2 = bbox

        print(f"\n  [{label}] bbox px → "
              f"x1={x1:.1f} y1={y1:.1f} x2={x2:.1f} y2={y2:.1f} "
              f"({x2-x1:.1f}×{y2-y1:.1f} px)")

        # mask          = mask_inside_bbox(uvs, depths, bbox, w, h)
        mask = mask_inside_bbox_nearest(
            uvs, depths, bbox, w, h,
            abs_tolerance=0.02   # 2cm band — tune this to your scene scale
        )
        candidate_pts = points_3d[mask]
        print(f"  [{label}] {mask.sum():,} points inside bbox")

        if len(candidate_pts) < MIN_POINTS:
            print(f"  [{label}] ✗ Too few points ({len(candidate_pts)}), skipping.")
            print(f"    → bbox may be too small, or area not covered by dense recon.")
            continue

        sub_pcd = o3d.geometry.PointCloud()
        sub_pcd.points = o3d.utility.Vector3dVector(candidate_pts)
        filtered, _ = sub_pcd.remove_statistical_outlier(
            nb_neighbors=OUTLIER_NB, std_ratio=OUTLIER_STD
        )
        clean_pts = np.asarray(filtered.points)
        print(f"  [{label}] {len(clean_pts):,} points after outlier removal")

        if len(clean_pts) < MIN_POINTS:
            print(f"  [{label}] Using raw points (outlier removal too aggressive).")
            clean_pts = candidate_pts

        obb = fit_obb(clean_pts)

        all_masks.append(mask)
        all_labels.append(label)
        all_base_labels.append(base_label)
        all_obbs.append(obb)

        output.append({"entity": label, "obb": obb})
        print(f"  [{label}] ✓ center={[f'{v:.4f}' for v in obb['center']]}  "
              f"extent={[f'{v:.4f}' for v in obb['extent']]}")

    if all_masks:
        save_segmented_ply(
            points_3d, colors_3d,
            all_masks, all_labels, all_base_labels, all_obbs,
            output_ply_dir
        )


# ─────────────────────────────────────────────
# 4. Save JSON
# ─────────────────────────────────────────────

with open(output_json, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nDone! {len(output)} OBBs → {output_json}")
print(f"PLYs → {output_ply_dir}/")