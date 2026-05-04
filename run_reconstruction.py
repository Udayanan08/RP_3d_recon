import pycolmap
from pathlib import Path

# 1. Setup Paths
project_path = Path.cwd()
image_path = project_path / "valid_dataset"
database_path = project_path / "database.db"
# This folder must contain your known cameras.txt and images.txt
input_sparse_path = project_path / "input_poses"
output_sparse_path = project_path / "triangulated_model"
dense_path = project_path / "dense_reconstruction"

# Create directories
output_sparse_path.mkdir(exist_ok=True)
dense_path.mkdir(exist_ok=True)


reconstruction = pycolmap.Reconstruction(input_sparse_path)

# 2. Extract and Match Features
# This finds keypoints in your images to link them together
options = pycolmap.ImageReaderOptions()
options.camera_model = "PINHOLE"
options.camera_params = "1477.00974684544, 1480.4424455584467, 1298.2501500778505, 686.8201623541711"
# SINGLE_CAMERA mode ensures all images share the same intrinsic ID
# options.camera_mode = pycolmap.CameraMode.SINGLE_CAMERA

image_list = [f"img{i:02d}.png" for i in range(1, 17)] 

pycolmap.extract_features(
    database_path=database_path, 
    image_path=image_path, 
    image_names=image_list,
    reader_options=options,
    camera_mode = pycolmap.CameraMode.SINGLE
)
pycolmap.match_exhaustive(database_path)

# 3. Triangulate Points (Metric Reconstruction)
# This uses your known poses to solve for 3D points
all_image_ids = reconstruction.images.keys()  
print("All image IDs:", list(all_image_ids)) 

pycolmap.triangulate_points(
    reconstruction=reconstruction,
    database_path=database_path,
    image_path=image_path,
    output_path=str(output_sparse_path)
)
reconstruction.write(output_sparse_path)


# maps = pycolmap.incremental_mapping(database_path, image_path, output_sparse_path)
# maps[0].write(output_sparse_path)

# 4. Dense Reconstruction (Optional but recommended for a full model)
# This generates a dense point cloud/mesh
pycolmap.undistort_images(dense_path, output_sparse_path, image_path)
pycolmap.patch_match_stereo(dense_path)

dense_path_save = dense_path / "dense"
dense_path_save.mkdir(exist_ok=True)
# 1. Initialize the options object
options2 = pycolmap.StereoFusionOptions()

# 2. Set the number of threads (e.g., 8 threads)
# Set to -1 to use all available logical processors
options2.num_threads = 4
# pycolmap.stereo_fusion(dense_path_save / "fuse.ply", dense_path, options=options2)
ply_output_path = dense_path / "fused.ply"
# reconstruction.export_PLY(str(ply_output_path))

import subprocess

subprocess.run([
    "colmap", "stereo_fusion",
    "--workspace_path",   str(dense_path),
    "--workspace_format", "COLMAP",
    "--input_type",       "geometric",
    "--output_path",      str(dense_path / "fuse.ply"),
    "--StereoFusion.num_threads", "4",
    "--StereoFusion.min_num_pixels",     "3",   # was 5
    "--StereoFusion.max_reproj_error",   "4",   # was 2
    "--StereoFusion.max_depth_error",    "0.1", # was 0.01
    "--StereoFusion.max_normal_error",   "20",  # was 17
], check=True)

print(f"Reconstruction finished! Dense model saved at: {dense_path}/fused.ply")


