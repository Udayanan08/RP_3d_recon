# RP_3D_recon
## DEPENDENCIES
  - pycolmap ( For CUDA support install colmap from source)
  - ultralytics
  - open3d
  - numpy
  
## To run 
1. ```python3 run_reconstruction.py```
2. ```python3 run_yolo.py```
3. ```python3 run_bbox_3d.py```

![Dense Model](assets/model.png)
![bbox](assets/bbox_detected.png)

## To run the application after buiding the model
1. ```python3 app.py```
![app](assets/app_img.png)

## OUTPUT
1. Sparse output
2. Dense output and PLY model
3. Segmented 3D model for individual bounding boxes
