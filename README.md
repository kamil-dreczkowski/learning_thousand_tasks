# Learning a Thousand Tasks in a Day

This repository contains the implementation of all methods evaluated in the paper "Learning a Thousand Tasks in a Day". We provide model architectures, training scripts, and deployment examples.

Paper published on **Science Robotics**: https://www.science.org/doi/10.1126/scirobotics.adv7594
<br>Website: https://www.robot-learning.uk/learning-1000-tasks

## Overview

This codebase implements five methods for learning manipulation tasks from limited per-task demonstrations:

### Decomposition-based Methods
- **MT3**: Retrieval-based alignment + retrieval-based interaction (no training required)
- **Ret-BC**: Retrieval-based alignment + behavioral cloning interaction
- **BC-Ret**: Behavioral cloning alignment + retrieval-based interaction
- **BC-BC**: Behavioral cloning alignment + behavioral cloning interaction

### Monolithic Method
- **MT-ACT+**: End-to-end multi-task transformer (our adaptation of MT-ACT - see "RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking" for details)

---

## Quick Start

**Tested on**: Ubuntu 22.04 LTS (Jammy) with Linux kernel 5.15.0-43-generic

### Prerequisites

1. **Docker** - [Install Docker for Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
2. **NVIDIA Container Toolkit** (required for GPU support) - [Installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/KamilDre/learning_thousand_tasks.git
cd learning_thousand_tasks

# 2. Download and extract demonstration data
# Download demonstrations.zip from Google Drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZjGuX73LEgmMhVvHuQYwAqkJNFYC-Mwb' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZjGuX73LEgmMhVvHuQYwAqkJNFYC-Mwb" -O demonstrations.zip && rm -rf /tmp/cookies.txt
unzip demonstrations.zip -d assets/
rm demonstrations.zip

# Download inference_example.zip from Google Drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TznEjtIi1o-3HR3dOeqbfLnYQgOMY9B_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TznEjtIi1o-3HR3dOeqbfLnYQgOMY9B_" -O inference_example.zip && rm -rf /tmp/cookies.txt
unzip inference_example.zip -d assets/
rm inference_example.zip

# 3. Clone XMem for demonstration preprocessing (required for BC training)
git clone https://github.com/hkchengrex/XMem.git
cd XMem
git checkout v1.0  # Use stable v1.0 release
mkdir -p saves
wget https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth -O saves/XMem.pth
cd ..

# 4. Build the Docker image
make build
```

**Note**: If the `wget` commands for Google Drive downloads fail, you can manually download the files:
- [Demonstrations](https://drive.google.com/file/d/1ZjGuX73LEgmMhVvHuQYwAqkJNFYC-Mwb/view?usp=sharing) - Extract to `assets/demonstrations/`
- [Inference Example](https://drive.google.com/file/d/1TznEjtIi1o-3HR3dOeqbfLnYQgOMY9B_/view?usp=sharing) - Extract to `assets/inference_example/`

---

## Makefile Commands

We provide a Makefile for convenience. All commands automatically handle Docker setup, X11 display forwarding for Open3D visualizations, and volume mounting.

### Available Commands

```bash
make help                       # Show all available commands
make build                      # Build Docker image
make deploy_mt3                 # Run MT3 inference example
make preprocess_demos           # Preprocess demonstrations for BC training
make create_alignment_dataset   # Create dataset for BC alignment training
make create_interaction_dataset # Create dataset for BC interaction training
make create_mtact_dataset       # Create dataset for MT-ACT+ training
make train_bc_alignment         # Train BC alignment policy
make train_bc_interaction       # Train BC interaction policy
make train_mtact                # Train MT-ACT+ policy
make debug                      # Start interactive shell in container
make stop                       # Stop running container
```
---

## Example Data

After completing the setup, you will have the following data to run MT3 immediately and showcase how to train the BC policies:

**Demonstrations** (`assets/demonstrations/` - downloaded separately):
- 2× bottle grasping
- 1× shoe grasping
- Each includes RGB-D images, segmentation, bottleneck pose, and end-effector twists

**Test Data** (`assets/inference_example/` - downloaded separately):
- Test RGB-D images: `head_camera_ws_rgb.png`, `head_camera_ws_depth_to_rgb.png`
- Pre-computed segmentation: `head_camera_ws_segmap.npy`
- Camera intrinsics: `head_camera_rgb_intrinsic_matrix.npy`

**Pre-trained Models** (`assets/` - included in repository):
- `geometry_encoder.ckpt` - Pre-trained PointNet++ for geometry encoding
- `pose_estimator.ckpt` - Pre-trained 4-DOF pose regressor

**Camera Extrinsics** (`assets/` - included in repository):
- `T_WC_head.npy` - Head camera extrinsics (world-to-camera transform) from our experiments

---

## MT3 Example (No Training Required)

MT3 does not require any task specific training. The supplied deployment script demonstrates the complete pipeline: retrieval, pose estimation, alignment, and interaction.

### Run the Example

```bash
make deploy_mt3
```

### What Happens

The script runs through 7 steps:
1. **Load test image** - Loads RGB-D image and segmentation from `assets/inference_example/`
2. **Initialize live scene state** - Creates point cloud from segmented region
3. **Retrieve demonstration** - Finds most similar demo from `assets/demonstrations/`
4. **Load retrieved demonstration** - Loads demo RGB-D and segmentation
5. **Estimate relative pose** - Uses PointNet++ to compute transformation
6. **Transform bottleneck pose** - Computes target pose for robot alignment
7. **Load end-effector twists** - Loads demonstrated velocities for interaction

### What You'll See

**4 Open3D visualization windows** (close each to continue):
- Live scene point cloud
- Live scene vs retrieved demo comparison
- Registration result showing aligned point clouds after PointNet++
- Registration result showing aligned point clouds after refinement

**Saved visualizations** in `assets/example_visualisations/`:
- `test_scene_visualization.png` - RGB, depth, and segmentation of test scene
- `retrieval_visualization.png` - Live scene vs retrieved demo (2×2 grid)

**Console output** showing:
- Retrieved demonstration name
- Estimated 4×4 transformation matrix
- Target bottleneck pose (T_WE) for alignment phase
- End-effector twist dimensions (N×7) for interaction phase

---

## Demonstration Preprocessing

### Overview

The provided demonstrations include workspace segmentation (first frame) which is sufficient for MT3 deployment. However, training BC policies requires **per-timestep segmentation masks** for the entire trajectory. This preprocessing step generates these masks using XMem.

### Segmentation Method

In our experiments, we used **LangSAM** (Language Segment Anything Model) to obtain the initial workspace segmentation from language prompts (e.g., "grey shoe on table"). The preprocessing script assumes you have already segmented the first frame and saved it as `head_camera_ws_segmap.npy`. The script then:

1. **Loads the pre-computed workspace segmentation** (`head_camera_ws_segmap.npy`)
2. **Tracks object** through all timesteps using XMem
3. **Saves outputs**:
   - `head_camera_masks.npy` - Per-timestep masks (for BC training)
   - `demo_video.mp4` - RGB trajectory visualization
   - `demo_segmented_video.mp4` - Segmentation overlay visualization

### Running Preprocessing

1. **Ensure demonstrations have pre-computed workspace segmentation** (`head_camera_ws_segmap.npy`) for each task directory in `assets/demonstrations/`.

2. **Configure task names** in `thousand_tasks/demo_preprocessing/preprocess_demos.py` by updating the `TASK_NAMES` list with your demonstration directory names.

3. **Run preprocessing**:

```bash
make preprocess_demos
```

### Output Files

For each demonstration task, the script generates:

```
task_name_0000/
├── head_camera_masks.npy            # Per-timestep masks (T, H, W) for BC training
├── demo_video.mp4                   # RGB trajectory video
└── demo_segmented_video.mp4         # RGB video with segmentation overlay
```

**Note**: If `SAVE_PNG_IMAGES = True`, an additional `segmented_images/` directory will be created with per-frame PNG visualizations.

---

## Training Policies

This section describes how to train the three policy types required for BC-based methods:

### BC Alignment Policy (used by BC-Ret and BC-BC)

The BC alignment policy learns to predict trajectories to reach alignment poses. This policy is used to align the robot to the target pose before executing the interaction phase.

**1. Create processed dataset:**
```bash
make create_alignment_dataset
```
This will process demonstrations from `assets/demonstrations/` and create a preprocessed dataset in `assets/demonstrations/bn_reaching_processed/`.

**2. Train the policy:**
```bash
make train_bc_alignment
```
This will train the BC alignment policy and save checkpoints to `assets/checkpoints/bc_alignment/`. The best model (lowest combined position + rotation error) is saved as `best.pt`, and the final model is saved as `final.pt`.

**Configuration:** The default configuration generates 10 synthetic trajectories per demonstration (see `num_traj_to_bn_train` in `thousand_tasks/training/act_bn_reaching/config.py`). For training on your own demonstrations, you should likely change this to 1000 to generate sufficient training data.

---

### BC Interaction Policy (used by Ret-BC and BC-BC)

The BC interaction policy learns to predict trajectories (end-effector poses) from the current state during manipulation. This policy replays the demonstrated interaction after alignment.

**1. Create processed dataset:**
```bash
make create_interaction_dataset
```
This will process demonstrations from `assets/demonstrations/` and create a preprocessed dataset in `assets/demonstrations/interaction_processed/`.

**2. Train the policy:**
```bash
make train_bc_interaction
```
This will train the BC interaction policy and save checkpoints to `assets/checkpoints/bc_interaction/`. The best model is saved as `best.pt`, and the final model is saved as `final.pt`.

**Configuration:** The default configuration generates 10 interaction sub-trajectories per demonstration (see `num_inter_traj` in `thousand_tasks/training/act_interaction/config.py`). For training on your own demonstrations, you should likely change this to 1000 to generate sufficient training data.

---

### MT-ACT+ End-to-End Policy (baseline)

MT-ACT+ is an end-to-end multi-task transformer that directly predicts action sequences from observations without explicit decomposition into alignment and interaction phases.

**1. Create processed dataset:**
```bash
make create_mtact_dataset
```
This will process demonstrations from `assets/demonstrations/` and create a preprocessed dataset in `assets/demonstrations/processed/`.

**2. Train the policy:**
```bash
make train_mtact
```
This will train the MT-ACT+ policy and save checkpoints to `assets/checkpoints/mtact_plus/`. The best model is saved as `best.pt`, and the final model is saved as `final.pt`.

**Configuration:** The default configuration generates 10 full trajectories per demonstration (see `num_inter_traj` in `thousand_tasks/training/act_end_to_end/config.py`). For training on your own demonstrations, you should likely change this to 200 to generate sufficient training data.

---

## Deploying Trained Policies

After training BC policies, you can deploy the various method combinations. Each deployment script demonstrates the complete pipeline with visualization.

### Ret-BC (Retrieval-based Alignment + BC Interaction)

This method uses retrieval-based alignment (like MT3) but replaces retrieval-based interaction with a learned BC policy.

**Prerequisites:** Trained BC interaction policy at `assets/checkpoints/bc_interaction/best.pt`

```bash
make deploy_ret_bc
```

**Pipeline Steps:**
1. Load test RGB-D image and segmentation
2. Retrieve similar demonstration via hierarchical retrieval
3. Estimate relative pose with PointNet++ and refine with ICP
4. Apply 4DOF inductive bias and transform bottleneck pose to live scene
5. Run BC interaction policy to predict waypoint trajectory
6. Visualize predicted trajectory with Open3D

**What You'll See:**
- 4 Open3D windows showing point clouds and registrations (same as MT3)
- Final trajectory visualization with waypoints
- Console output with instructions for robot deployment
- Saved visualizations in `assets/example_visualisations/`

### BC-Ret (BC Alignment + Retrieval-based Interaction)

This method uses a learned BC policy for alignment and retrieval-based interaction (like MT3).

**Prerequisites:** Trained BC alignment policy at `assets/checkpoints/bc_alignment/best.pt`

```bash
make deploy_bc_ret
```

**Pipeline Steps:**
1. Load test RGB-D image and segmentation
2. Retrieve similar demonstration via hierarchical retrieval
3. Run BC alignment policy to predict trajectory to bottleneck pose
4. Visualize predicted alignment trajectory
5. [SIMULATED] Track waypoints until bottleneck pose is reached
6. Replay demonstrated end-effector velocities (like MT3)

**What You'll See:**
- 2 Open3D windows showing point clouds and retrieval comparison
- Alignment trajectory visualization with waypoints leading to target bottleneck
- Console output with instructions for robot deployment
- Saved visualizations in `assets/example_visualisations/`

### BC-BC (BC Alignment + BC Interaction)

This method uses learned BC policies for both alignment and interaction phases, with no retrieval required.

**Prerequisites:**
- Trained BC alignment policy at `assets/checkpoints/bc_alignment/best.pt`
- Trained BC interaction policy at `assets/checkpoints/bc_interaction/best.pt`

```bash
make deploy_bc_bc
```

**Pipeline Steps:**
1. Load test RGB-D image and segmentation
2. Run BC alignment policy to predict trajectory to bottleneck pose
3. Track waypoints until alignment policy signals termination (terminate_prob > 0.95)
4. Run BC interaction policy to predict manipulation trajectory
5. Track waypoints with gripper control until interaction policy signals termination (terminate_prob > 0.95)
6. Visualize both alignment and interaction trajectories

**What You'll See:**
- 3 Open3D windows: point cloud, alignment trajectory (green), interaction trajectory (red)
- Console output with instructions for both phases
- Saved visualizations in `assets/example_visualisations/`

**Key Difference from Other Methods:**
- No retrieval required - policies generalize directly to novel objects
- Both phases use learned policies with closed-loop re-inference
- Alignment continues until bottleneck reached (terminate > 0.95)
- Interaction continues until task complete (terminate > 0.85)

### MT-ACT+ (End-to-End Multi-Task Transformer)

This is an end-to-end baseline that directly predicts manipulation trajectories without explicit decomposition.

**Prerequisites:** Trained MT-ACT+ policy at `assets/checkpoints/mtact_plus/best.pt`

```bash
make deploy_mtact
```

**Pipeline Steps:**
1. Load test RGB-D image and segmentation
2. Run MT-ACT+ policy to predict end-to-end trajectory
3. Track waypoints with gripper control
4. Re-run inference in closed-loop until termination (terminate_prob > 0.95)
5. Visualize predicted trajectory

**What You'll See:**
- 2 Open3D windows: point cloud and predicted trajectory (blue)
- Console output with instructions for closed-loop deployment
- Saved visualizations in `assets/example_visualisations/`

**Key Difference from Decomposition Methods:**
- No explicit alignment + interaction phases
- Single policy handles complete task from start to finish
- Directly predicts waypoints without retrieving demonstrations
- Trained on full demonstration trajectories (not decomposed)

---

## Demonstration Data Format

Each demonstration is a directory containing RGB-D observations, segmentation masks, robot poses, and interaction trajectories. Example from `pick_up_grey_shoe`:

### File Structure with Specifications

| File | Shape | Type | Description |
|------|-------|------|-------------|
| **Required for MT3** | | | |
| `head_camera_ws_rgb.png` | (720, 1280, 3) | uint8 | Workspace RGB image (gripper out of view) |
| `head_camera_ws_depth_to_rgb.png` | (720, 1280) | uint16 | Aligned depth in millimeters |
| `head_camera_ws_segmap.npy` | (720, 1280) | bool | Binary mask (True=object, False=background) |
| `head_camera_rgb_intrinsic_matrix.npy` | (3, 3) | float64 | Camera intrinsics `[[fx,0,cx],[0,fy,cy],[0,0,1]]` |
| `bottleneck_pose.npy` | (4, 4) | float64 | Target end-effector pose (SE(3) matrix) |
| `demo_eef_twists.npy` | (T, 7) | float64 | Velocity commands `[vx,vy,vz,wx,wy,wz,gripper]` |
| `task_name.txt` | - | text | Task description (e.g., "pick_up_grey_shoe") |
| **Additional for BC Training** | | | |
| `head_camera_rgb.npy` | (T, 720, 1280, 3) | uint8 | Full RGB trajectory |
| `head_camera_depth_to_rgb.npy` | (T, 720, 1280) | uint16 | Full depth trajectory in millimeters |
| `head_camera_masks.npy` | (T, 720, 1280) | bool | Per-timestep masks (generated by preprocessing) |
| **Optional** | | | |
| `geometry_encoding.npy` | (512,) | float32 | Pre-computed PointNet++ embedding |
| `workspace_img_T_WE.npy` | (4, 4) | float64 | Robot pose at workspace image capture |

Where T = number of timesteps (e.g., 179 frames at 30Hz = ~6 seconds)

### Additional Notes

* Geometry Embeddings: The geometry encoder can be used to generate them. Retrieval should also regenerate them if they are missing.

* Workspace Images ("ws" prefix): These are frames captured with the **gripper moved out of the camera's field of view** to avoid occluding the object.

* End-Effector Twists: Velocities in the end-effector frame, not world frame. Format per row: `[vx, vy, vz, wx, wy, wz, gripper_next]` where velocities are in m/s and rad/s, gripper is 0=close, 1=open.

* Bottleneck Pose: This is the alignment pose. It is just a pose suitable for the upcoming manipulation. Typically, the closer to the target object, the better.

* Segmentation: The workspace segmentation (`head_camera_ws_segmap.npy`) must be computed before using the demonstrations. We used LangSAM with language prompts in our experiments, but any segmentation method (SAM, manual annotation, etc.) can be used. For BC training, run `make preprocess_demos` to generate per-timestep masks via XMem tracking.
