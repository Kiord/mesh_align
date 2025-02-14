# mesh_align

A tool for rigid mesh alignment written in Python.

https://github.com/user-attachments/assets/8f5eb941-9a9d-435a-9025-8ff0b2971220


## Usage

`python mesh_align.py <path to source mesh> <path to target mesh> -tp <path to write the 4x4 transform matrix> -tmp <path to write the transformed source mesh>`

example:

`python mesh_align.py meshes/source.obj meshes/target.obj -tp results/source2target.npy -tmp results/source2target.obj -tref`

## Notes


This tool is based on the Iterative Closest Point (ICP) algorithm. At each iteration, we look for the best rigid transform that aligns the source mesh on the target mesh. As a greedy algorithm, ICP is very sensitive to the data, the initial guess and the management of outliers. Depending on your scenario, you may want to tweak the parameters to increase robustness or speed. 
This implementation shares similarities with Trimesh's `registration.mesh_other` procedure, but has the following differences:
- Management of outliers
  - This algorithm uses the outlier ratio rule for more robustness with noisy/incomplete data
  - `trimesh.registration.mesh_other` does not manage outliers
- Scaling factor clamping
  - This algorithm clamps the scaling factor to avoid wrong convergence by excessive shrinking/enlarging
  - `trimesh.registration.mesh_other` does not constrain the scaling factor
- Rotations testing
  - This algorithms (can) test for 9 axis-aligned rotations in addition to the 7 reflections in the coarse phase
  - `trimesh.registration.mesh_other` does not test rotations in the coarse phase
- Modularity
  - This tool only aligns Trimesh objects
  - `trimesh.registration.mesh_other` adapts to more data structures

## Coarse-to-fine
This tool uses a coarse-to-fine algorithm:

- Initial guess
  - The source mesh is scaled and centered like the target mesh
- Coarse ICP phase
  - Based on the transform from the initial guess
  - ICP instances are performed with few sampled points and few iterations
  - Transformation combintations can be tested
    - Test reflections (7 combinations)
    - Test rotations (9 combinations)
- Fine ICP phase
  - Based on the transform from the coarse ICP phase
  - A single ICP instance is performed with more sampled points and more iterations.
  - Outputs the resulting transform

## Parameters

Type `python mesh_align --help` for a description of each input/parameter.

### Outlier ratio

To account for outliers, we use the outlier ratio method with the option `-o`, which is the **most important parameter**.  This is usually easier to manage than a distance threshold, but it depends on the data. `-o 0.5` will count the 50% worst point matches as outliers. 

### Reflections and rotations testing

In the coarse phase, the flags `-tref` and `-trot` enable the testing of multiple reflections and rotation, respectively. This helps ensure that we start with a decent solution for the fine phase.
In the embedded example, the source mesh has been mirrored, so adding -tref is mandatory.


### Fixed scale

The two 3D objects to be aligned might be known to already be at the same scale. Use the flag -fs to fix the scale.

### Points and iterations

The number of sampled points and iterations can be set using:
- `--iterations_raw 50`
- `--count_source_raw 1_000`
- `--count_target_raw 5_000`
- `--iterations_fine 500`
- `--count_source_fine 10_000`
- `--count_target_fine 20_000`

### Plot

Plotting the registration can be helpful to understand failing cases. Plot with the flag `-p`.

### Scaling factor bounds

A 'good' solution according to ICP is to shrink the source mesh to a point where the distance between the matched points is very small. To avoid this issue, a minimum and maximum scaling factor are provided with `min_scale` and `max_scale`, respectively. Default values are 0.7 and 1.3.

## Requirement
Strong dependence:
  - python 3.7+
  - numpy
  - trimesh
  - scipy

Required dependencies (can be removed easily by modifying the code):
  - click
  - pyvista
  - tqdm
