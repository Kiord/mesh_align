# mesh_align

A tool for rigid mesh alignment written in Python.

## Usage

`python mesh_align source.obj target.obj -tp source2target.npy -tmp source2target.obj`

## Notes


This tool is based on the Iterative Closest Point (ICP) algorithm. At each iteration, we look for the best rigid transform that aligns the source mesh on the target mesh. As a greedy algorithm, ICP is very sensitive to the data, the initial guess and the management of the outliers. Depending on your scenario, you may want to tweak the parameters to increase robustness or speed.

## Coarse-to-fine
This tool does a coarse-to-fine algorithm:

- Initial guess
  - The source mesh is scaled and centered like the target mesh
- Coarse ICP phase
  - Based on the transform from the initial guess
  - Many ICPs are performed with few sampled points and few iterations.
  - Reflections and rotations can optionally be tested during this phase.
- Fine ICP phase
  - Based on the transform from the coarse ICP phase
  - A single ICP is performed with more sampled points and more iterations.
  - Outputs the resulting transform

## Parameters

Type `python mesh_align --help` for a description of each input/parameter.

### Outlier ratio

To account for outliers, we use an outlier ratio method with the option `-o`, which is the **most important parameter**.  This is usually easier to manage than a distance threshold, but it depends on the data. `-o 0.5` will count the 50% worst point matches as outliers. 

### Fixed scale

The two 3D objects to align might be known to be already at the same scale. Fix the scale using the flag `-fs`.

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

A "good" solution according to the ICP is to shrink the source mesh to a point such that the distance between the matched points is very small. To avoid this issue, a minimum and maximum scaling factor are provided with `min_scale` and `max_scale`, respectively. Default values are 0.7 and 1.3.

## Requirement
Strong dependence:
  - python 3.7+
  - numpy
  - trimesh
  - scipy

Can removed easily:
  - click
  - pyvista
  - tqdm
  - matplotlib