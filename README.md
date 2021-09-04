# blend
Blend two rasters together so they transition smoothly into each other.

# Example

Using inverse distance weighting interpolation, a buffer region around the inner nest is smoothed to blend smoothly into the outer grid.

```python
import numpy as np

from blend import Grid

grid_coarse = Grid((0, 1, 0, 1), 1 / 100, values=np.ones((100, 100)), extrapolate=True)
grid_fine = Grid(
    (0.40, 0.70, 0.40, 0.70), 0.30 / 10, values=np.ones((10, 10)) + 1, extrapolate=False
)
grid_coarse_2 = grid_fine.blend_into(grid_coarse, blend_width=3, p=1, nnear=100)

grid_coarse_2.plot(vmin=0, vmax=2.0)
```

using some NetCDF DEMs 
```
 from blend import DEM
 
 dn1 = "/Users/keithroberts/codes/oceanmesh/datasets/PostSandyNCEI.nc"
 
 dn2 = "/Users/keithroberts/codes/OceanMesh2D/datasets/SRTM15+.nc"
 
 bbox1 = (-74.24, -73.75, 40.5, 41)
 bbox2 = (-76.01, -70.0, 38, 42)
 
 dem1 = DEM(dn1, bbox1)
 
 dem2 = DEM(dn2, bbox2)
 
 dem3 = dem1.blend_into(dem2, blend_width=500, p=2, nnear=28)
 
 dem2.plot(vmin=-100, vmax=500, show=False, filename="unmerged.png", dpi=1000)
 dem3.plot(vmin=-100, vmax=500, show=False, filename="merged.png", dpi=1000)
 ```
