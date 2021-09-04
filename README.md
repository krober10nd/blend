# blend
Blend two rasters together so they transition smoothly into each other using inverse distance weighting interpolation (IDW).

# Example

A buffer region of `blend_width` grid cells around an inner nest is created. In this buffer region, IDW interpolation is used to smoothly blend from the nest to the outer grid. 

Key parameters involved in the blend are `nnear` the number of nearest neighbors to do the nearest neighbor query for IDW, the `blend_width` or number of grid cells to buffer the inner nest, and `p` which controls `1 / distance**p` in the IDW calculation.

```python
from blend import Grid

grid_coarse = Grid((0, 1, 0, 1), 1 / 100, values=1)
grid_fine = Grid((0.40, 0.70, 0.40, 0.70), 1/1000, values=5)
grid_coarse_2 = grid_fine.blend_into(grid_coarse, blend_width=50, p=1, nnear=28)
grid_coarse_2.plot(vmin=1, vmax=5.0)
```

using some NetCDF DEMs with real geophysical data. Note the `DEM` class which is a child class of `Grid`.

```python
 from blend import DEM
 
 dn1 = "PostSandyNCEI.nc"
 
 dn2 = "SRTM15+.nc"
 
 bbox1 = (-74.24, -73.75, 40.5, 41)
 bbox2 = (-76.01, -70.0, 38, 42)
 
 dem1 = DEM(dn1, bbox1)
 
 dem2 = DEM(dn2, bbox2)
 
 dem3 = dem1.blend_into(dem2, blend_width=500, p=2, nnear=28)
 
 dem2.plot(vmin=-100, vmax=500, show=False, filename="unmerged.png", dpi=1000)
 dem3.plot(vmin=-100, vmax=500, show=False, filename="merged.png", dpi=1000)
 ```
