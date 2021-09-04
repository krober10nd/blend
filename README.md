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
