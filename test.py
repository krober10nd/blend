import numpy as np

from blend import Grid

grd_a = Grid((0, 1, 0, 1), 1 / 100, values=np.ones((100, 100)), fill=None)
grd_b = Grid(
    (0.60, 0.70, 0.60, 0.70), 0.11 / 20, values=np.ones((20, 20)) + 1, fill=999
)

grd_c2 = grd_b.blend_into(grd_a, pad=15)

grd_c2.plot()
