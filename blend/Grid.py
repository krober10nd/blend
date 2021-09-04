import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

from .idw import Invdisttree


class Grid:
    """Abstracts a structured grid along with
    primitive operations (e.g., min, project, etc.) and
    stores data `values` defined at each grid point.
    Parameters
    ----------
    extent: tuple
        domain extents
    dx: float
        spacing between grid points
    values: scalar or array-like
        values at grid points
    Attributes
    ----------
        x0y0: tuple
            bottom left corner coordinate
        nx: int
            number of grid points in x-direction
        ny: int
            number of grid points in y-direction
    """

    def __init__(self, extent, dx, dy=None, values=None, extrapolate=False):
        if dy is None:
            dy = dx
        self.extent = extent
        self.x0y0 = (extent[0], extent[2])  # bottom left corner coordinates
        self.dx = dx
        self.dy = dy
        self.nx = int((self.extent[1] - self.extent[0]) // self.dx) + 1
        self.ny = int((self.extent[3] - self.extent[2]) // self.dy) + 1
        self.values = values
        self.eval = None
        self.extrapolate = extrapolate

    @property
    def dx(self):
        return self.__dx

    @dx.setter
    def dx(self, value):
        if value < 0:
            raise ValueError("Grid spacing (dx) must be > 0.0")
        self.__dx = value

    @property
    def dy(self):
        return self.__dy

    @dy.setter
    def dy(self, value):
        if value < 0:
            raise ValueError("Grid spacing (dy) must be > 0.0")
        self.__dy = value

    @property
    def extent(self):
        return self.__extent

    @extent.setter
    def extent(self, value):
        if value is None:
            self.__extent = value
        else:
            if len(value) < 4:
                raise ValueError("extent has wrong number of values.")
            if value[1] < value[0]:
                raise ValueError("extent has wrong values.")
            if value[3] < value[2]:
                raise ValueError("extent has wrong values.")
            self.__extent = value

    @property
    def values(self):
        return self.__values

    @values.setter
    def values(self, data):
        if np.isscalar(data):
            data = np.tile(data, (self.nx, self.ny))
        elif data is None:
            return
        assert data.shape == (self.nx, self.ny)
        self.__values = data  # [: self.nx+1, : self.ny]

    def create_vectors(self):
        """Build coordinate vectors
        Parameters
        ----------
            None
        Returns
        -------
        x: ndarray
            1D array contain data with `float` type of x-coordinates.
        y: ndarray
            1D array contain data with `float` type of y-coordinates.
        """
        x = self.x0y0[0] + np.arange(0, self.nx) * self.dx
        y = self.x0y0[1] + np.arange(0, self.ny) * self.dy
        return x, y

    def create_grid(self):
        """Build a structured grid
        Parameters
        ----------
            None
        Returns
        -------
        xg: ndarray
            2D array contain data with `float` type.
        yg: ndarray
            2D array contain data with `float` type.
        """
        x, y = self.create_vectors()
        return np.meshgrid(x, y, sparse=False, indexing="ij")

    def find_indices(self, points, lon, lat, tree=None):
        """Find linear indices `indices` into a 2D array such that they
        return the closest point in the structured grid defined by `x` and `y`
        to `points`.
        Parameters
        ----------
        points: ndarray
            Query points. 2D array with `float` type.
        lon: ndarray
            Grid points in x-dimension. 2D array with `float` type.
        lat: ndarray
            Grid points in y-dimension. 2D array with `float` type.
        tree: :obj:`scipy.spatial.ckdtree`, optional
            A KDtree with coordinates from :class:`Shoreline`
        Returns
        -------
        indices: ndarray
            Indicies into an array. 1D array with `int` type.
        """
        points = points[~np.isnan(points[:, 0]), :]
        if tree is None:
            lonlat = np.column_stack((lon.ravel(), lat.ravel()))
            tree = scipy.spatial.cKDTree(lonlat)
        dist, idx = tree.query(points, k=1)
        return np.unravel_index(idx, lon.shape)

    def interpolate_to(self, grid2, method="linear"):
        """Interpolates linearly self.values onto :class`Grid` grid2 forming a new
        :class:`Grid` object grid3.
        Note
        ----
        In other words, in areas of overlap, grid1 values
        take precedence elsewhere grid2 values are retained. Grid3 has
        dx and resolution of grid2.
        Parameters
        ----------
        grid2: :obj:`Grid`
            A :obj:`Grid` with `values`.
        method: str, optional
            Way to interpolate data between grids
        Returns
        -------
        grid3: :obj:`Grid`
            A new `obj`:`Grid` with projected `values`.
        """
        # is grid2 even a grid object?
        assert isinstance(grid2, Grid), "Object must be Grid."
        # check if they overlap
        x1min, x1max, y1min, y1max = self.extent
        x2min, x2max, y2min, y2max = self.extent
        overlap = x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max
        assert overlap, "Grid objects do not overlap."
        lon1, lat1 = self.create_vectors()
        lon2, lat2 = grid2.create_vectors()
        if self.extrapolate:
            _FILL = None
        else:
            _FILL = -999
        # take data from grid1 --> grid2
        fp = RegularGridInterpolator(
            (lon1, lat1),
            self.values,
            method=method,
            bounds_error=False,
            fill_value=_FILL,
        )
        xg, yg = np.meshgrid(lon2, lat2, indexing="ij", sparse=True)
        new_values = fp((xg, yg))
        # where fill replace with grid2 values
        new_values[new_values == _FILL] = grid2.values[new_values == _FILL]
        return Grid(
            extent=grid2.extent,
            dx=grid2.dx,
            dy=grid2.dy,
            values=new_values,
        )

    def blend_into(self, coarse, blend_width=10, p=1, nnear=6, eps=0.0):
        """Blend self.Grid into the coarse one so values transition smoothly"""
        _FILL = -99999  # uncommon value
        if not isinstance(coarse, Grid):
            raise ValueError("Object must be Grid.")
        # check if they overlap
        x1min, x1max, y1min, y1max = self.extent
        x2min, x2max, y2min, y2max = self.extent
        overlap = x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max
        assert overlap, "Grid objects do not overlap."
        _fine = self.values
        # 1. Pad the finer grid's values
        _fine_w_pad_values = np.pad(
            _fine, pad_width=blend_width, mode="constant", constant_values=_FILL
        )
        # 2. Create a new Grid fine_w_pad
        _add_length = self.dx * blend_width
        _add_height = self.dy * blend_width
        _new_fine_extent = (
            self.extent[0] - _add_length,
            self.extent[1] + _add_length,
            self.extent[2] - _add_height,
            self.extent[3] + _add_height,
        )
        _fine_w_pad = Grid(
            _new_fine_extent,
            self.dx,
            dy=self.dy,
            values=_fine_w_pad_values,
            extrapolate=self.extrapolate,
        )
        # 2. Interpolate _fine_w_pad onto coarse
        _coarse_w_fine = _fine_w_pad.interpolate_to(coarse, method="nearest")

        # 3. Perform inverse distance weighting on the points with 0
        _xg, _yg = _coarse_w_fine.create_grid()
        _pts = np.column_stack((_xg.flatten(), _yg.flatten()))
        _vals = _coarse_w_fine.values.flatten()

        # find buffer
        ask_index = _vals == _FILL
        known_index = _vals != _FILL

        _tree = Invdisttree(_pts[known_index], _vals[known_index])
        _vals[ask_index] = _tree(_pts[ask_index], nnear=nnear, eps=eps, p=p)

        plt.scatter(_pts[ask_index, 0], _pts[ask_index, 1], c=_vals[ask_index])
        plt.show()
        # put it back
        _coarse_w_fine.values = _vals.reshape(*_coarse_w_fine.values.shape)
        return _coarse_w_fine

    def plot(
        self,
        hold=False,
        show=True,
        vmin=None,
        vmax=None,
        coarsen=1,
        xlabel=None,
        ylabel=None,
        title=None,
        cbarlabel=None,
        filename=None,
        dpi=600,
    ):
        """Visualize the values in :obj:`Grid`
        Parameters
        ----------
        hold: boolean, optional
            Whether to create a new plot axis.
        Returns
        -------
        ax: handle to axis of plot
            handle to axis of plot.
        """

        x, y = self.create_grid()

        fig, ax = plt.subplots()
        ax.axis("equal")
        c = ax.pcolormesh(
            x[:-1:coarsen, :-1:coarsen],
            y[:-1:coarsen, :-1:coarsen],
            self.values[:-1:coarsen, :-1:coarsen],
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        cbar = plt.colorbar(c)
        if cbarlabel is not None:
            cbar.set_label(cbarlabel)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if hold is False and show:
            plt.show()
        if filename is not None:
            plt.savefig(filename, dpi=dpi)
        return ax

    def build_interpolant(self, method="linear"):
        """Construct a RegularGriddedInterpolant sizing function stores it as
        the `eval` field.
        Parameters
        ----------
        values: array-like
            An an array of values that form the gridded interpolant:w
        """
        lon1, lat1 = self.create_vectors()
        if self.extrapolate:
            _FILL = None
        else:
            _FILL = -999

        fp = RegularGridInterpolator(
            (lon1, lat1),
            self.values,
            method=method,
            bounds_error=False,
            fill_value=_FILL,
        )

        def sizing_function(x):
            return fp(x)

        self.eval = sizing_function
        return self


def _extract_bounds(lons, lats, extent):
    """Extract the indexes of the subregion"""
    # bounds (from DEM)
    blol, blou = np.amin(lons), np.amax(lons)
    blal, blau = np.amin(lats), np.amax(lats)
    # check bounds
    if extent[0] < blol or extent[1] > blou:
        raise ValueError(
            "bounding box "
            + str(extent)
            + " exceeds DEM extents "
            + str((blol, blou, blal, blau))
            + "!"
        )
    if extent[2] < blal or extent[3] > blau:
        raise ValueError(
            "bounding box "
            + str(extent)
            + " exceeds DEM extents "
            + str((blol, blou, blal, blau))
            + "!"
        )
    # latitude lower and upper index
    latli = np.argmin(np.abs(lats - extent[2]))
    latui = np.argmin(np.abs(lats - extent[3])) + 1
    # longitude lower and upper index
    lonli = np.argmin(np.abs(lons - extent[0]))
    lonui = np.argmin(np.abs(lons - extent[1])) + 1

    return latli, latui, lonli, lonui


def _from_netcdf(filename, extent, verbose):
    """Read in digitial elevation model from a NetCDF file"""

    if verbose > 0:
        print("Reading in NetCDF... " + filename)
    # well-known variable names
    wkv_x = ["x", "Longitude", "longitude", "lon"]
    wkv_y = ["y", "Latitude", "latitude", "lat"]
    wkv_z = ["Band1", "z"]
    try:
        with Dataset(filename, "r") as nc_fid:
            for x, y in zip(wkv_x, wkv_y):
                for var in nc_fid.variables:
                    if var == x:
                        lon_name = var
                    if var == y:
                        lat_name = var
            for z in wkv_z:
                for var in nc_fid.variables:
                    if var == z:
                        z_name = var
            lons = nc_fid.variables[lon_name][:]
            lats = nc_fid.variables[lat_name][:]
            latli, latui, lonli, lonui = _extract_bounds(lons, lats, extent)
            topobathy = nc_fid.variables[z_name][latli:latui, lonli:lonui]
            return (
                (lats, slice(latli, latui)),
                (lons, slice(lonli, lonui)),
                topobathy,
            )
    except IOError:
        print("Unable to open file...quitting")
        quit()


class DEM(Grid):
    """Digitial elevation model read in from a tif or NetCDF file
    parent class is a :class:`Grid`
    """

    def __init__(self, dem, extent, verbose=1):

        basename, ext = os.path.splitext(dem)
        if ext.lower() in [".nc"]:
            la, lo, topobathy = _from_netcdf(dem, extent, verbose)
        else:
            raise ValueError(f"DEM file {self.dem} has unknown format {ext[1:]}.")
        self.dem = dem
        # determine grid spacing in degrees
        lats = la[0]
        lons = lo[0]
        dy = np.abs(lats[1] - lats[0])
        dx = np.abs(lons[1] - lons[0])
        super().__init__(
            extent=extent,
            dx=dx,
            dy=dy,
        )
        self.values = topobathy[: self.ny, : self.nx].T
        super().build_interpolant()
