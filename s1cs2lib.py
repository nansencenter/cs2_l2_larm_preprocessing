from datetime import datetime
import os

import cartopy.crs as ccrs
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from shapely import Polygon
from netCDF4 import Dataset

proj_nps = ccrs.NorthPolarStereo()
proj_pc = ccrs.PlateCarree()

class s1_nc_geolocation:
    def __init__(self, ifile):
        self.ifile = ifile

    def read_geolocation(self, read_sar=False):
        try_skip_last_rows = [None, -500, -1000, -2000, -3000, -4000, -5000, -6000, -7000]
        with Dataset(self.ifile) as ds:
            self.raw_shape = self.shape = ds.variables['sar_primary'].shape
            self.sar_grid_line = ds['sar_grid_line'][:]
            self.sar_grid_sample = ds['sar_grid_sample'][:]
            self.sar_grid_latitude = ds['sar_grid_latitude'][:]
            self.sar_grid_longitude = ds['sar_grid_longitude'][:]
            self.shape_ratio = 1.0
            if not read_sar:
                return
            for try_skip_last_row in try_skip_last_rows:
                try:
                    self.sar_primary = ds['sar_primary'][:try_skip_last_row, :]
                    self.sar_secondary = ds['sar_secondary'][:try_skip_last_row, :]
                    read_success = True
                    print('Read success', try_skip_last_row)
                    break
                except RuntimeError:
                    read_success = False
                    #print('Read error', try_skip_last_row)
                    pass
        if not read_success:
            raise RuntimeError(f'Cannot read SAR data from {self.ifile}')
        mask_idx = np.nonzero(
            (np.diff(self.sar_secondary) == 0) *
            (np.diff(self.sar_primary) == 0)
        )
        self.mask = np.zeros(self.sar_primary.shape, dtype=bool)
        self.mask[mask_idx] = True
        self.shape = self.sar_primary.shape
        self.shape_ratio = self.shape[0] / self.raw_shape[0]

    def prepare_interpolators(self):
        sar_grid_x, sar_grid_y, _ = proj_nps.transform_points(proj_pc, self.sar_grid_longitude, self.sar_grid_latitude).T
        self.lndi_x = LinearNDInterpolator((self.sar_grid_line, self.sar_grid_sample), sar_grid_x)
        self.lndi_y = LinearNDInterpolator((self.sar_grid_line, self.sar_grid_sample), sar_grid_y)
        self.lndi_s = LinearNDInterpolator((sar_grid_x, sar_grid_y), self.sar_grid_sample)
        self.lndi_l = LinearNDInterpolator((sar_grid_x, sar_grid_y), self.sar_grid_line)

    def get_border(self, border_pixels=10):
        border_line = np.hstack([
            np.linspace(0, 0, border_pixels),
            np.linspace(0, self.shape[0] - 1, border_pixels),
            np.linspace(self.shape[0] - 1, self.shape[0] - 1, border_pixels),
            np.linspace(self.shape[0] - 1, 0, border_pixels),
        ]).astype(int)

        border_sample = np.hstack([
            np.linspace(0, self.shape[1] - 1, border_pixels),
            np.linspace(self.shape[1] - 1, self.shape[1] - 1, border_pixels),
            np.linspace(self.shape[1] - 1, 0, border_pixels),
            np.linspace(0, 0, border_pixels),
        ]).astype(int)

        border_x = self.lndi_x((border_line, border_sample))
        border_y = self.lndi_y((border_line, border_sample))
        return border_x, border_y

    def get_datetime(self):
        return datetime.strptime(os.path.basename(self.ifile).split('_')[4], '%Y%m%dT%H%M%S')

    def get_border_polygon(self):
        border_x, border_y = self.get_border()
        return Polygon(zip(border_x, border_y))

def labelize(sir, bins):
    sir_labels = np.zeros(sir.shape, 'uint8') + 255
    for class_no, (b0, b1) in enumerate(zip(bins[:-1], bins[1:])):
        sir_labels[(sir >= b0) * (sir < b1)] = class_no
    return sir_labels


class IrregularGridInterpolator2(object):
    def __init__(self, x0, y0, x1, y1, triangles=None):
        """
        Parameters:
        -----------
        x0 : np.ndarray(float)
            x-coords of source points
        y0 : np.ndarray(float)
            y-coords of source points
        x1 : np.ndarray(float)
            x-coords of destination points
        y1 : np.ndarray(float)
            y-coords of destination points
        triangles : np.ndarray(int)
            shape (num_triangles, 3)
            indices of nodes for each triangle

        Sets:
        -----
        self.inside: np.ndarray(bool)
            shape = (num_target_points,)
        self.vertices: np.ndarray(int)
            shape = (num_good_target_points, 3)
            good target points are those inside the source triangulation
        self.weights: np.ndarray(float)
            shape = (num_good_target_points, 3)
            good target points are those inside the source triangulation

        Follows this suggestion:
        https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
        x_target[i] = \\sum_{j=0}^2 weights[i, j]*x_source[vertices[i, j]]
        y_target[i] = \\sum_{j=0}^2 weights[i, j]*y_source[vertices[i, j]]
        We can do (linear) interpolation by replacing x_target, x_source with z_target, z_source
        where z_source is the field to be interpolated and z_target is the interpolated field
        """

        # define and triangulate source points
        self.src_shape = x0.shape
        self.src_points = np.array([x0.flatten(), y0.flatten()]).T
        self.tri = Delaunay(np.array([x0.flatten(), y0.flatten()]).T) #Triangulation(x0.flatten(), y0.flatten(), triangles=triangles)
        #self.tri_finder = self.tri.get_trifinder()
        #self.num_triangles = len(self.tri.triangles)
        self.num_triangles = len(self.tri.simplices)
        self._set_transform()

        # define target points
        self.dst_points = np.array([x1.flatten(), y1.flatten()]).T
        self.dst_shape = x1.shape
        #self.triangle_map = self.tri_finder(x1, y1)
        self.triangle_map = self.tri.find_simplex(np.array([x1.flatten(), y1.flatten()]).T)
        self.dst_mask = (self.triangle_map < 0)
        self.triangle_map[self.dst_mask] = 0
        self.inside = ~self.dst_mask.flatten()

        """
        get barycentric coords
        https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Barycentric_coordinates_on_triangles
        each row of bary is (lambda_1, lambda_2) for 1 destination point
        """
        d = 2
        inds = self.triangle_map.flatten()[self.inside]
        #self.vertices = np.take(self.tri.triangles, inds, axis=0)
        self.vertices = np.take(self.tri.simplices, inds, axis=0)
        temp = np.take(self.transform, inds, axis=0)
        delta = self.dst_points[self.inside] - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)

        # set weights
        self.weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    def _set_transform(self):
        """
        Used for getting the barycentric coordinates on a triangle.
        Follows:
        https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Barycentric_coordinates_on_triangles

        Sets:
        -----
        self.transform : numpy.ndarray
            For the i-th triangle,
                self.transform[i] = [[a', b'], [c', d'], [x_3, y3]]
            where the first 2 rows are the inverse of the matrix T in the wikipedia link
            and (x_3, y_3) are the coordinates of the 3rd vertex of the triangle
        """
        #x = self.tri.x[self.tri.triangles]
        #y = self.tri.y[self.tri.triangles]
        x = self.tri.points[:,0][self.tri.simplices]
        y = self.tri.points[:,1][self.tri.simplices]
        a = x[:,0] - x[:,2]
        b = x[:,1] - x[:,2]
        c = y[:,0] - y[:,2]
        d = y[:,1] - y[:,2]
        det = a*d-b*c

        self.transform = np.zeros((self.num_triangles, 3, 2))
        self.transform[:,0,0] = d/det
        self.transform[:,0,1] = -b/det
        self.transform[:,1,0] = -c/det
        self.transform[:,1,1] = a/det
        self.transform[:,2,0] = x[:,2]
        self.transform[:,2,1] = y[:,2]

    def interp_field(self, fld, method='linear'):
        """
        Interpolate field from elements elements or nodes of source triangulation
        to destination points

        Parameters:
        -----------
        fld: np.ndarray
            field to be interpolated
        method : str
            interpolation method if interpolating from nodes
            - 'linear'  : linear interpolation
            - 'nearest' : nearest neighbour

        Returns:
        -----------
        fld_interp : np.ndarray
            field interpolated onto the destination points
        """
        if fld.shape == self.src_shape:
            return self._interp_nodes(fld, method=method)
        fld_ = fld.flatten()
        if len(fld_) == self.num_triangles:
            return self._interp_elements(fld_)
        msg = f"""Field to interpolate should have the same size as the source points
        i.e. {self.src_shape}, or be a vector with the same number of triangles
        as the source triangulation i.e. self.num_triangles"""
        raise ValueError(msg)

    def _interp_elements(self, fld):
        """
        Interpolate field from elements of source triangulation to destination points

        Parameters:
        -----------
        fld: np.ndarray
            field to be interpolated

        Returns:
        -----------
        fld_interp : np.ndarray
            field interpolated onto the destination points
        """
        fld_interp = fld[self.triangle_map]
        fld_interp[self.dst_mask] = np.nan
        return fld_interp

    def _interp_nodes(self, fld, method='linear'):
        """
        Interpolate field from nodes of source triangulation to destination points

        Parameters:
        -----------
        fld: np.ndarray
            field to be interpolated
        method : str
            interpolation method
            - 'linear'  : linear interpolation
            - 'nearest' : nearest neighbour

        Returns:
        -----------
        fld_interp : np.ndarray
            field interpolated onto the destination points
        """
        ndst = self.dst_points.shape[0]
        fld_interp = np.full((ndst,), np.nan)
        w = self.weights
        if method == 'linear':
            # sum over the weights for each node of triangle
            v = self.vertices # shape = (ngood,3)
            fld_interp[self.inside] = np.einsum(
                    'nj,nj->n', np.take(fld.flatten(), v), w)

        elif method == 'nearest':
            # find the node of the triangle with the maximum weight
            v = np.array(self.vertices) # shape = (ngood,3)
            v = v[np.arange(len(w), dtype=int), np.argmax(w, axis=1)] # shape = (ngood,)
            fld_interp[self.inside] = fld.flatten()[v]

        else:
            raise ValueError("'method' should be 'nearest' or 'linear'")

        return fld_interp.reshape(self.dst_shape)


class s1_pp_geolocation(s1_nc_geolocation):
    def read_geolocation(self, read_sar=False):
        with np.load(self.ifile) as npz:
            self.sar_grid_line = npz['sar_grid_line']
            self.sar_grid_sample = npz['sar_grid_sample']
            self.sar_grid_latitude = npz['sar_grid_latitude']
            self.sar_grid_longitude = npz['sar_grid_longitude']
            self.shape = npz['shape']
            if read_sar:
                self.sar_primary = npz['hh']
                self.sar_secondary = npz['hv']
                self.mask = npz['mask']
