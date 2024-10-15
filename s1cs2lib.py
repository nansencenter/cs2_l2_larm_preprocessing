from datetime import datetime
import os

import cartopy.crs as ccrs
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from shapely import Polygon
import xarray as xa
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
                    print('Read error', try_skip_last_row)
                    pass
            if not read_success:
                raise RuntimeError(f'Cannot read SAR data from {self.ifile}')
            self.sar_mask = self.sar_primary.mask + (self.sar_primary.filled(-30) < -25)
            good_line_index = np.nonzero(self.sar_grid_line < self.sar_primary.shape[0])[0]
            self.sar_grid_line = self.sar_grid_line[good_line_index]
            self.sar_grid_sample = self.sar_grid_sample[good_line_index]
            self.sar_grid_latitude = self.sar_grid_latitude[good_line_index]
            self.sar_grid_longitude = self.sar_grid_longitude[good_line_index]
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
