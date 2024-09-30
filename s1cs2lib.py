from datetime import datetime
import os

import cartopy.crs as ccrs
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from shapely import Polygon
import xarray as xa

proj_nps = ccrs.NorthPolarStereo()
proj_pc = ccrs.PlateCarree()

class s1_nc_geolocation:
    def __init__(self, ifile):
        self.ifile = ifile

    def read_geolocation(self):
        with xa.open_dataset(self.ifile) as ds:
            self.shape = ds['sar_primary'].shape
            self.sar_grid_line = ds['sar_grid_line'].to_numpy()
            self.sar_grid_sample = ds['sar_grid_sample'].to_numpy()
            self.sar_grid_latitude = ds['sar_grid_latitude'].to_numpy()
            self.sar_grid_longitude = ds['sar_grid_longitude'].to_numpy()

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
