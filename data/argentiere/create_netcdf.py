#!/usr/bin/env python3
"""
Create NetCDF file from 100m resolution TIF files
"""

import numpy as np
import netCDF4 as nc
import rasterio
import os
from scipy.ndimage import distance_transform_edt

# Define input TIF files
base_dir = "/home/klleshi/Downloads/argentiere-20260216T152116Z-1-001/argentiere"
tif_files = {
    'dem_2012': os.path.join(base_dir, 'dem_2012_100m_utm32n.tif'),
    'vx': os.path.join(base_dir, 'vx_100m_utm32n.tif'),
    'vy': os.path.join(base_dir, 'vy_100m_utm32n.tif'),
    'thickness_2012': os.path.join(base_dir, 'thickness_2012_100m_utm32n.tif'),
}

# Output NetCDF file
output_nc = os.path.join(base_dir, 'input_100m.nc')

def read_geotiff(filename):
    """Read a GeoTIFF file and return data and geotransform"""
    with rasterio.open(filename) as src:
        data = src.read(1)
        transform = src.transform
        # Convert rasterio transform to GDAL-style geotransform
        # (x_origin, pixel_width, 0, y_origin, 0, pixel_height)
        geotransform = (transform.c, transform.a, 0, transform.f, 0, transform.e)
        nx = src.width
        ny = src.height

    return data, geotransform, nx, ny

def create_coordinates(geotransform, nx, ny):
    """Create x and y coordinate arrays from geotransform"""
    x_origin = geotransform[0]
    y_origin = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]  # This is negative

    # Create coordinate arrays (center of pixels)
    x = x_origin + (np.arange(nx) + 0.5) * pixel_width
    y = y_origin + (np.arange(ny) + 0.5) * pixel_height

    return x, y

def create_icemask(thickness_data):
    """Create ice mask based on thickness data"""
    # Ice mask: 1 where thickness > 0, 0 otherwise
    # Handle NaN values
    icemask = np.zeros_like(thickness_data)
    icemask[~np.isnan(thickness_data) & (thickness_data > 0)] = 1
    return icemask


def fill_nan_nearest(arr):
    """Fill NaN pixels with the nearest non-NaN value (nearest-neighbour extrapolation)."""
    arr = arr.astype(np.float32)
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return arr
    # distance_transform_edt returns indices of the nearest non-NaN pixel
    _, (iy, ix) = distance_transform_edt(nan_mask, return_indices=True)
    filled = arr.copy()
    filled[nan_mask] = arr[iy[nan_mask], ix[nan_mask]]
    return filled

# Read all TIF files
print("Reading TIF files...")
dem_2012, geotransform, nx, ny = read_geotiff(tif_files['dem_2012'])
vx, _, _, _ = read_geotiff(tif_files['vx'])
vy, _, _, _ = read_geotiff(tif_files['vy'])
thickness_2012, _, _, _ = read_geotiff(tif_files['thickness_2012'])

print(f"Grid dimensions: {ny} x {nx}")
print(f"Resolution: {geotransform[1]:.2f} m x {abs(geotransform[5]):.2f} m")

# Create coordinates
x, y = create_coordinates(geotransform, nx, ny)
print(f"X range: {x.min():.2f} to {x.max():.2f}")
print(f"Y range: {y.min():.2f} to {y.max():.2f}")

# Create ice mask (pure glacier outline from thickness > 0)
icemask = create_icemask(thickness_2012)
print(f"Ice mask: {int(np.sum(icemask))} pixels with ice")

# usurfobs: keep the raw DEM with NaNs preserved (no fabrication).
# Used by DA cost misfit_usurf / diagnostics.
usurfobs = dem_2012.astype(np.float32)
n_nan_usurf = int(np.isnan(usurfobs).sum())
print(f"usurfobs: {n_nan_usurf} NaN pixels preserved")

# usurf: cleaned copy (NaN filled via nearest-neighbour). Fed to the emulator
# as a required input. Off-glacier NaNs are scaffolding only; thk=0 and
# icemaskobs=0 there, so these filled values never enter any cost term.
usurf_clean = fill_nan_nearest(dem_2012)
assert not np.isnan(usurf_clean).any(), "usurf (emulator input) must not contain NaN"

# icemaskobs: refined valid-data mask. Starts from icemask, then zeros out
# any pixel where an observation (usurfobs, uvelsurfobs, vvelsurfobs) is NaN.
# Every DA cost term that gates on icemaskobs > 0.5 (misfit_usurf, regu_thk,
# cost_divfluxfcz, total_cost icemask term, ...) will automatically skip
# these pixels. Velocity misfit also separately guards NaN via ~is_nan.
icemaskobs = icemask.copy()
invalid_obs = np.isnan(usurfobs) | np.isnan(vx) | np.isnan(vy)
icemaskobs[invalid_obs] = 0
n_dropped = int(((icemask > 0.5) & invalid_obs).sum())
print(f"icemaskobs: dropped {n_dropped} on-glacier pixels due to NaN obs "
      f"({int(np.sum(icemaskobs))} valid pixels remain)")

# Create NetCDF file
print(f"\nCreating NetCDF file: {output_nc}")
with nc.Dataset(output_nc, 'w', format='NETCDF4') as ncfile:

    # Create dimensions
    ncfile.createDimension('x', nx)
    ncfile.createDimension('y', ny)

    # Create coordinate variables
    x_var = ncfile.createVariable('x', 'f4', ('x',))
    x_var.units = 'm'
    x_var.long_name = 'x'
    x_var.standard_name = 'x'
    x_var.axis = 'X'
    x_var[:] = x

    y_var = ncfile.createVariable('y', 'f4', ('y',))
    y_var.units = 'm'
    y_var.long_name = 'y'
    y_var.standard_name = 'y'
    y_var.axis = 'Y'
    y_var[:] = y

    # usurf: clean surface (NaN filled) — fed to the iceflow emulator.
    usurf_var = ncfile.createVariable('usurf', 'f4', ('y', 'x'))
    usurf_var.long_name = 'Surface Topography (NaN-filled, emulator input)'
    usurf_var.units = 'm'
    usurf_var.standard_name = 'usurf'
    usurf_var[:, :] = usurf_clean

    # usurfobs: raw DEM with NaNs preserved — DA observation reference.
    usurfobs_var = ncfile.createVariable('usurfobs', 'f4', ('y', 'x'))
    usurfobs_var.long_name = 'Observed Surface Topography (raw DEM, NaNs preserved)'
    usurfobs_var.units = 'm'
    usurfobs_var.standard_name = 'usurfobs'
    usurfobs_var[:, :] = usurfobs

    # Create uvelsurfobs (vx)
    uvelsurfobs_var = ncfile.createVariable('uvelsurfobs', 'f4', ('y', 'x'))
    uvelsurfobs_var.long_name = 'x surface velocity of ice'
    uvelsurfobs_var.units = 'm/y'
    uvelsurfobs_var.standard_name = 'uvelsurfobs'
    uvelsurfobs_var[:, :] = vx

    # Create vvelsurfobs (vy)
    vvelsurfobs_var = ncfile.createVariable('vvelsurfobs', 'f4', ('y', 'x'))
    vvelsurfobs_var.long_name = 'y surface velocity of ice'
    vvelsurfobs_var.units = 'm/y'
    vvelsurfobs_var.standard_name = 'vvelsurfobs'
    vvelsurfobs_var[:, :] = vy

    # icemask: pure glacier outline (thickness > 0).
    icemask_var = ncfile.createVariable('icemask', 'f4', ('y', 'x'))
    icemask_var.long_name = 'Ice mask'
    icemask_var.units = 'no unit'
    icemask_var.standard_name = 'icemask'
    icemask_var[:, :] = icemask

    # icemaskobs: valid-data mask used by DA cost gating. Equals icemask minus
    # any pixel where an observation (usurfobs / uvelsurfobs / vvelsurfobs)
    # is NaN, so NaN obs never enter cost/regularization terms.
    icemaskobs_var = ncfile.createVariable('icemaskobs', 'f4', ('y', 'x'))
    icemaskobs_var.long_name = 'Observation validity mask (icemask ∧ obs non-NaN)'
    icemaskobs_var.units = 'no unit'
    icemaskobs_var.standard_name = 'icemaskobs'
    icemaskobs_var[:, :] = icemaskobs

    # Create thkinit (thickness 2012)
    thkinit_var = ncfile.createVariable('thkinit', 'f4', ('y', 'x'))
    thkinit_var.long_name = 'Ice Thickness'
    thkinit_var.units = 'm'
    thkinit_var.standard_name = 'thkinit'
    thkinit_var[:, :] = thickness_2012

    # Add global attributes
    ncfile.title = 'Argentiere Glacier Input Data - 100m resolution'
    ncfile.source = 'Generated from 100m resolution GeoTIFF files'
    ncfile.history = 'Created with create_netcdf.py'

print("\nNetCDF file created successfully!")
print(f"Output: {output_nc}")

# Verify the created file
print("\nVerifying NetCDF file structure:")
with nc.Dataset(output_nc, 'r') as ncfile:
    print(f"Dimensions: {dict(ncfile.dimensions)}")
    print(f"Variables: {list(ncfile.variables.keys())}")
    print(f"\nVariable shapes:")
    for var_name in ncfile.variables:
        var = ncfile.variables[var_name]
        print(f"  {var_name}: {var.shape}")
