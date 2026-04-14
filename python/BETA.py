import os
import psutil
import argparse
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import shapely.vectorized as sv
from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.abspath(".."))
from function import DOWN_raw
from function import ART_downscale

# =============================================================================
# Example
# python BETA.py -pr CHIRPS -np 2 -ys 2002 -ye 2023 -proc 25

# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-pr", "--product", type=str, required=True)
parser.add_argument("-np", "--npix", type=int, required=True)
parser.add_argument("-ys", "--yys", type=int, required=True)
parser.add_argument("-ye", "--yye", type=int, required=True)
parser.add_argument("-proc", type=int, required=True)

args = parser.parse_args()

product = args.product
npix = args.npix
yy_s = args.yys
yy_e = args.yye
nproc = args.proc

years_num = yy_e - yy_s + 1
NEIBHR = 2*npix+1

# =============================================================================
# PADOVA
# lon_min, lon_max, lat_min, lat_max, area = 11, 12.5, 45, 46, 'PADOVA'
# VENETO
lon_min, lon_max, lat_min, lat_max, area = 10.5, 13.5, 44.5, 47, 'VENETO'
# ITALY
# lon_min, lon_max, lat_min, lat_max, area = 6.5, 19, 36.5, 48, 'ITALY'

# =============================================================================
if area == 'PADOVA':
    GEOMETRY = gpd.read_file(os.path.join('..', 'geometry','Padova.geojson'))
elif area == 'VENETO':
    GEOMETRY = gpd.read_file(os.path.join('..', 'geometry','Veneto.geojson'))
elif area == 'ITALY':
    GEOMETRY = gpd.read_file(os.path.join('..', 'geometry','Italy_simple.geojson'))
else:
    sys.exit("Area not recognized. Please choose 'VENETO' or 'ITALY'.")

GEOMETRY_union = GEOMETRY.unary_union

# =============================================================================
if product == 'IMERG':
    time_reso = '3h'
    filename = 'IMERG_Italy_3h_2001_01_01_2023_12_31.nc'

elif product == 'CMORPH':
    time_reso = '3h'
    filename = 'CMORPH_Italy_3hr_1998_01_01_2023_12_31.nc'

elif product == 'MSWEP':
    time_reso = '3h'
    filename = 'MSWEP_Italy_3h_1980_01_01_2023_12_31.nc'

elif product == 'ERA5':
    time_reso = '3h'
    filename = 'ERA5_Italy_3h_2000_01_01_2023_12_31.nc'

elif product == 'GSMaP':
    time_reso = '3h'
    filename = 'GSMaP_Italy_3h_2002_01_01_2024_12_31.nc'

elif product == 'CHIRPS':
    time_reso = '1dy'
    filename = 'CHIRPS_Italy_1dy_1981_01_01_2024_06_30.nc'

else:
    sys.exit('Product not found')

# =============================================================================
dir_base = os.path.join('/','media','arturo','T9','Data','Italy','Satellite')

print(f'Product: {product}')
dir_input = os.path.join(dir_base,product,time_reso,filename)
DATA = xr.open_dataset(dir_input)
DATA = DATA.sel(time=DATA.time.dt.year.isin(np.arange(yy_s,yy_e+1)))
DATA = DATA.sel(lat=slice(lat_min-1.5, lat_max+1.5), lon=slice(lon_min-1.5, lon_max+1.5))
DATA = DATA.where(DATA>= -0.001)

lats = DATA['lat'].data
lons = DATA['lon'].data

lon2d, lat2d = np.meshgrid(lons, lats)
mask_study = sv.contains(GEOMETRY_union, lon2d, lat2d)

indices_lat, indices_lon = np.where(mask_study)

# =============================================================================
if product == 'IMERG':
    origin = [10, 3]
    target = [0, 24]

elif product == 'CMORPH':
    origin = [25, 3]
    target = [0, 24]

elif product == 'MSWEP':
    origin = [10, 3]
    target = [0, 24]

elif product == 'ERA5':
    origin = [25, 3]
    target = [0, 24]

elif product == 'GSMaP':
    origin = [10, 3]
    target = [0, 24]

elif product == 'CHIRPS':
    origin = [5, 24]
    target = [0, 24]

print(f'Product: {product}')
print(f'xscale : {origin[0]} km to {target[0]} km')
print(f'tscale : {origin[1]} hr to {target[1]} hr')

# =============================================================================
PRE_data_T = DATA.transpose('lon', 'lat', 'time')
time_vector_dt = pd.to_datetime(PRE_data_T['PRE']['time'].values)
DATA_3h = xr.DataArray(PRE_data_T['PRE'],  
                        coords={
                            'lon':PRE_data_T['lon'].values, 
                            'lat':PRE_data_T['lat'].values, 
                            'time':time_vector_dt},
                        dims=('lon', 'lat', 'time'))

# =============================================================================
def compute_for_point(args):
    DATA_3h, la, lo = args
    lat_c = lats[la]
    lon_c = lons[lo]
    box_3h = DOWN_raw.create_box_v2(DATA_3h, lat_c, lon_c, npix)
    pwets, xscales, tscales = DOWN_raw.compute_pwet_xr(box_3h, 1, cube1size=npix, dt=24, tmax=24*6)  
    beta_ = ART_downscale.compute_beta(pwets, origin, target, xscales, tscales)
    return la, lo, beta_

tasks = (
    (DATA_3h, la, lo)
    for la, lo in zip(indices_lat, indices_lon)
)

with Pool(processes=nproc) as pool:
    results = list(pool.imap(compute_for_point, tasks, chunksize=1))

BETA = np.ones((len(lats), len(lons)))*np.nan
for la, lo, beta_ in results:
    BETA[la, lo] = beta_

# =============================================================================
BETA_xr = xr.Dataset(data_vars={
                    "BETA": (("lat","lon"), BETA),
                    },
    coords={'lat': lats, 'lon': lons},
    attrs=dict(description=f"Beta for '{product}' in the '{area}' area bounded by longitudes {lon_min} to {lon_max} and box size '{NEIBHR}x{NEIBHR}'."))

BETA_xr.BETA.attrs["units"] = "dimensionless"
BETA_xr.BETA.attrs["long_name"] = "Itermittency function between two generic scales"
BETA_xr.BETA.attrs["origname"] = "Beta"

DOWN_out = os.path.join('..','output','Beta',f'{area}_Beta_{product}_{time_reso}_{yy_s}_{yy_e}_npix_{npix}.nc')
print(f'Export Data to {DOWN_out}')
BETA_xr.to_netcdf(DOWN_out)
