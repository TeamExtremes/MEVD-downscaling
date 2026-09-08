import os
import time
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

os.system("cls" if os.name == "nt" else "clear")
start_time = time.time()

# =============================================================================
# Example
# python GAMMA.py -pr IMERG -np 2 -ys 2002 -ye 2023 -proc 25
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
NEIBHR = 2*npix + 1

# =============================================================================
# DIRECTORIES
# =============================================================================

data_input = os.path.join('/','media','arturo','T9','Data','Italy','Satellite')
dir_out = os.path.join('..','output','Gamma')

# =============================================================================
# PARAMETERS
# =============================================================================

thresh = 1
acf = 'mar'
parnames = ['eps', 'alp'] if acf == 'mar' else ['d0', 'mu0']

# =============================================================================
# STUDY AREA
# =============================================================================

# PADOVA
# lon_min, lon_max, lat_min, lat_max, area = 11, 12.5, 45, 46, 'PADOVA'

# VENETO
lon_min, lon_max, lat_min, lat_max, area = 10.5, 13.5, 44.5, 47, 'VENETO'

# ITALY
# lon_min, lon_max, lat_min, lat_max, area = 6.5, 19, 36.5, 48, 'ITALY'

# =============================================================================

if area == 'PADOVA':
    GEOMETRY = gpd.read_file(os.path.join('..', 'geometry', 'Padova.geojson'))

elif area == 'VENETO':
    GEOMETRY = gpd.read_file(os.path.join('..', 'geometry', 'Veneto.geojson'))

elif area == 'ITALY':
    GEOMETRY = gpd.read_file(os.path.join('..', 'geometry', 'Italy_simple.geojson'))

else:
    sys.exit("Area not recognized. Please choose 'PADOVA', 'VENETO' or 'ITALY'.")

GEOMETRY_union = GEOMETRY.unary_union

# =============================================================================
# PRODUCT
# =============================================================================

if product == 'IMERG':
    filename = 'IMERG_Italy_3h_2001_01_01_2023_12_31.nc'
    L1, dt, time_reso = 10, 3, '3h'
    origin_x_t = [10, 24]
    target_x_t = [0.001, 24]


elif product == 'CMORPH':
    filename = 'CMORPH_Italy_3hr_1998_01_01_2023_12_31.nc'
    L1, dt, time_reso = 25, 3, '3h'
    origin_x_t = [25, 24]
    target_x_t = [0.001, 24]


elif product == 'MSWEP':
    filename = 'MSWEP_Italy_3h_1980_01_01_2023_12_31.nc'
    L1, dt, time_reso = 10, 3, '3h'
    origin_x_t = [10, 24]
    target_x_t = [0.001, 24]


elif product == 'ERA5':
    filename = 'ERA5_Italy_3h_2000_01_01_2023_12_31.nc'
    L1, dt, time_reso = 25, 3, '3h'
    origin_x_t = [25, 24]
    target_x_t = [0.001, 24]


elif product == 'GSMaP':
    filename = 'GSMaP_Italy_3h_2002_01_01_2024_12_31.nc'
    L1, dt, time_reso = 10, 3, '3h'
    origin_x_t = [10, 24]
    target_x_t = [0.001, 24]


elif product == 'CHIRPS':
    filename = 'CHIRPS_Italy_1dy_1981_01_01_2024_06_30.nc'
    L1, dt, time_reso = 5, 24, '1dy'
    origin_x_t = [5, 24]
    target_x_t = [0.001, 24]

else:
    sys.exit('Product not found')

# =============================================================================
# INFORMATION
# =============================================================================

print()
print(f'Product: {product}')
print(f'Area   : {area}')
print(f'xscale : {origin_x_t[0]} km to {target_x_t[0]} km')
print(f'tscale : {origin_x_t[1]} hr to {target_x_t[1]} hr')
print(f'acf    : {acf}')
print(f'thresh : {thresh}')
print()

# =============================================================================
# INPUT DATA
# =============================================================================

dir_base = os.path.join(data_input)

dir_input = os.path.join(dir_base, product, time_reso, filename)

DATA = xr.open_dataset(dir_input)
DATA = DATA.sel(time=DATA.time.dt.year.isin(np.arange(yy_s, yy_e + 1)))
DATA = DATA.sel(lat=slice(lat_min - 1.5,lat_max + 1.5),lon=slice(lon_min - 1.5,lon_max + 1.5))
DATA = DATA.where(DATA >= -0.001)

# =============================================================================
# COORDINATES AND STUDY AREA MASK
# =============================================================================

lats = DATA['lat'].data
lons = DATA['lon'].data

lon2d, lat2d = np.meshgrid(lons, lats)
mask_study = sv.contains(GEOMETRY_union,lon2d,lat2d)
indices_lat, indices_lon = np.where(mask_study)

# =============================================================================
# DATA ARRAY
# =============================================================================

PRE_data_T = DATA.transpose('lon','lat','time')

time_vector_dt = pd.to_datetime(PRE_data_T['PRE']['time'].values)

DATA_3h = xr.DataArray(
                    PRE_data_T['PRE'],
                    coords={'lon': PRE_data_T['lon'].values,
                            'lat': PRE_data_T['lat'].values,
                            'time': time_vector_dt},
                    dims=('lon', 'lat', 'time'))

# =============================================================================
# GAMMA CALCULATION
# =============================================================================

def compute_for_point(args):
    (
        DATA_3h,
        la,
        lo,
        thresh,
        npix,
        origin_x_t,
        target_x_t,
        acf,
    ) = args

    lat_c = lats[la]
    lon_c = lons[lo]

    # -------------------------------------------------------------------------
    # Create spatial box
    # -------------------------------------------------------------------------

    BOX = DOWN_raw.create_box_v2(DATA_3h,lat_c,lon_c,npix)

    if np.isnan(BOX).all():
        return la, lo, np.nan

    # -------------------------------------------------------------------------
    # Spatial correlation
    # -------------------------------------------------------------------------

    rcorr_pearson = DOWN_raw.grid_corr(BOX,plot=False,thresh=thresh,cor_method='pearson')

    # -------------------------------------------------------------------------
    # Fit spatial correlation function
    # -------------------------------------------------------------------------

    dcorr = DOWN_raw.down_corr(
        rcorr_pearson['vdist'],
        rcorr_pearson['vcorr'],
        origin_x_t[0],
        acf=acf,
        use_ave=True,
        opt_method='genetic',
        toll=0.005,
        plot=False
    )

    # -------------------------------------------------------------------------
    # Parameters of the autocorrelation function
    # -------------------------------------------------------------------------

    par_acf = (dcorr[f'{parnames[0]}_d'],dcorr[f'{parnames[1]}_d'])

    # -------------------------------------------------------------------------
    # Calculate GAMMA
    # -------------------------------------------------------------------------

    gamma_ = DOWN_raw.vrf(origin_x_t[0],target_x_t[0],par_acf,acf=acf)

    return la, lo, gamma_

# =============================================================================
# PARALLEL PROCESSING
# =============================================================================

tasks = (
    (
        DATA_3h,
        la,
        lo,
        thresh,
        npix,
        origin_x_t,
        target_x_t,
        acf,
    )
    for la, lo in zip(
        indices_lat,
        indices_lon
    )
)

with Pool(processes=nproc) as pool:
        results = list(
            pool.imap(
                compute_for_point,
                tasks,
                chunksize=1
            )
        )

# =============================================================================
# STORE GAMMA
# =============================================================================

GAMMA = np.ones((len(lats), len(lons))) * np.nan

for la, lo, gamma_ in results:
    GAMMA[la, lo] = gamma_

# =============================================================================
# EXPORT GAMMA TO NETCDF
# =============================================================================

print()

GAMMA_xr = xr.Dataset(
                data_vars={"GAMMA": (("lat", "lon"),GAMMA),},
                coords={'lat': lats,'lon': lons},
                attrs=dict(
                description=(
                    f"Gamma for '{product}' in the '{area}' area "
                    f"bounded by longitudes {lon_min} to {lon_max} "
                    f"and box size '{NEIBHR}x{NEIBHR}'.")))

GAMMA_xr.GAMMA.attrs["units"] = "dimensionless"
GAMMA_xr.GAMMA.attrs["long_name"] = ("Variance reduction function between two spatial scales")
GAMMA_xr.GAMMA.attrs["origname"] = "Gamma"
GAMMA_xr.GAMMA.attrs["acf"] = acf
GAMMA_xr.GAMMA.attrs["threshold"] = thresh
GAMMA_xr.GAMMA.attrs["origin_scale_km"] = origin_x_t[0]
GAMMA_xr.GAMMA.attrs["target_scale_km"] = target_x_t[0]

# =============================================================================
# OUTPUT
# =============================================================================

DOWN_out = os.path.join(dir_out,f'{area}_Gamma_{product}_{time_reso}_{yy_s}_{yy_e}_npix_{npix}.nc')

print(f'Export Data to {DOWN_out}')

GAMMA_xr.to_netcdf(DOWN_out)

print()

# =============================================================================
# EXECUTION TIME
# =============================================================================

elapsed_time = time.time() - start_time

hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
print()