import os
import json
import time
import psutil
import argparse
import itertools
import numpy as np
import xarray as xr
import pandas as pd
from multiprocessing import Pool
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.abspath(".."))
from function import DOWN_raw

# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-pr", "--product", type=str, required=True)
parser.add_argument("-tr", "--time_reso", type=str, required=True)
parser.add_argument("-ys", "--yys", type=int, required=True)
parser.add_argument("-ye", "--yye", type=int, required=True)

args = parser.parse_args()

product = args.product
time_reso = args.time_reso
yy_s = args.yys
yy_e = args.yye

years_num = yy_e - yy_s + 1
full_years = np.arange(yy_s, yy_e + 1)

# =============================================================================
Tr = np.array([5, 10, 20, 50, 100])

# # Coast area 
# lon_min, lon_max, lat_min, lat_max, area, toll = 12, 12.5, 45.2, 45.7, 'COAST', 0.05
# # Fast area
# lon_min, lon_max, lat_min, lat_max, area, toll = 11.5, 12, 45.5, 46, 'FAST', 0.05
# # Test area
lon_min, lon_max, lat_min, lat_max, area, toll = 11, 12.5, 45, 46.5, 'TEST', 0.05
# # Veneto area
# lon_min, lon_max, lat_min, lat_max, area, toll = 10.5, 13.5, 44.5, 47, 'VENETO', 0.002
# # Italy
# lon_min, lon_max, lat_min, lat_max, area, toll = 6.5, 19, 36.5, 48, 'ITALY', 0.002

# =============================================================================
json_read = f'../json/{product}_{time_reso}.json'
if os.path.exists(json_read):
    with open (json_read) as f:
        param = json.load(f)
else:
    raise SystemExit(f"File not found: {json_read}")

nproces = param['cores']
thresh = param['thresh']
acf_fun = param['acf']

NEIBHR = 2*param['npix']+1

print()
print(f'Json file   : {json_read.split('/')[-1]}')
print(f'Region      : {area}')
print(f'ACF func    : {param['acf']}')
print(f'Threshold   : {thresh}')
print(f'Threads     : {nproces}')
print(f'Neighborhood: {NEIBHR}x{NEIBHR}')
print()

# =============================================================================
print(f'Reading data: {param['file']}')
print()
dir_data = os.path.join(f'../data/{param["file"]}')

if os.path.exists(dir_data):
    pass
else:
    raise FileNotFoundError("Directory doesn't exist")

# =============================================================================
PRE_data = xr.open_dataset(dir_data)
PRE_data = PRE_data.sel(time=PRE_data.time.dt.year.isin(np.arange(yy_s,yy_e+1)))

PRE_data = PRE_data.sel(lat=slice(lat_min-1.5, lat_max+1.5), lon=slice(lon_min-1.5, lon_max+1.5))

lats = PRE_data['lat'].data
lons = PRE_data['lon'].data

lon2d, lat2d = np.meshgrid(lons, lats)

nlon = np.size(lons)
nlat = np.size(lats)
ntime = len(PRE_data['time'])

year_vector = np.unique(pd.to_datetime(PRE_data['time']).year)

PRE_data = PRE_data.where(PRE_data >= 0) 

# =============================================================================
print(f'Extracting lat and lon points for {area} area')
print()
PRE_veneto = PRE_data.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

lat_ref = PRE_veneto.lat.values
lon_ref = PRE_veneto.lon.values

ndices_lat = np.where(np.isin(lats, lat_ref))[0]
ndices_lon = np.where(np.isin(lons, lon_ref))[0]

lon2d_ref, lat2d_ref = np.meshgrid(lon_ref, lat_ref)

del PRE_veneto

# =============================================================================
def downscale_clear(DATA_3h,la,lo, param):

    if (la * len(lon_ref) + lo) % 200 == 0:
        print(f'Processing point {la},{lo}')
    
    lat_c = lat_ref[la]
    lon_c = lon_ref[lo]

    Tr = np.array([5, 10, 20, 50, 100, 200])

    box_3h = DOWN_raw.create_box_v2(DATA_3h, lat_c, lon_c, param['npix'])

    all_nan_mask = box_3h.isnull().all(dim='time')
    num_all_nan_pixels = all_nan_mask.sum().item()

    if num_all_nan_pixels == 0:
        downres = DOWN_raw.downscale(box_3h, Tr, thresh=1.0, L0=0.0001, 
                                    cor_method=param['corr_method'], toll=toll,
                                    acf=param['acf'], save_yearly=True,
                                    maxmiss=40, clat=lat_c, clon=lon_c,
                                    opt_method=param['opt_method'], plot=False)
    else:
        downres = {}

    return downres

# =============================================================================
PRE_data_T = PRE_data.transpose('lon', 'lat', 'time')
time_vector_dt = pd.to_datetime(PRE_data_T['PRE']['time'].values)
DATA_3h = xr.DataArray(PRE_data_T['PRE'],  
                        coords={
                            'lon':PRE_data_T['lon'].values, 
                            'lat':PRE_data_T['lat'].values, 
                            'time':time_vector_dt},
                        dims=('lon', 'lat', 'time'))


# =============================================================================
print('Start with downscale processes')
print()

start_time = time.time()

def compute_for_point(args):
    DATA_3h, la, lo, param = args
    return la, lo, downscale_clear(DATA_3h, la, lo, param)

tasks = (
    (DATA_3h, la, lo, param)
    for la in range(len(lat_ref))
    for lo in range(len(lon_ref))
)

with Pool(processes=nproces) as pool:
    results = list(pool.imap(compute_for_point, tasks, chunksize=1))

end_time = time.time()

elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60
memory_consumed = psutil.virtual_memory().used / 1024**3

# =============================================================================
print(f'Export Downscale process info')
print()
INFO = pd.DataFrame({
                    'Product':product,
                    'Resolution_t':[time_reso],
                    'Ntimes':[ntime],
                    'Region':[area],
                    'Neighborhood':[NEIBHR],
                    'Cores':nproces,
                    'Time(min)':np.round(elapsed_minutes,3),
                    'Memory(Gb)':np.round(memory_consumed,3)
                    })

INFO.to_csv('../csv/DOWN_INFO.csv', 
            mode='a', 
            header=not pd.io.common.file_exists('../csv/DOWN_INFO.csv'), 
            index=False)

# ==============================================================================
print(f'Creating Downscale data')
Tr = np.array([5, 10, 20, 50, 100, 200])

shape = (len(lat_ref), len(lon_ref))

NYs = np.full([years_num, *shape], np.nan)
CYs = np.full([years_num, *shape], np.nan)
WYs = np.full([years_num, *shape], np.nan)

NYd = np.full([years_num, *shape], np.nan)
CYd = np.full([years_num, *shape], np.nan)
WYd = np.full([years_num, *shape], np.nan)

Mev_d = np.zeros((len(Tr), *shape))
Mev_s = np.zeros((len(Tr), *shape))

BETA = np.zeros([len(lat_ref), len(lon_ref)])
GAMMA = np.zeros([len(lat_ref), len(lon_ref)])

FUNVAL = np.zeros([len(lat_ref), len(lon_ref)])

for la, lo, downres in results:
    
    if len(downres) == 0:

        NYs[:, la, lo] = [np.nan]*years_num
        CYs[:, la, lo] = [np.nan]*years_num
        WYs[:, la, lo] = [np.nan]*years_num
        
        NYd[:, la, lo] = [np.nan]*years_num
        CYd[:, la, lo] = [np.nan]*years_num
        WYd[:, la, lo] = [np.nan]*years_num
        
        Mev_d[:, la, lo] = [np.nan]*len(Tr)
        Mev_s[:, la, lo] = [np.nan]*len(Tr)
        
        BETA[la, lo] = np.nan
        GAMMA[la, lo] = np.nan
        
        FUNVAL[la,lo] = np.nan
    
    else:
    
        available_years = downres['YEARS'].astype(int) 
        indices = np.searchsorted(full_years, available_years)
        
        NYs[indices, la, lo] = downres['NYs']
        CYs[indices, la, lo] = downres['CYs']
        WYs[indices, la, lo] = downres['WYs']
        
        NYd[indices, la, lo] = downres['NYd']
        CYd[indices, la, lo] = downres['CYd']
        WYd[indices, la, lo] = downres['WYd']
        
        Mev_d[:, la, lo] = downres['mev_d']
        Mev_s[:, la, lo] = downres['mev_s']
        
        BETA[la, lo] = downres['beta']
        GAMMA[la, lo] = downres['gam_d']
        
        FUNVAL[la,lo] = downres['corr_down_funval']

# =============================================================================
DOWN_xr = xr.Dataset(data_vars={
                    "NYs": (("year","lat","lon"), NYs),
                    "CYs": (("year","lat","lon"), CYs),
                    "WYs": (("year","lat","lon"), WYs),
                    "Mev_s": (("Tr","lat","lon"), Mev_s),
                    "NYd": (("year","lat","lon"), NYd),
                    "CYd": (("year","lat","lon"), CYd),
                    "WYd": (("year","lat","lon"), WYd),
                    "Mev_d": (("Tr","lat","lon"), Mev_d),
                    "BETA": (("lat","lon"), BETA),
                    "GAMMA": (("lat","lon"), GAMMA),
                    "FUNVAL": (("lat","lon"), FUNVAL)
                    },
    coords={'year':full_years,'Tr':Tr,'lat': lat_ref, 'lon': lon_ref},
    attrs=dict(description=f"Downscaling for '{product}' in the '{area}' area bounded by longitudes {lon_min} to {lon_max} and latitudes {lat_min} to {lat_max}, using '{param['acf']}' as the acf function, '{param['thresh']} mm' threshold, '{param['corr_method']}' correlation, optimization method '{param['opt_method']}', toll equal '{toll}' and box size '{NEIBHR}x{NEIBHR}'."))

DOWN_xr.NYs.attrs["units"] = "day"
DOWN_xr.NYs.attrs["long_name"] = "Number of wet days"
DOWN_xr.NYs.attrs["origname"] = "Wet days"

DOWN_xr.CYs.attrs["units"] = "dimensionless"
DOWN_xr.CYs.attrs["long_name"] = "Weibull scale parameter"
DOWN_xr.CYs.attrs["origname"] = "Scale"

DOWN_xr.WYs.attrs["units"] = "dimensionless"
DOWN_xr.WYs.attrs["long_name"] = "Weibull shape parameter"
DOWN_xr.WYs.attrs["origname"] = "Shape"

DOWN_xr.Mev_s.attrs["units"] = "mm/day"
DOWN_xr.Mev_s.attrs["long_name"] = "Satellite Maximum Quantiles"
DOWN_xr.Mev_s.attrs["origname"] = "Sat quantiles"

DOWN_xr.NYd.attrs["units"] = "day"
DOWN_xr.NYd.attrs["long_name"] = "Downscale Number of wet days"
DOWN_xr.NYd.attrs["origname"] = "Down wet days"

DOWN_xr.CYd.attrs["units"] = "dimensionless"
DOWN_xr.CYd.attrs["long_name"] = "Downscale Weibull scale parameter"
DOWN_xr.CYd.attrs["origname"] = "Down scale"

DOWN_xr.WYd.attrs["units"] = "dimensionless"
DOWN_xr.WYd.attrs["long_name"] = "Downscale Weibull shape parameter"
DOWN_xr.WYd.attrs["origname"] = "Down shape"

DOWN_xr.Mev_d.attrs["units"] = "mm/day"
DOWN_xr.Mev_d.attrs["long_name"] = "Downscaling Maximum Quantiles"
DOWN_xr.Mev_d.attrs["origname"] = "Downscaling quantiles"

DOWN_xr.BETA.attrs["units"] = "dimensionless"
DOWN_xr.BETA.attrs["long_name"] = "Itermittency function between two generic scales"
DOWN_xr.BETA.attrs["origname"] = "Beta"

DOWN_xr.GAMMA.attrs["units"] = "dimensionless"
DOWN_xr.GAMMA.attrs["long_name"] = "Variance function between two generic scales"
DOWN_xr.GAMMA.attrs["origname"] = "Gamma"

DOWN_xr.FUNVAL.attrs["units"] = "dimensionless"
DOWN_xr.FUNVAL.attrs["long_name"] = "Minimum error achieved by the optimization"
DOWN_xr.FUNVAL.attrs["origname"] = "Funval"

DOWN_xr.lat.attrs["units"] = "degrees_north"
DOWN_xr.lat.attrs["long_name"] = "Latitude"

DOWN_xr.lon.attrs["units"] = "degrees_east"
DOWN_xr.lon.attrs["long_name"] = "Longitude"

# ==============================================================================
DOWN_out = os.path.join('..','output',f'{area}_DOWN_{product}_{time_reso}_{yy_s}_{yy_e}_npix_{param['npix']}_thr_{param['thresh']}_acf_{param['acf']}_{param['opt_method']}_{param['corr_method']}.nc')
print(f'Export Data to {DOWN_out}')
DOWN_xr.to_netcdf(DOWN_out)