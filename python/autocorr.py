import os
import sys
import argparse
import numpy as np
import xarray as xr
import pandas as pd

from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.abspath(".."))
from function import DOWN_raw

# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-pr", "--product", type=str, required=True)
parser.add_argument("-nj", "--n_jobs", type=int, required=True)

npix = 2
tscale = 24
method = 'pearson'

product = vars(parser.parse_args())['product']
n_jobs = vars(parser.parse_args())['n_jobs']

# =============================================================================
lon_min, lon_max, lat_min, lat_max, area, toll = 6.5, 19, 36.5, 48, 'Italy', 0.002

dir_base = os.path.join('..','..','data')
dir_dist = os.path.join('..','..','output','autocorr',f'{area}_{product}_Npix_{npix}_dist.csv')
dir_para = os.path.join('..','..','output','autocorr',f'{area}_{product}_Npix_{npix}_mar_{method}.csv')

# =============================================================================
if product == 'IMERG':
    filename = 'IMERG_Italy_1dy_2000_06_01_2024_02_29.nc'

elif product == 'CMORPH':
    filename = 'CMORPH_Italy_3hr_1998_01_01_2023_12_31.nc'

elif product == 'MSWEP':
    filename = 'MSWEP_Italy_3h_1980_01_01_2023_12_31.nc'

elif product == 'ERA5':
    filename = 'ERA5_Italy_3h_2000_01_01_2023_12_31.nc'

elif product == 'GSMaP':
    filename = 'GSMaP_Italy_3h_2002_01_01_2024_12_31.nc'

elif product == 'CHIRPS':
    filename = 'CHIRPS_Italy_1dy_1981_01_01_2024_06_30.nc'

else:
    sys.exit('Product not found')

# =============================================================================
print(f'Product: {product}')
dir_input = os.path.join(dir_base,filename)
DATA = xr.open_dataset(dir_input)
DATA = DATA.sel(lat=slice(lat_min-1.5, lat_max+1.5), lon=slice(lon_min-1.5, lon_max+1.5))
DATA = DATA.where(DATA>= -0.001)

lats = DATA['lat'].data
lons = DATA['lon'].data

lon2d, lat2d = np.meshgrid(lons, lats)

PRE_study = DATA.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

lat_ref = PRE_study.lat.values
lon_ref = PRE_study.lon.values

del PRE_study

# =============================================================================
lat_c = lat_ref[int(len(lat_ref)/2)]
lon_c = lon_ref[int(len(lon_ref)/2)]

box_3h = DOWN_raw.create_box_v2(DATA, lat_c, lon_c, npix)
DAILY = box_3h.resample(time ='{}h'.format(tscale)).sum(dim='time', skipna=False)
DAILY = DAILY.dropna(dim='time', how='all')

rcorr_ref = DOWN_raw.grid_corr(DAILY.PRE, plot=False, thresh=1, cor_method='pearson')
rcorr_ref = pd.DataFrame(rcorr_ref)
xdist = rcorr_ref.sort_values(by='vdist')['vdist'].values
pd_xdist = pd.DataFrame({'dist':xdist})

print(f'Export dataframe to: {dir_dist}')
pd_xdist.to_csv(dir_dist, header=True, index=False)

# =============================================================================
def process_point(la, lo):
    try:
        lat_c = lat_ref[la]
        lon_c = lon_ref[lo]

        box_3h = DOWN_raw.create_box_v2(DATA, lat_c, lon_c, npix)
        DAILY = box_3h.resample(time=f'{tscale}h').sum(dim='time', skipna=False)
        DAILY = DAILY.dropna(dim='time', how='all')

        rcorr = DOWN_raw.grid_corr(
            DAILY.PRE,
            plot=False,
            thresh=1,
            cor_method='pearson'
        )

        return (lat_c, lon_c, rcorr['eps_s'], rcorr['alp_s'])

    except:
        return None


results = Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(process_point)(la, lo)
    for la in range(len(lat_ref))
    for lo in range(len(lon_ref))
)

# filtra falhas
results = [r for r in results if r is not None]

# dataframe final
pd_corr = pd.DataFrame(results, columns=['lat','lon','epsilon','alpha'])

print(f'Export dataframe to: {dir_para}')
pd_corr.to_csv(dir_para, header=True, index=False)

# =============================================================================
