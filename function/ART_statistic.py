import os
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from pyproj import CRS, Transformer
from scipy.interpolate import griddata
from rasterio.transform import from_origin
from sklearn.linear_model import LinearRegression

def linear_regression(OBS,TARGET):
    
    OBS = np.array(OBS)
    TARGET = np.array(TARGET)
    
    mask = ~np.isnan(OBS) & ~np.isnan(TARGET)
    obs_clean = OBS[mask].reshape(-1, 1) 
    down_clean = TARGET[mask]

    reg = LinearRegression()
    reg.fit(obs_clean, down_clean)

    # Obtener el slope (pendiente)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    
    x_line = np.linspace(np.min(obs_clean), np.max(obs_clean), 100).reshape(-1, 1)
    y_line = reg.predict(x_line)

    return x_line, y_line, slope

def inverse_distance_weighting(station_points, station_values, grid_points, power=2, n_neighbors=10, max_distance=None):
    """
    Implementación de Inverse Distance Weighting (IDW) para interpolación.
    """

    # Construir árbol KD para búsqueda eficiente
    tree = cKDTree(station_points)

    # Buscar vecinos más cercanos para cada punto de grilla
    distances, indices = tree.query(grid_points, k=min(n_neighbors, len(station_points)))

    grid_values = np.zeros(len(grid_points))

    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        # Filtrar por distancia máxima si se especifica
        if max_distance is not None:
            valid_mask = dists <= max_distance
            if not np.any(valid_mask):
                grid_values[i] = np.nan
                continue
            dists = dists[valid_mask]
            idxs = idxs[valid_mask]

        # Evitar división por cero
        dists = np.maximum(dists, 1e-10)

        # Calcular pesos
        weights = 1.0 / (dists ** power)
        weights /= weights.sum()
        
        # Calcular valor interpolado
        grid_values[i] = np.sum(weights * station_values[idxs])

    return grid_values

def interpolate_factors_to_grid(stations_df, sat_data, method='linear', 
                                use_idw=True, power=2, n_neighbors=10, max_distance=None):
    """
    Interpola los factores de las estaciones a la grilla completa.

    Parameters:
    -----------
    stations_df : DataFrame
        Con columnas 'lat', 'lon', 'factor'
    sat_data : xarray.DataArray
        Datos satelitales para obtener la grilla de destino
    method : str
        Método de interpolación ('linear', 'cubic', 'nearest', 'idw')
    use_idw : bool
        Si True, usa Inverse Distance Weighting en lugar de griddata
    power : float
        Potencia para IDW (solo si use_idw=True)
    n_neighbors : int
        Número de vecinos para IDW (solo si use_idw=True)
    max_distance : float
        Distancia máxima para considerar vecinos en IDW (grados)

    Returns:
    --------
    factor_grid : xarray.DataArray
        Factores interpolados en la misma grilla que sat_data
    """

    # Extraer coordenadas de la grilla
    lats = sat_data.lat.values
    lons = sat_data.lon.values

    # Crear malla de puntos de la grilla
    grid_lons, grid_lats = np.meshgrid(lons, lats)
    grid_points = np.column_stack([grid_lons.ravel(), grid_lats.ravel()])

    # Puntos de las estaciones
    station_points = stations_df[['lon', 'lat']].values
    station_factors = stations_df['factor'].values

    if use_idw:
        # Interpolación IDW personalizada
        factor_grid_values = inverse_distance_weighting(
            station_points, station_factors, grid_points, 
            power=power, n_neighbors=n_neighbors, max_distance=max_distance
        )
    else:
        # Usar griddata de scipy
        factor_grid_values = griddata(
            station_points, station_factors, grid_points, 
            method=method, fill_value=np.nan
        )

    # Remodelar a la forma de la grilla
    factor_grid_values = factor_grid_values.reshape(grid_lats.shape)

    # Crear xarray DataArray
    factor_grid = xr.DataArray(
        factor_grid_values,
        dims=['lat', 'lon'],
        coords={'lat': lats, 'lon': lons},
        name='bias_factor',
        attrs={
            'description': f'Factores de corrección de sesgo interpolados ({method})',
            'n_stations': len(stations_df),
            'interpolation_method': 'IDW' if use_idw else method
        }
    )

    return factor_grid

def export_geotiff(DATA_input, lat, lon, dist, nameout):

    DATA = np.flipud(DATA_input)

    lon_res = lon[1] - lon[0]
    lat_res = lat[1] - lat[0]

    west  = lon[0] - lon_res / 2
    north = lat[-1] + lat_res / 2

    transform = from_origin(west, north, lon_res, lat_res)

    with rasterio.open(
        os.path.join('..','output','geotiff', dist, f"{nameout}.tif"),
        "w",
        driver="GTiff",
        height=DATA.shape[0],
        width=DATA.shape[1],
        count=1,
        dtype=DATA.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(DATA, 1)

def export_geotiff_utm(DATA_input, lat, lon, dist, nameout, utm_epsg="EPSG:32632"):
    """
    Export 2D array to GeoTIFF in UTM coordinates (QGIS compatible)

    Parameters
    ----------
    DATA_input : 2D np.ndarray
        Data array [lat, lon]
    lat, lon : 1D arrays
        Latitude and longitude coordinates (EPSG:4326)
    dist : str
        Subfolder name
    nameout : str
        Output filename (without .tif)
    utm_epsg : str
        UTM CRS (default: EPSG:32632, Italy) - UTM 32N
    """

    # ------------------------------------------------------------------
    # 1. Ensure north-up orientation
    # ------------------------------------------------------------------
    if lat[0] < lat[-1]:
        DATA = np.flipud(DATA_input)
        lat = lat[::-1]
    else:
        DATA = DATA_input.copy()

    # ------------------------------------------------------------------
    # 2. Project lon/lat → UTM
    # ------------------------------------------------------------------
    transformer = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)

    lon2d, lat2d = np.meshgrid(lon, lat)
    x2d, y2d = transformer.transform(lon2d, lat2d)

    # ------------------------------------------------------------------
    # 3. Compute pixel resolution
    # ------------------------------------------------------------------
    x_res = np.mean(np.diff(x2d, axis=1))
    y_res = np.mean(np.diff(y2d, axis=0))

    x_res = abs(x_res)
    y_res = abs(y_res)

    # ------------------------------------------------------------------
    # 4. Define top-left origin (QGIS standard)
    # ------------------------------------------------------------------
    west  = x2d[0, 0] - x_res / 2
    north = y2d[0, 0] + y_res / 2

    transform = from_origin(west, north, x_res, y_res)

    # ------------------------------------------------------------------
    # 5. Write GeoTIFF
    # ------------------------------------------------------------------
    out_path = os.path.join("..", "output", "geotiff", dist, f"{nameout}.tif")

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=DATA.shape[0],
        width=DATA.shape[1],
        count=1,
        dtype=DATA.dtype,
        crs=utm_epsg,
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(DATA, 1)

    print(f"✅ Exported GeoTIFF (UTM): {out_path}")

def calculate_mare(obs, mod, eps=1e-6):
    """
    Mean Absolute Relative Error (MARE)

    obs, mod : arrays (nt, ny, nx) o compatibles
    eps      : evita división por cero
    """
    re = (mod - obs) / (obs + eps)
    return np.nanmean(np.abs(re), axis=0)

def calculate_rmse(obs, mod):
    """
    Root Mean Squared Error (RMSE)

    obs, mod : arrays (nt, ny, nx) o compatibles
    """
    mse = (mod - obs) ** 2
    return np.sqrt(np.nanmean(mse, axis=0))

def calculate_nse(obs, mod, axis=0):
    """
    Nash-Sutcliffe Efficiency (NSE)

    obs, mod : arrays (nt, ny, nx) o similares
    axis     : eje temporal (default=0)

    Returns:
        NSE por pixel (ny, nx)
    """
    num = np.nansum((mod - obs) ** 2, axis=axis)
    den = np.nansum((obs - np.nanmean(obs, axis=axis)) ** 2, axis=axis)

    return 1 - (num / (den + 1e-12))

def calculate_kge(obs, mod, axis=0):
    """
    Kling-Gupta Efficiency (KGE)

    Descompone en:
    - correlación (r)
    - variabilidad (alpha = std_mod / std_obs)
    - sesgo (beta = mean_mod / mean_obs)

    Returns:
        KGE por pixel (ny, nx)
    """

    obs_mean = np.nanmean(obs, axis=axis)
    mod_mean = np.nanmean(mod, axis=axis)

    obs_std = np.nanstd(obs, axis=axis)
    mod_std = np.nanstd(mod, axis=axis)

    # Correlación (pixel-wise)
    obs_anom = obs - obs_mean
    mod_anom = mod - mod_mean

    cov = np.nanmean(obs_anom * mod_anom, axis=axis)
    r = cov / (obs_std * mod_std + 1e-12)

    # Componentes KGE
    alpha = mod_std / (obs_std + 1e-12)
    beta = mod_mean / (obs_mean + 1e-12)

    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge

def Statistics_RAW_DOWN(DF_IMERG, DF_CMORPH, DF_MSWEP, DF_ERA5, DF_GSMaP, DF_CHIRPS, DF_ENSEMBLE_MEDIAN):
    
    labels = ["IMERG", "CMORPH", "MSWEP", "ERA5", "GSMaP", "CHIRPS", "ENSEMBLE"]

    # ==================================================================================================================
    ## RAW METRICS FOR DAILY ANNUAL MAXIMA (QUANTILES)
    RAW_mare = np.array([
                    np.round(calculate_mare(DF_IMERG.Mevd_OBS, DF_IMERG.Mevd_SAT),3),
                    np.round(calculate_mare(DF_CMORPH.Mevd_OBS, DF_CMORPH.Mevd_SAT),3),
                    np.round(calculate_mare(DF_MSWEP.Mevd_OBS, DF_MSWEP.Mevd_SAT),3),
                    np.round(calculate_mare(DF_ERA5.Mevd_OBS, DF_ERA5.Mevd_SAT),3),
                    np.round(calculate_mare(DF_GSMaP.Mevd_OBS, DF_GSMaP.Mevd_SAT),3),
                    np.round(calculate_mare(DF_CHIRPS.Mevd_OBS, DF_CHIRPS.Mevd_SAT),3),
                    np.round(calculate_mare(DF_ENSEMBLE_MEDIAN.Mevd_OBS, DF_ENSEMBLE_MEDIAN.Mevd_SAT),3)
                    ])

    RAW_rmse = np.array([
                    np.round(calculate_rmse(DF_IMERG.Mevd_OBS, DF_IMERG.Mevd_SAT),3),
                    np.round(calculate_rmse(DF_CMORPH.Mevd_OBS, DF_CMORPH.Mevd_SAT),3),
                    np.round(calculate_rmse(DF_MSWEP.Mevd_OBS, DF_MSWEP.Mevd_SAT),3),
                    np.round(calculate_rmse(DF_ERA5.Mevd_OBS, DF_ERA5.Mevd_SAT),3),
                    np.round(calculate_rmse(DF_GSMaP.Mevd_OBS, DF_GSMaP.Mevd_SAT),3),
                    np.round(calculate_rmse(DF_CHIRPS.Mevd_OBS, DF_CHIRPS.Mevd_SAT),3),
                    np.round(calculate_rmse(DF_ENSEMBLE_MEDIAN.Mevd_OBS, DF_ENSEMBLE_MEDIAN.Mevd_SAT),3)
                    ])

    RAW_nse = np.array([
                    np.round(calculate_nse(DF_IMERG.Mevd_OBS, DF_IMERG.Mevd_SAT),3),
                    np.round(calculate_nse(DF_CMORPH.Mevd_OBS, DF_CMORPH.Mevd_SAT),3),
                    np.round(calculate_nse(DF_MSWEP.Mevd_OBS, DF_MSWEP.Mevd_SAT),3),
                    np.round(calculate_nse(DF_ERA5.Mevd_OBS, DF_ERA5.Mevd_SAT),3),
                    np.round(calculate_nse(DF_GSMaP.Mevd_OBS, DF_GSMaP.Mevd_SAT),3),
                    np.round(calculate_nse(DF_CHIRPS.Mevd_OBS, DF_CHIRPS.Mevd_SAT),3),
                    np.round(calculate_nse(DF_ENSEMBLE_MEDIAN.Mevd_OBS, DF_ENSEMBLE_MEDIAN.Mevd_SAT),3)
                    ])

    RAW_kge = np.array([
                    np.round(calculate_kge(DF_IMERG.Mevd_OBS, DF_IMERG.Mevd_SAT),3),
                    np.round(calculate_kge(DF_CMORPH.Mevd_OBS, DF_CMORPH.Mevd_SAT),3),
                    np.round(calculate_kge(DF_MSWEP.Mevd_OBS, DF_MSWEP.Mevd_SAT),3),
                    np.round(calculate_kge(DF_ERA5.Mevd_OBS, DF_ERA5.Mevd_SAT),3),
                    np.round(calculate_kge(DF_GSMaP.Mevd_OBS, DF_GSMaP.Mevd_SAT),3),
                    np.round(calculate_kge(DF_CHIRPS.Mevd_OBS, DF_CHIRPS.Mevd_SAT),3),
                    np.round(calculate_kge(DF_ENSEMBLE_MEDIAN.Mevd_OBS, DF_ENSEMBLE_MEDIAN.Mevd_SAT),3)
                    ])

    RAW_corrs = np.array([
        np.round(DF_IMERG.Mevd_OBS.corr(DF_IMERG.Mevd_SAT),3),
        np.round(DF_CMORPH.Mevd_OBS.corr(DF_CMORPH.Mevd_SAT),3),
        np.round(DF_MSWEP.Mevd_OBS.corr(DF_MSWEP.Mevd_SAT),3),
        np.round(DF_ERA5.Mevd_OBS.corr(DF_ERA5.Mevd_SAT),3),
        np.round(DF_GSMaP.Mevd_OBS.corr(DF_GSMaP.Mevd_SAT),3),
        np.round(DF_CHIRPS.Mevd_OBS.corr(DF_CHIRPS.Mevd_SAT),3),
        np.round(DF_ENSEMBLE_MEDIAN.Mevd_OBS.corr(DF_ENSEMBLE_MEDIAN.Mevd_SAT),3)
    ])

    ## RER METRICS FOR RELATIVE ERRORS ANALYSIS
    RAW_std = np.array([
                    np.round(np.std(DF_IMERG.RE_SAT),3),
                    np.round(np.std(DF_CMORPH.RE_SAT),3), 
                    np.round(np.std(DF_MSWEP.RE_SAT),3),
                    np.round(np.std(DF_ERA5.RE_SAT),3), 
                    np.round(np.std(DF_GSMaP.RE_SAT),3),
                    np.round(np.std(DF_CHIRPS.RE_SAT),3),
                    np.round(np.std(DF_ENSEMBLE_MEDIAN.RE_SAT),3)
                    ])

    RAW_mean = np.array([
        np.round(np.nanmean(DF_IMERG.RE_SAT),3),
        np.round(np.nanmean(DF_CMORPH.RE_SAT),3),
        np.round(np.nanmean(DF_MSWEP.RE_SAT),3),
        np.round(np.nanmean(DF_ERA5.RE_SAT),3),
        np.round(np.nanmean(DF_GSMaP.RE_SAT),3),
        np.round(np.nanmean(DF_CHIRPS.RE_SAT),3),
        np.round(np.nanmean(DF_ENSEMBLE_MEDIAN.RE_SAT),3)
    ])

    RAW_median = np.array([
        np.round(np.nanmedian(DF_IMERG.RE_SAT),3),
        np.round(np.nanmedian(DF_CMORPH.RE_SAT),3),
        np.round(np.nanmedian(DF_MSWEP.RE_SAT),3),
        np.round(np.nanmedian(DF_ERA5.RE_SAT),3),
        np.round(np.nanmedian(DF_GSMaP.RE_SAT),3),
        np.round(np.nanmedian(DF_CHIRPS.RE_SAT),3),
        np.round(np.nanmedian(DF_ENSEMBLE_MEDIAN.RE_SAT),3)
    ])

    RAW_diff = abs(RAW_mean - RAW_median)

    RAW_IQ = np.array([
        np.round(np.nanpercentile(DF_IMERG.RE_SAT, 75) - np.nanpercentile(DF_IMERG.RE_SAT, 25),3),
        np.round(np.nanpercentile(DF_CMORPH.RE_SAT, 75) - np.nanpercentile(DF_CMORPH.RE_SAT, 25),3),
        np.round(np.nanpercentile(DF_MSWEP.RE_SAT, 75) - np.nanpercentile(DF_MSWEP.RE_SAT, 25),3),
        np.round(np.nanpercentile(DF_ERA5.RE_SAT, 75) - np.nanpercentile(DF_ERA5.RE_SAT, 25),3),
        np.round(np.nanpercentile(DF_GSMaP.RE_SAT, 75) - np.nanpercentile(DF_GSMaP.RE_SAT, 25),3),
        np.round(np.nanpercentile(DF_CHIRPS.RE_SAT, 75) - np.nanpercentile(DF_CHIRPS.RE_SAT, 25),3),
        np.round(np.nanpercentile(DF_ENSEMBLE_MEDIAN.RE_SAT, 75) - np.nanpercentile(DF_ENSEMBLE_MEDIAN.RE_SAT, 25),3)
    ])

    RSR_RAW_compare = pd.DataFrame({
        "Dataset": labels,
        "STD": RAW_std,
        "Mean": RAW_mean,
        "Median": RAW_median,
        "DIFF":RAW_diff,
        "IQR": RAW_IQ,
        "CORR": RAW_corrs,
        "MARE": RAW_mare,
        "RMSE": RAW_rmse,
        "NSE": RAW_nse,
        "KGE": RAW_kge,
    })

    # ==================================================================================================================
    ## DOWNSCALED METRICS FOR DAILY ANNUAL MAXIMA (QUANTILES)
    DOWN_mare = np.array([
                    np.round(calculate_mare(DF_IMERG.Mevd_OBS, DF_IMERG.Mevd_DOWN),3),
                    np.round(calculate_mare(DF_CMORPH.Mevd_OBS, DF_CMORPH.Mevd_DOWN),3),
                    np.round(calculate_mare(DF_MSWEP.Mevd_OBS, DF_MSWEP.Mevd_DOWN),3),
                    np.round(calculate_mare(DF_ERA5.Mevd_OBS, DF_ERA5.Mevd_DOWN),3),
                    np.round(calculate_mare(DF_GSMaP.Mevd_OBS, DF_GSMaP.Mevd_DOWN),3),
                    np.round(calculate_mare(DF_CHIRPS.Mevd_OBS, DF_CHIRPS.Mevd_DOWN),3),
                    np.round(calculate_mare(DF_ENSEMBLE_MEDIAN.Mevd_OBS, DF_ENSEMBLE_MEDIAN.Mevd_DOWN),3)
                    ])

    DOWN_rmse = np.array([
                    np.round(calculate_rmse(DF_IMERG.Mevd_OBS, DF_IMERG.Mevd_DOWN),3),
                    np.round(calculate_rmse(DF_CMORPH.Mevd_OBS, DF_CMORPH.Mevd_DOWN),3),
                    np.round(calculate_rmse(DF_MSWEP.Mevd_OBS, DF_MSWEP.Mevd_DOWN),3),
                    np.round(calculate_rmse(DF_ERA5.Mevd_OBS, DF_ERA5.Mevd_DOWN),3),
                    np.round(calculate_rmse(DF_GSMaP.Mevd_OBS, DF_GSMaP.Mevd_DOWN),3),
                    np.round(calculate_rmse(DF_CHIRPS.Mevd_OBS, DF_CHIRPS.Mevd_DOWN),3),
                    np.round(calculate_rmse(DF_ENSEMBLE_MEDIAN.Mevd_OBS, DF_ENSEMBLE_MEDIAN.Mevd_DOWN),3)
                    ])

    DOWN_corrs = np.array([
        np.round(DF_IMERG.Mevd_OBS.corr(DF_IMERG.Mevd_DOWN),3),
        np.round(DF_CMORPH.Mevd_OBS.corr(DF_CMORPH.Mevd_DOWN),3),
        np.round(DF_MSWEP.Mevd_OBS.corr(DF_MSWEP.Mevd_DOWN),3),
        np.round(DF_ERA5.Mevd_OBS.corr(DF_ERA5.Mevd_DOWN),3),
        np.round(DF_GSMaP.Mevd_OBS.corr(DF_GSMaP.Mevd_DOWN),3),
        np.round(DF_CHIRPS.Mevd_OBS.corr(DF_CHIRPS.Mevd_DOWN),3),
        np.round(DF_ENSEMBLE_MEDIAN.Mevd_OBS.corr(DF_ENSEMBLE_MEDIAN.Mevd_DOWN),3)
    ])
    
    DOWN_nse = np.array([
                np.round(calculate_nse(DF_IMERG.Mevd_OBS, DF_IMERG.Mevd_DOWN),3),
                np.round(calculate_nse(DF_CMORPH.Mevd_OBS, DF_CMORPH.Mevd_DOWN),3),
                np.round(calculate_nse(DF_MSWEP.Mevd_OBS, DF_MSWEP.Mevd_DOWN),3),
                np.round(calculate_nse(DF_ERA5.Mevd_OBS, DF_ERA5.Mevd_DOWN),3),
                np.round(calculate_nse(DF_GSMaP.Mevd_OBS, DF_GSMaP.Mevd_DOWN),3),
                np.round(calculate_nse(DF_CHIRPS.Mevd_OBS, DF_CHIRPS.Mevd_DOWN),3),
                np.round(calculate_nse(DF_ENSEMBLE_MEDIAN.Mevd_OBS, DF_ENSEMBLE_MEDIAN.Mevd_DOWN),3)
                ])

    DOWN_kge = np.array([
                    np.round(calculate_kge(DF_IMERG.Mevd_OBS, DF_IMERG.Mevd_DOWN),3),
                    np.round(calculate_kge(DF_CMORPH.Mevd_OBS, DF_CMORPH.Mevd_DOWN),3),
                    np.round(calculate_kge(DF_MSWEP.Mevd_OBS, DF_MSWEP.Mevd_DOWN),3),
                    np.round(calculate_kge(DF_ERA5.Mevd_OBS, DF_ERA5.Mevd_DOWN),3),
                    np.round(calculate_kge(DF_GSMaP.Mevd_OBS, DF_GSMaP.Mevd_DOWN),3),
                    np.round(calculate_kge(DF_CHIRPS.Mevd_OBS, DF_CHIRPS.Mevd_DOWN),3),
                    np.round(calculate_kge(DF_ENSEMBLE_MEDIAN.Mevd_OBS, DF_ENSEMBLE_MEDIAN.Mevd_DOWN),3)
                    ])

    ## DOWNSCALED METRICS FOR RELATIVE ERRORS ANALYSIS
    DOWN_std = np.array([
                    np.round(np.std(DF_IMERG.RE_DOWN),3),
                    np.round(np.std(DF_CMORPH.RE_DOWN),3), 
                    np.round(np.std(DF_MSWEP.RE_DOWN),3),
                    np.round(np.std(DF_ERA5.RE_DOWN),3), 
                    np.round(np.std(DF_GSMaP.RE_DOWN),3),
                    np.round(np.std(DF_CHIRPS.RE_DOWN),3),
                    np.round(np.std(DF_ENSEMBLE_MEDIAN.RE_DOWN),3)
                    ])

    DOWN_mean = np.array([
        np.round(np.nanmean(DF_IMERG.RE_DOWN),3),
        np.round(np.nanmean(DF_CMORPH.RE_DOWN),3),
        np.round(np.nanmean(DF_MSWEP.RE_DOWN),3),
        np.round(np.nanmean(DF_ERA5.RE_DOWN),3),
        np.round(np.nanmean(DF_GSMaP.RE_DOWN),3),
        np.round(np.nanmean(DF_CHIRPS.RE_DOWN),3),
        np.round(np.nanmean(DF_ENSEMBLE_MEDIAN.RE_DOWN),3)
    ])

    DOWN_median = np.array([
        np.round(np.nanmedian(DF_IMERG.RE_DOWN),3),
        np.round(np.nanmedian(DF_CMORPH.RE_DOWN),3),
        np.round(np.nanmedian(DF_MSWEP.RE_DOWN),3),
        np.round(np.nanmedian(DF_ERA5.RE_DOWN),3),
        np.round(np.nanmedian(DF_GSMaP.RE_DOWN),3),
        np.round(np.nanmedian(DF_CHIRPS.RE_DOWN),3),
        np.round(np.nanmedian(DF_ENSEMBLE_MEDIAN.RE_DOWN),3)
    ])

    DOWN_diff = abs(DOWN_mean - DOWN_median)

    DOWN_IQ = np.array([
        np.round(np.nanpercentile(DF_IMERG.RE_DOWN, 75) - np.nanpercentile(DF_IMERG.RE_DOWN, 25),3),
        np.round(np.nanpercentile(DF_CMORPH.RE_DOWN, 75) - np.nanpercentile(DF_CMORPH.RE_DOWN, 25),3),
        np.round(np.nanpercentile(DF_MSWEP.RE_DOWN, 75) - np.nanpercentile(DF_MSWEP.RE_DOWN, 25),3),
        np.round(np.nanpercentile(DF_ERA5.RE_DOWN, 75) - np.nanpercentile(DF_ERA5.RE_DOWN, 25),3),
        np.round(np.nanpercentile(DF_GSMaP.RE_DOWN, 75) - np.nanpercentile(DF_GSMaP.RE_DOWN, 25),3),
        np.round(np.nanpercentile(DF_CHIRPS.RE_DOWN, 75) - np.nanpercentile(DF_CHIRPS.RE_DOWN, 25),3),
        np.round(np.nanpercentile(DF_ENSEMBLE_MEDIAN.RE_DOWN, 75) - np.nanpercentile(DF_ENSEMBLE_MEDIAN.RE_DOWN, 25),3)
    ])

    RSR_DOWN_compare = pd.DataFrame({
        "Dataset": labels,
        "STD": DOWN_std,
        "Mean": DOWN_mean,
        "Median": DOWN_median,
        "DIFF":DOWN_diff,
        "IQR": DOWN_IQ,
        "CORR": DOWN_corrs,
        "MARE": DOWN_mare,
        "RMSE": DOWN_rmse,
        "NSE": DOWN_nse,
        "KGE": DOWN_kge,
    })
    
    return RSR_RAW_compare, RSR_DOWN_compare