import os
import rasterio
import numpy as np
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