# GEV and MEVD Modeling of Extreme Precipitation with Statistical Downscaling Applied to MEVD

This repository contains Python codes for modelling extreme precipitation using the Generalized Extreme Value (GEV) distribution and the Metastatistical Extreme Value Distribution (MEVD) [[1]](#ref1), [[2]](#ref2). 

## Spatial Downscaling Approach

The repository also includes the application of a spatial statistical downscaling method [[3]](#ref3). This approach links precipitation statistics at the satellite grid scale with those at the point scale by exploiting the scaling properties of rainfall fields. In particular, the parameters of the Weibull distribution describing rainfall intensities are translated from the satellite resolution to the point scale through two key functions: the intermittency function (β), which accounts for the scaling of the wet fraction, and the variance reduction function (γ), which describes how rainfall variance changes across spatial scales. These parameters are estimated directly from the satellite datasets using high-temporal-resolution rainfall information and the spatial correlation structure of the rainfall field. The resulting point-scale Weibull parameters are then used within the MEVD framework to derive extreme precipitation statistics at the point scale from satellite observations.

The GIF below illustrates a conceptual representation of the procedure applied over the Veneto region to estimate the two key parameters involved in the spatial downscaling process: β (intermittency function) and γ (variance reduction function). The animation highlights how the methodology is applied independently to each pixel within the study area in order to derive the spatial distribution of these parameters.

<div align="center">
  <img src="https://raw.githubusercontent.com/TeamExtremes/MEVD-downscaling/refs/heads/main/figures/gif/Veneto_box_beta_gamma_v3.gif" alt="Seasonal_Climatology" />
</div>

## MEV Example

The figure below illustrates the results obtained using the Metastatistical Extreme Value Distribution (MEVD) for six different precipitation products, considering a 50-year return period. Each panel represents the estimated extreme precipitation for the corresponding product over the study area.

<div align="center">
  <img src="https://raw.githubusercontent.com/TeamExtremes/MEVD-downscaling/refs/heads/main/figures/ALL/Quantiles_ALL_MEV_raw_50yrs.png" alt="Seasonal_Climatology" />
</div>

## Bias Correction

The repository also implements a bias correction procedure to adjust satellite-derived precipitation statistics using rain gauge observations. The correction is applied at the pixel level and is based on the ratio between rain gauge measurements and satellite estimates.

First, a correction factor (OBS/SAT) is computed at each rain gauge location for the parameter of interest (e.g., number of wet days, scale, or shape parameters). These station-based correction factors are then spatially interpolated onto the grid of the satellite rainfall product (e.g., GSMaP), producing a continuous correction map over the entire study domain. Finally, the satellite-derived parameters are adjusted using this interpolated correction field.

For example, the correction of the number of wet days is obtained by computing the OBS/SAT ratio at rain gauge locations and interpolating this information to the satellite grid. The resulting field is then used to derive the corrected value of the parameter across all pixels. The same procedure is applied to correct the scale and shape parameters of the rainfall distribution.

<div align="center">
  <img src="https://raw.githubusercontent.com/TeamExtremes/MEVD-downscaling/refs/heads/main/figures/corrected/Quantiles_IMERG_GEV_raw_200yrs.png" alt="Seasonal_Climatology" />
</div>

## Zenodo Repository

The results obtained for the satellite and reanalysis products, including outputs from GEV, MEVD, and MEVD-downscaling, are openly available and can be accessed from the following Zenodo repository: [https://zenodo.org/records/18885925](https://zenodo.org/records/18885925).

## References

<a id="ref1"></a>[1] Marani M, Ignaccolo M (2015). *A Metastatistical Approach to Rainfall Extremes*. Advances in Water Resources, 79: 121–126. https://doi.org/10.1016/j.advwatres.2015.03.001

<a id="ref2"></a>[2] Marra F, Borga M, Morin E (2020). *A Uniﬁed Framework for Extreme Subdaily Precipitation Frequency Analyses Based on Ordinary Events*. Geophysical Research Letters 47: 1-8. https://doi.org/10.1029/2020GL090209

<a id="ref3"></a>[3] Zorzetto E, Marani M (2018) Downscaling of Rainfall Extremes From Satellite Observations. Water Resources Research 55(1): 156-174. http://doi.org/10.1029/2018WR022950
