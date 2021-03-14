# Ensemble forecast post-processing (fcstpp)

A collection of ensemble forecast post-processing functions.

| `fcstpp.gridpp`| Usage | Reference |
|:------------------|:--------------------------|:----------------|
| `schaake_shuffle` | Ensemble member shuffling based on given trajectories. | [Clark et al. (2004)](https://journals.ametsoc.org/view/journals/hydr/5/1/1525-7541_2004_005_0243_tssamf_2_0_co_2.xml) |
| `quantile_mapping_stencil` | Quantile mapping with stencil grid points. | [Hamill et al. (2017)](https://journals.ametsoc.org/view/journals/mwre/145/9/mwr-d-16-0331.1.xml) |
| `bootstrap_fill` | Create "pseudo" ensembles from bootstrapping. | |

| `fcstpp.metrics`| Usage | Reference |
|:------------------|:--------------------------|:----------------|
| `CRPS_1d` | CRPS and its two-term decomposition for 1-d ensembles. | [Grimit et al. (2006)](https://doi.org/10.1256/qj.05.235) |
| `CRPS_1d_nan`| Similar to `CRPS_1d` but ignores `np.nan` elements. | [Grimit et al. (2006)](https://doi.org/10.1256/qj.05.235) |
| `CRPS_2d` | CRPS and its two-term decompositions for 2-d ensembles. | [Grimit et al. (2006)](https://doi.org/10.1256/qj.05.235) |
| `CRPS_1d_from_quantiles` | Compute CRPS from quantile values. | |
| `BS_binary_1d` | Brier Score for 1-d ensembles. | [Hamill and Juras (2006)](https://doi.org/10.1256/qj.06.25) |


| `fcstpp.utils`| Usage | Reference |
|:------------------|:--------------------------|:----------------|
| `score_bootstrap_1d`| Bootstrap the last dimension of an array. | |
| `window_slider_cycled_1d` | Given an index, search its cyclied indices within a sliding window. | |
| `climate_subdaily_prob`| Compute climatological probabilities from a time window. | |
| `facet`| Compute terrain facet from gridded elevation. | [Daly et al. (2002)](https://www.int-res.com/abstracts/cr/v22/n2/p99-113) |


# Contact

Yingkai (Kyle) Sha <<yingkai@eoas.ubc.ca>> <<yingkaisha@gmail.com>>

# License

[MIT License](https://github.com/yingkaisha/fcstpp/blob/main/LICENSE)
