# Machine learning algorithm for survival analysis

## LTRC trees and LTRC Random Forest

**LTRC - trees** Based on partykit and Rpart algorithm (conditional inference) 


**Splitting method -- log rank test**

Denote $(D_i, R_i, \delta_i)$ respectively the survival/censored time, the left truncation time and the event indicator, then for each 
instance $i$ its contribution to the residual deviance
$d_i = 2 \left[\delta_i \log \left( \dfrac{\delta_i}{(\hat{\Lambda}_0(R_i) - \hat{\Lambda}_0(L_i)) \hat{\theta}} \right) - \left(\delta_i - (\hat{\Lambda}_0(R_i) - \hat{\Lambda}_0(L_i))\hat{\theta} \right) \right]$
with $\hat{\theta} = \sum \delta_i / \sum (\Lambda_0(R_i) - \Lambda_0(L_i))$

**Extension to LTRC - forest**


- build m independent LTRCART using bootstrap procedure
- Random feature selection layer
- Compute the average estimation $\hat{s}(t, \textbf{x}) = 1/n\sum_{j \leqslant n} \tau_j(t, \textbf{x})$



## Benchmark

![benchmark](public/benchmark.png)

## Install notice

To install the package you can run


    python -m pip install git+https://gitlab.eurobios.com/vlaurent/survival-trees



## References

* https://academic.oup.com/biostatistics/article/18/2/352/2739324

## Requirements

Having `R` compiler installed

## Project

This implementation come from an SNCF DTIPG project and is developped and maintained by Eurobios Scientific Computation Branch and SNCF IR

<img src="https://www.sncf.com/themes/contrib/sncf_theme/images/logo-sncf.svg?v=3102549095" alt="drawing" width="100"/>

<img src="https://www.mews-partners.com/wp-content/uploads/2021/09/Eurobios-Mews-Labs-logo-768x274.png.webp" alt="drawing" width="175"/>


## Authors

- Vincent LAURENT : vlaurent@eurobios.com
