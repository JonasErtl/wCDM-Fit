# wCDM-Fit

A concise Python project comparing the **ΛCDM** and **wCDM** cosmological models  
using the latest **DESI BAO data**.

Both models are fitted using **χ² minimization** and **MCMC sampling**,  
and evaluated with the **Akaike Information Criterion (AIC)**  
to test for possible deviations from \(w = -1\).

**Results:**  
| Model  | Ωₘ | h | rₛ [Mpc] | w | AIC |
|---------|----|---|-----------|---|-----|
| ΛCDM | 0.2979 ± 0.0086 | 0.7044 ± 0.0425 | 144.62 ± 8.67 | – | **16.27** |
| wCDM | 0.2969 ± 0.0089 | 0.6937 ± 0.0431 | 144.49 ± 8.59 | −0.917 ± 0.079 | **17.04** |

**ΔAIC = −0.77 → ΛCDM slightly preferred.**

The MCMC analysis confirms that both models are statistically consistent,  
with no significant deviation from a constant \(w = -1\).  
DESI BAO data alone do not require a dynamical dark energy component.

---

*Author: Jonas Ertl*  
*GitHub Repository:* [github.com/JonasErtl/wCDM-Fit](https://github.com/JonasErtl/wCDM-Fit)
