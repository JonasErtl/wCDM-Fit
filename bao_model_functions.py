import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import quad
from scipy.optimize import minimize
import emcee
from tqdm import tqdm

# desi daten laden
MEAN_PATH = Path("desi_gaussian_bao_ALL_GCcomb_mean.txt")
COV_PATH = Path("desi_gaussian_bao_ALL_GCcomb_cov.txt")

def load_desi_all_gccomb(mean_path=MEAN_PATH, cov_path=COV_PATH):
    mean_df = pd.read_csv(mean_path, comment="#", sep=r"\s+", header=None, names=["z", "value", "quantity"])
    cov = np.loadtxt(cov_path)
    return mean_df, cov

# kosmologische funktionen
C_KM_S = 299_792.458

def E_z(z, Om, Or=0.0, w=-1.0):
    Ode = 1.0 - Om - Or
    Ez = Om*(1+z)**3 + Or*(1+z)**4 + Ode*(1+z)**(3*(1+w))
    return np.sqrt(np.maximum(Ez, 1e-12))  # Vermeidet negatives sqrt

def H_of_z(z, h, Om, Or=0.0, w=-1.0):
    return 100.0 * h * E_z(z, Om=Om, Or=Or, w=w)

def D_H(z, h, Om, Or=0.0, w=-1.0):
    return C_KM_S / H_of_z(z, h=h, Om=Om, Or=Or, w=w)

def D_M(z, h, Om, Or=0.0, w=-1.0):
    integrand = lambda zp: C_KM_S / H_of_z(zp, h=h, Om=Om, Or=Or, w=w)
    val, _ = quad(integrand, 0.0, float(z), epsabs=1e-7, epsrel=1e-7)
    return val

def D_V(z, h, Om, Or=0.0, w=-1.0):
    DH = D_H(z, h=h, Om=Om, Or=Or, w=w)
    DM = D_M(z, h=h, Om=Om, Or=Or, w=w)
    return np.cbrt(np.maximum(z * DH * DM**2, 1e-12))

def model_vector(zs, quantities, h, Om, rd, w=-1.0):
    return np.array([
        D_M(z, h, Om, w=w)/rd if q == "DM_over_rs" else
        D_H(z, h, Om, w=w)/rd if q == "DH_over_rs" else
        D_V(z, h, Om, w=w)/rd if q == "DV_over_rs" else np.nan
        for z, q in zip(zs, quantities)
    ])

# chi^2 funktion
def chi2(params, z, quantity, data, cov, model="LCDM"):
    if model == "LCDM":
        Om, h, rd = params
        w = -1.0
    else:
        Om, h, rd, w = params
    model_vals = model_vector(z, quantity, h=h, Om=Om, rd=rd, w=w)
    diff = data - model_vals
    return diff @ np.linalg.inv(cov) @ diff

# unsicherheiten mit monte carlo markov chain
def log_prior(params, model):
    if model == "LCDM":
        Om, h, rd = params
        if 0.1 < Om < 0.5 and 0.5 < h < 0.9 and 130 < rd < 160:
            return 0.0
    elif model == "wCDM":
        Om, h, rd, w = params
        if 0.1 < Om < 0.5 and 0.5 < h < 0.9 and 130 < rd < 160 and -1.5 < w < -0.5:
            return 0.0
    return -np.inf

def log_likelihood(params, z, q, d, cov, model):
    return -0.5 * chi2(params, z, q, d, cov, model)

def log_posterior(params, z, q, d, cov, model):
    lp = log_prior(params, model)
    return lp + log_likelihood(params, z, q, d, cov, model) if np.isfinite(lp) else -np.inf

def run_mcmc(result, z, q, d, cov, model, param_names):
    ndim = len(result.x)
    nwalkers = 50
    p0 = result.x + 1e-4 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(z, q, d, cov, model))
    sampler.run_mcmc(p0, 5000, progress=True)
    samples = sampler.get_chain(discard=1000, flat=True)
    means = np.mean(samples, axis=0)
    stds = np.std(samples, axis=0)
    return means, stds

# ausfuehrung 
mean_df, cov = load_desi_all_gccomb()
z = mean_df["z"].values
q = mean_df["quantity"].values
d = mean_df["value"].values

# Fit
res_LCDM = minimize(chi2, [0.3, 0.7, 147], args=(z, q, d, cov, "LCDM"), bounds=[(0.1,0.5),(0.5,0.9),(130,160)])
res_wCDM = minimize(chi2, [0.3, 0.7, 147, -1.0], args=(z, q, d, cov, "wCDM"), bounds=[(0.1,0.5),(0.5,0.9),(130,160),(-1.5,-0.5)])

# MCMC
params_LCDM, errors_LCDM = run_mcmc(res_LCDM, z, q, d, cov, "LCDM", ["Ωm", "h", "rs"])
params_wCDM, errors_wCDM = run_mcmc(res_wCDM, z, q, d, cov, "wCDM", ["Ωm", "h", "rs", "w"])

# Ausgabe
print("\nΛCDM Parameter mit 1σ-Unsicherheiten:")
for name, val, err in zip(["Ωm", "h", "rs"], params_LCDM, errors_LCDM):
    print(f"{name} = {val:.4f} ± {err:.4f}")

print("\nwCDM Parameter mit 1σ-Unsicherheiten:")
for name, val, err in zip(["Ωm", "h", "rs", "w"], params_wCDM, errors_wCDM):
    print(f"{name} = {val:.4f} ± {err:.4f}")

# AIC
aic_LCDM = res_LCDM.fun + 2 * 3
aic_wCDM = res_wCDM.fun + 2 * 4
print(f"\nAIC (ΛCDM): {aic_LCDM:.2f}")
print(f"AIC (wCDM): {aic_wCDM:.2f}")
print(f"ΔAIC = {aic_LCDM - aic_wCDM:.2f}")

