import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import quad
from scipy.optimize import minimize

#lade desi roh daten 

MEAN_PATH = Path("desi_gaussian_bao_ALL_GCcomb_mean.txt")
COV_PATH  = Path("desi_gaussian_bao_ALL_GCcomb_cov.txt")

def load_desi_all_gccomb(mean_path=MEAN_PATH, cov_path=COV_PATH):
    mean_df = pd.read_csv(
        mean_path, comment="#", sep=r"\s+",
        header=None, names=["z", "value", "quantity"]
    )
    cov = np.loadtxt(cov_path)
    mean_df["sigma_1sigma"] = np.sqrt(np.diag(cov))
    return mean_df, cov

#definition der kosmologischen funktioinen

C_KM_S = 299_792.458  # speed of light [km/s]

def E_z(z, Om=0.3, Or=0.0, w=-1.0):
    Ode = 1.0 - Om - Or
    return np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ode*(1+z)**(3*(1+w)))

def H_of_z(z, h=0.7, Om=0.3, Or=0.0, w=-1.0):
    H0 = 100.0 * h
    return H0 * E_z(z, Om=Om, Or=Or, w=w)

def D_H(z, h=0.7, Om=0.3, Or=0.0, w=-1.0):
    return C_KM_S / H_of_z(z, h=h, Om=Om, Or=Or, w=w)

def D_M(z, h=0.7, Om=0.3, Or=0.0, w=-1.0):
    integrand = lambda zp: C_KM_S / H_of_z(zp, h=h, Om=Om, Or=Or, w=w)
    val, _ = quad(integrand, 0.0, float(z), epsabs=1e-7, epsrel=1e-7, limit=200)
    return val

def D_V(z, h=0.7, Om=0.3, Or=0.0, w=-1.0):
    DH = D_H(z, h=h, Om=Om, Or=Or, w=w)
    DM = D_M(z, h=h, Om=Om, Or=Or, w=w)
    return (z * DH * DM**2) ** (1.0/3.0)

def model_vector(zs, quantities, h=0.7, Om=0.3, Or=0.0, w=-1.0, rd=147.1):
    out = []
    for z, q in zip(zs, quantities):
        if q == "DM_over_rs":
            out.append(D_M(z, h=h, Om=Om, Or=Or, w=w) / rd)
        elif q == "DH_over_rs":
            out.append(D_H(z, h=h, Om=Om, Or=Or, w=w) / rd)
        elif q == "DV_over_rs":
            out.append(D_V(z, h=h, Om=Om, Or=Or, w=w) / rd)
        else:
            raise ValueError(f"Unknown quantity: {q}")
    return np.array(out)

#x2 funktion

def chi2(params, z, quantity, data, cov, model="LCDM"):
    if model == "LCDM":
        Om, h, rd = params
        w = -1.0
    elif model == "wCDM":
        Om, h, rd, w = params
    else:
        raise ValueError("Model must be 'LCDM' or 'wCDM'")
    m = model_vector(z, quantity, Om=Om, h=h, rd=rd, w=w)
    diff = data - m
    invcov = np.linalg.inv(cov)
    return diff @ invcov @ diff

#fitting wird durchgefuehrt

if __name__ == "__main__":
    mean_df, cov = load_desi_all_gccomb()
    z = mean_df["z"].to_numpy()
    q = mean_df["quantity"].to_numpy()
    d = mean_df["value"].to_numpy()

    #Fit fuer ΛCDM (3 parameter)
    x0_LCDM = [0.3, 0.7, 147.0]  # guess: [Om, h, rd]
    res_LCDM = minimize(chi2, x0_LCDM, args=(z, q, d, cov, "LCDM"),
                        bounds=[(0.1,0.5),(0.5,0.9),(130,160)])
    
    #Fit fuer wCDM (4 parameter)
    x0_wCDM = [0.3, 0.7, 147.0, -1.0]  # guess: [Om, h, rd, w]
    res_wCDM = minimize(chi2, x0_wCDM, args=(z, q, d, cov, "wCDM"),
                        bounds=[(0.1,0.5),(0.5,0.9),(130,160),(-1.5,-0.5)])

    print("\n=== DESI BAO ALL_GCcomb Fits ===")
    print("\nΛCDM best-fit:")
    print("Params [Ωm, h, rd] =", res_LCDM.x)
    print("χ² =", res_LCDM.fun)

    print("\nwCDM best-fit:")
    print("Params [Ωm, h, rd, w] =", res_wCDM.x)
    print("χ² =", res_wCDM.fun)

    # Compare χ² values
    delta_chi2 = res_LCDM.fun - res_wCDM.fun
    print(f"\nΔχ² (ΛCDM - wCDM) = {delta_chi2:.3f}")
