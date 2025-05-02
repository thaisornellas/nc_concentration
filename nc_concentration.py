from numcosmo_py import Nc, Ncm
import numpy as np
import math
from scipy.optimize import brentq
import sys

Ncm.cfg_init()
Ncm.cfg_set_log_handler(lambda msg: sys.stdout.write(msg) and sys.stdout.flush())

def sigmafunc(M, z, cosmo):
    dist = Nc.Distance.new(15.0)
    dist.prepare(cosmo)
    tf = Nc.TransferFuncEH.new()
    ps_lin = Nc.PowspecMLTransfer.new(tf)
    ps_lin.prepare(cosmo)
    psf = Ncm.PowspecFilter.new(ps_lin, Ncm.PowspecFilterType.TOPHAT)
    psf.set_best_lnr0()
    multf = Nc.MultiplicityFuncWatson.new()
    hmfunc = Nc.HaloMassFunction.new (dist, psf, multf)
    hmfunc.prepare(cosmo) 

    s = np.array([])

    if isinstance(M, np.ndarray):
        for i in M:
            m_r = hmfunc.lnM_to_lnR(cosmo, math.log(i))
            R = math.exp(m_r)
            sigma = psf.eval_sigma(z, R)
            s = np.append(s, sigma)

    else:
            m_r = hmfunc.lnM_to_lnR(cosmo, math.log(M))
            R = math.exp(m_r)
            s = psf.eval_sigma(z, R)
            
    return s

def _dln_sigma_dlnR(M, z, k, cosmo, delta_R=1e-5):

    dist = Nc.Distance.new(15.0)
    dist.prepare(cosmo)
    tf = Nc.TransferFuncEH.new()
    ps_lin = Nc.PowspecMLTransfer.new(tf)
    ps_lin.prepare(cosmo)
    psf = Ncm.PowspecFilter.new(ps_lin, Ncm.PowspecFilterType.TOPHAT)
    psf.set_best_lnr0()
    multf = Nc.MultiplicityFuncWatson.new()
    hmfunc = Nc.HaloMassFunction.new (dist, psf, multf)
    hmfunc.prepare(cosmo)

    if isinstance(M, np.ndarray):
        R = np.array([])
        for i in M:
            m_r = hmfunc.lnM_to_lnR(cosmo, math.log(i))
            r = math.exp(m_r) * k
            R = np.append(R, r)

    else:
            m_r = hmfunc.lnM_to_lnR(cosmo, math.log(M))
            R = math.exp(m_r) * k

    if isinstance(R, np.ndarray):
        deriv = np.array([])
        i = 0
        while i < len(R):
            ln_sigma_plus = np.log(psf.eval_sigma(z, R[i] + delta_R))
            ln_sigma_minus = np.log(psf.eval_sigma(z, R[i] - delta_R))

            ln_R_plus = np.log(R[i] + delta_R)
            ln_R_minus = np.log(R[i] - delta_R)

            d = (ln_sigma_plus - ln_sigma_minus) / (ln_R_plus - ln_R_minus)
            deriv = np.append(deriv, d)
            i = i + 1

    else:
        ln_sigma_plus = np.log(psf.eval_sigma(z, R + delta_R))
        ln_sigma_minus = np.log(psf.eval_sigma(z, R - delta_R))

        ln_R_plus = np.log(R + delta_R)
        ln_R_minus = np.log(R - delta_R)

        deriv = (ln_sigma_plus - ln_sigma_minus) / (ln_R_plus - ln_R_minus)

    return deriv

def _dln_Pk(M, z, k, cosmo, delta_k=1e-5):

    dist = Nc.Distance.new(15.0)
    dist.prepare(cosmo)
    #tf = Nc.TransferFuncBBKS.new()
    tf = Nc.TransferFuncEH.new()
    ps_lin = Nc.PowspecMLTransfer.new(tf)
    ps_lin.prepare(cosmo)
    psf = Ncm.PowspecFilter.new(ps_lin, Ncm.PowspecFilterType.TOPHAT)
    psf.set_best_lnr0()
    multf = Nc.MultiplicityFuncWatson.new()
    hmfunc = Nc.HaloMassFunction.new (dist, psf, multf)
    hmfunc.prepare(cosmo)

    if isinstance(M, np.ndarray):
        R = np.array([])
        for i in M:
            m_r = hmfunc.lnM_to_lnR(cosmo, math.log(i))
            r = math.exp(m_r) * k
            R = np.append(R, r)

    else:
            m_r = hmfunc.lnM_to_lnR(cosmo, math.log(M))
            R = math.exp(m_r) * k

    k_R = 2.0 * np.pi / R
    if isinstance(k_R, np.ndarray):
        i = 0
        deriv = np.array([])
        while i < len(k_R):
            ln_kR_plus = np.log(Ncm.Powspec.eval(ps_lin, cosmo, z, k_R[i] + delta_k))
            ln_kR_minus = np.log(Ncm.Powspec.eval(ps_lin, cosmo, z, k_R[i] - delta_k))

            ln_k_plus = np.log(k_R[i] + delta_k)
            ln_k_minus = np.log(k_R[i] - delta_k)

            d = (ln_kR_plus - ln_kR_minus) / (ln_k_plus - ln_k_minus)
            deriv = np.append(deriv, d)
            i = i + 1
    else:
        ln_kR_plus = np.log(Ncm.Powspec.eval(ps_lin, cosmo, z, k_R + delta_k))
        ln_kR_minus = np.log(Ncm.Powspec.eval(ps_lin, cosmo, z, k_R - delta_k))

        ln_k_plus = np.log(k_R + delta_k)
        ln_k_minus = np.log(k_R - delta_k)

        deriv = (ln_kR_plus - ln_kR_minus) / (ln_k_plus - ln_k_minus)
    return deriv

def _find_M(sigma, z, cosmo):

    dist = Nc.Distance.new(15.0)
    dist.prepare(cosmo)
    tf = Nc.TransferFuncEH.new()
    ps_lin = Nc.PowspecMLTransfer.new(tf)
    ps_lin.prepare(cosmo)
    psf = Ncm.PowspecFilter.new(ps_lin, Ncm.PowspecFilterType.TOPHAT)
    psf.set_best_lnr0()
    multf = Nc.MultiplicityFuncWatson.new()
    hmfunc = Nc.HaloMassFunction.new (dist, psf, multf)
    hmfunc.prepare(cosmo) 

    def f(R):
        sig = psf.eval_sigma(z, R)
        return sig - sigma
    
    R = brentq(f, a=0.1, b=50.0)
    m_r = hmfunc.lnR_to_lnM(cosmo, math.log(R))
    Mstar = math.exp(m_r)

    return Mstar

def _find_z(M, F, cosmo):

    dist = Nc.Distance.new(15.0)
    dist.prepare(cosmo)
    tf = Nc.TransferFuncEH.new()
    ps_lin = Nc.PowspecMLTransfer.new(tf)
    ps_lin.prepare(cosmo)
    psf = Ncm.PowspecFilter.new(ps_lin, Ncm.PowspecFilterType.TOPHAT)
    psf.set_best_lnr0()
    multf = Nc.MultiplicityFuncWatson.new()
    hmfunc = Nc.HaloMassFunction.new (dist, psf, multf)
    hmfunc.prepare(cosmo) 
    sigma = 1.686
    M = F * M

    R = np.array([])
    if isinstance(M, np.ndarray):
        for i in M:
            r_m = hmfunc.lnM_to_lnR(cosmo, math.log(i))
            r = math.exp(r_m)
            R = np.append(R, r)

    else:
            r_m = hmfunc.lnM_to_lnR(cosmo, math.log(M))
            R = math.exp(r_m)
    
    if isinstance(R, np.ndarray):
        i = 0
        zc = np.array([])
        while i < len(R):
            def f(z):
                sig = psf.eval_sigma(z, R[i]) - sigma
                return sig
            z = brentq(f, a=-1.0, b=2.0)
            zc = np.append(zc, z)
            i = i + 1
    else:
        def f(z):
            return psf.eval_sigma(z, R) - sigma
        zc = brentq(f, a=-1.0, b=2.0)

    return zc

def bullock_maccio(M, z, mdef, cosmo):
    
    F = 0.01
    K = 3.85

    zc = _find_z(M, F, cosmo)
    H = cosmo.H(z)
    if isinstance(zc, np.ndarray):
        Hzc = np.array([])
        for i in zc:
            H_zc = cosmo.H(i)
            Hzc = np.append(Hzc, H_zc)
    else:
        Hzc = cosmo.H(zc)

    c = K * (Hzc / H) ** (2/3)

    return c

def duffy08(M, z, mdef, cosmo, sample = 'full', profile = 'nfw'):
	
    # mass range of (10**11 to 10**15) h**-1
    # redshift range of 0-2.
    # calibrated with WMAP5 cosmology

    if profile == 'nfw':
        if sample == 'full':
            if mdef == '200c':
                A = 5.71
                B = -0.084
                C = -0.47
            elif mdef == 'vir':
                A = 7.85
                B = -0.081
                C = -0.71
            elif mdef == '200m':
                A = 10.14
                B = -0.081
                C = -1.01
        elif sample == 'relaxed':
            if mdef == '200c':
                A = 6.71
                B = -0.091
                C = -0.44
            elif mdef == 'vir':
                A = 9.23
                B = -0.091
                C = -0.69
            elif mdef == '200m':
                A = 11.93
                B = -0.090
                C = -0.99
    elif profile == 'einasto':
        if sample == 'full':
            if mdef == '200c':
                A = 6.40
                B = -0.108
                C = -0.62
            elif mdef == 'vir':
                A = 8.82
                B = -0.106
                C = -0.87
            elif mdef == '200m':
                A = 11.39
                B = -0.107
                C = -1.16
        elif sample == 'relaxed':
            if mdef == '200c':
                A = 7.74
                B = -0.123
                C = -0.60
            elif mdef == 'vir':
                A = 10.77
                B = -0.124
                C = -0.87
            elif mdef == '200m':
                A = 13.96
                B = -0.119
                C = -1.17

    c = A * (M * cosmo.h() / 2E12 )**B * (1.0 + z)**C
	
    return c

def klypin11(M, z, mdef, cosmo):
    c = 9.60 * (M * cosmo.h() / 1E12) ** -0.075
    return c

def prada12(M, z, mdef, cosmo):

    def cmin(x):
        return 3.681 + (5.033 - 3.681) * (1/np.pi * np.arctan(6.948 * (x - 0.424)) + 0.5)
    
    def sigmin(x):
        # it's in sigma^{-1}
        # the parameters inside the equation are also ^{-1}
        return 1.047 + (1.646 - 1.047) * (1/np.pi * np.arctan(7.386 * (x - 0.526)) + 0.5)
    
    s = sigmafunc(M, z, cosmo)
    a = 1.0 / (1.0 + z)
    x = (cosmo.E2Omega_de(z) / cosmo.Omega_m0()) ** (1.0 / 3.0) * a
    B0 = cmin(x)/cmin(1.393)
    B1 = sigmin(x)/sigmin(1.393)

    sig = B1 * s
    C = 2.881 * ((sig / 1.257)**1.022 + 1.0) * np.exp(0.060 / sig ** 2.0)
    c = B0 * C

    return c

def bhattacharya13(M, z, mdef, cosmo):

    """
    mass range of (2x10**12 to 2x10**15)h**-1
    redshift range of 0-2.
    calibrated with WMAP7 cosmology
    omegam = 0.1296 (Omegam = 0.25)
    omegab = 0.0224 (Omegab = 0.043)
    ns = 0.97
    sigma8 = 0.8
    h = 0.72
    """

    #delta_c = 1.68647
    delta_c = 1.6857555359455918
    s = sigmafunc(M, z, cosmo)
    nu = delta_c/s 
    func = Nc.GrowthFunc.new()
    func.prepare(cosmo)
    D = Nc.GrowthFunc.eval(func, cosmo, z)
    
    if mdef == '200c':
        A = 5.9
        B = 0.54
        C = -0.35
    elif mdef == 'vir':
        A = 7.7
        B = 0.90
        C = -0.29
    elif mdef == '200m':
        A = 9.0
        B = 1.15
        C = -0.29

    c_fit = A * D**B * nu**C
	
    return c_fit

def dutton14(M, z, mdef, cosmo):
    """
    This power-law fit was calibrated for the ``planck13`` cosmology.
    redshift varies in a range of 0-5
    """

    if mdef == '200c':
        a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z ** 1.21)
        b = -0.101 + 0.026 * z
    elif mdef == 'vir':
        a = 0.537 + (1.025 - 0.537) * np.exp(-0.718 * z ** 1.08)
        b = -0.097 + 0.024 * z

    c = 10 ** (a + b * np.log10(M / 1E12 * cosmo.h()))

    return c

def diemer15(M, z, mdef, cosmo, statistic = 'median'):

    if statistic == 'median':
        kappa = 1.00
        phi_0 = 6.58
        phi_1 = 1.27
        eta_0 = 7.28
        eta_1 = 1.56
        alpha = 1.08
        beta  = 1.77

    elif statistic == 'mean':
        kappa = 1.00
        phi_0 = 6.66
        phi_1 = 1.37
        eta_0 = 5.41
        eta_1 = 1.06
        alpha = 1.22
        beta  = 1.22

    n = _dln_Pk(M, z, kappa, cosmo)
    cmin = phi_0 + phi_1 * n
    nmin = eta_0 + eta_1 * n

    s = sigmafunc(M, z, cosmo)
    #delta_c = 1.68647019984
    delta_c = 1.68647 
    nu = delta_c / s

    c = cmin * 0.5 * ((nmin/nu) ** alpha + (nu/nmin) ** beta)

    return c

def klypin16(M, z, mdef, cosmo):

    s = sigmafunc(M, z, cosmo)

    if mdef == '200c':
        z_vals = [0.0, 0.38, 0.50, 1.00, 1.44, 2.50, 2.89, 5.41]
        a0_vals = [0.40, 0.65, 0.82, 1.08, 1.23, 1.60, 1.68, 1.70]
        b0_vals = [0.278, 0.375, 0.411, 0.436, 0.426, 0.375, 0.360, 0.351]
        a0 = np.interp(z, z_vals, a0_vals)
        b0 = np.interp(z, z_vals, b0_vals)

    elif mdef == 'vir':
        z_vals = [0.0, 0.38, 0.50, 1.00, 1.44, 2.50, 5.50]
        a0_vals = [0.75, 0.90, 0.97, 1.12, 1.28, 1.52, 1.62]
        b0_vals = [0.567, 0.541, 0.529, 0.496, 0.474, 0.421, 0.393]
        a0 = np.interp(z, z_vals, a0_vals)
        b0 = np.interp(z, z_vals, b0_vals)

    c = b0 * (1.0 + 7.37 * (s/a0)**(3/4)) * (1.0 + 0.14 * (s/a0)**-2.0)

    return c

def klypin16_m(M, z, mdef, cosmo):

    dist = Nc.Distance.new(15.0)
    dist.prepare(cosmo)
    tf = Nc.TransferFuncEH.new()
    ps_lin = Nc.PowspecMLTransfer.new(tf)
    ps_lin.prepare(cosmo)
    psf = Ncm.PowspecFilter.new(ps_lin, Ncm.PowspecFilterType.TOPHAT)
    psf.set_best_lnr0()
    multf = Nc.MultiplicityFuncWatson.new()
    hmfunc = Nc.HaloMassFunction.new (dist, psf, multf)
    hmfunc.prepare(cosmo) 

    def find_cosmology(cosmo, valores, tolerancia=1e-4):
        for key, valor in valores.items():
            if isinstance(valor, tuple):
                param_value, param = valor
                if abs(getattr(cosmo, key)(param) - param_value) >= tolerancia:
                    return False
            else:
                if abs(getattr(cosmo, key)() - valor) >= tolerancia:
                    return False
        return True

    plank13 = {
    'H0': 67.77,
    'Omega_m0': 0.3071,
    'Omega_b0': 0.0483,
    'sigma8': (0.8288, psf)
    }

    bolshoi = {
    'H0': 70.00,
    'Omega_m0': 0.27,
    'Omega_b0': 0.0469,
    'sigma8': (0.8200, psf)
    }

    if find_cosmology(cosmo, bolshoi):
        if mdef == '200c':
            z_vals = [0.0, 0.50, 1.0, 1.44, 2.15, 2.50, 2.90, 4.10]
            c0_vals = [6.6, 5.25, 3.85, 3.0, 2.1, 1.8, 1.6, 1.4]
            gamma_vals = [0.110, 0.105, 0.103, 0.097, 0.095, 0.095, 0.095, 0.095]
            M0_vals = [2E6, 6E4, 800.0, 110.0, 13.0, 6.0, 3.0, 1.0]
        elif mdef == 'vir':
            z_vals = [0.0, 0.50, 1.0, 1.44, 2.15, 2.50, 2.90, 4.10]
            c0_vals = [9.0, 6.0, 4.3, 3.3, 2.3, 2.1, 1.85, 1.7]
            gamma_vals = [0.1, 0.1, 0.1, 0.1, 0.095, 0.095, 0.095, 0.095]
            M0_vals = [2E6, 7E3, 550.0, 90.0, 11.0, 6.0, 2.5, 1.0]
		
    elif find_cosmology(cosmo, plank13):
        if mdef == '200c':
            z_vals = [0.0, 0.35, 0.5, 1.0, 1.44, 2.15, 2.5, 2.9, 4.1, 5.4]
            c0_vals = [7.4, 6.25, 5.65, 4.3, 3.53, 2.7, 2.42, 2.2, 1.92, 1.65]
            gamma_vals = [0.120, 0.117, 0.115, 0.110, 0.095, 0.085, 0.08, 0.08, 0.08, 0.08]
            M0_vals = [5.5E5, 1E5, 2E4, 900.0, 300.0, 42.0, 17.0, 8.5, 2.0, 0.3]
        elif mdef == 'vir':
            z_vals = [0.0, 0.35, 0.5, 1.0, 1.44, 2.15, 2.5, 2.9, 4.1, 5.4]
            c0_vals = [9.75, 7.25, 6.5, 4.75, 3.8, 3.0, 2.65, 2.42, 2.1, 1.86]
            gamma_vals = [0.110, 0.107, 0.105, 0.1, 0.095, 0.085, 0.08, 0.08, 0.08, 0.08]
            M0_vals = [5E5, 2.2E4, 1E4, 1000.0, 210.0, 43.0, 18.0, 9.0, 1.9, 0.42]

    else:
        raise Exception('Invalid cosmology for this model.')

    c0 = np.interp(z, z_vals, c0_vals)
    gamma = np.interp(z, z_vals, gamma_vals)
    M0 = np.interp(z, z_vals, M0_vals)
    M0 = M0 * 1E12 / cosmo.h()

    c = c0 * (M * cosmo.h() / 1E12 ) ** -gamma * (1 + (M / M0)**0.4)

    return c
     
def child18(M, z, mdef, cosmo, sample = 'individual_all'):
    sigma = 1.68647
    mstar = _find_M(sigma, z, cosmo)

    if sample == 'individual_all':
        A = 3.44
        b = 430.49
        c0 = 3.19
        m = -0.10
    elif sample == 'individual_relaxed':
        A = 2.88
        b = 1644.53
        c0 = 3.54
        m = -0.09
    elif sample == 'stack_nfw':
        A = 4.61
        b = 638.65
        c0 = 3.59
        m = -0.07
    elif sample == 'stack_ein':
        A = 63.2
        b = 431.48
        c0 = 3.36
        m = -0.01
    else:
        raise Warning('Invalid sample')

    c = A * (((M / mstar) / b) ** m * (1 + (M / mstar) / b)** -m - 1) + c0
    return c

def diemer19(M, z, mdef, cosmo):

    kappa = 0.41
    a0 = 2.45
    a1 = 1.82
    b0 = 3.20
    b1 = 2.30
    c_alpha = 0.21

    dlnsigma = _dln_sigma_dlnR(M, z, kappa, cosmo)
    n_eff = -2 * dlnsigma -3

    def _g(x):
        return math.log(1 + x) - x / (1 + x)
    
    def _G(x, n_eff):
        return x / _g(x) ** ((5 + n_eff)/6)
        
    func = Nc.GrowthFunc.new()
    func.prepare(cosmo)
    dD = Nc.GrowthFunc.eval_deriv(func, cosmo, z)
    D = Nc.GrowthFunc.eval(func, cosmo, z)
    alpha_eff = -dD * (1.0 + z) / D

    s = sigmafunc(M, z, cosmo)
    nu = 1.68647 / s

    A_neff = a0 * (1 + a1 * (n_eff + 3))
    B_neff = b0 * (1 + b1 * (n_eff + 3))
    C_aeff = 1 - c_alpha * (1 - alpha_eff)

    y = (1 + nu **2 / B_neff) * A_neff / nu
    
    def _G_inv(y, n_eff):
        if isinstance(y, np.ndarray):
            i = 0
            x_rt = np.array([])
            while i < len(y):
                def f(x):
                    x_value = _G(x, n_eff) - y[i]
                    x_value = np.asarray(x_value)
                    return x_value[i]
                x = brentq(f, a=0.05, b=200)
                x_rt = np.append(x_rt, x)
                i = i + 1

            return x_rt
            
        else:
            def f(y):
                func = lambda x: _G(x, n_eff) - y
                x_root = brentq(func, a=0.05, b=200)
                return x_root
            
            x_roots = f(y)
            return x_roots
        
    c = C_aeff * _G_inv(y, n_eff)

    return c

def ishiyama21(M, z, mdef, cosmo, sample = 'all', fit = True):

    if not fit:
        if sample == 'all':
            if mdef == '200c':
                kappa = 1.10
                a0 = 2.30
                a1 = 1.64
                b0 = 1.72
                b1 = 3.60
                c_alpha = 0.32
            elif mdef == 'vir':
                kappa = 0.76
                a0 = 2.34
                a1 = 1.82
                b0 = 1.83
                b1 = 3.52
                c_alpha = -0.18

        elif sample == 'relaxed':
            if mdef == '200c':
                kappa = 1.79
                a0 = 2.15
                a1 = 2.06
                b0 = 0.88
                b1 = 9.24
                c_alpha = 0.51
            elif mdef == 'vir':
                kappa = 2.40
                a0 = 2.27
                a1 = 1.80
                b0 = 0.56
                b1 = 13.24
                c_alpha = 0.079
    
    elif fit:
        if sample == 'all':
            if mdef == '200c':
                kappa = 1.19
                a0 = 2.54
                a1 = 1.33
                b0 = 4.04
                b1 = 1.21
                c_alpha = 0.22
            elif mdef == 'vir':
                kappa = 1.64
                a0 = 2.67
                a1 = 1.23
                b0 = 3.92
                b1 = 1.30
                c_alpha = -0.19
            elif mdef == '500c':
                kappa = 1.83
                a0 = 1.95
                a1 = 1.17
                b0 = 3.57
                b1 = 0.91
                c_alpha = 0.26
        elif sample == 'relaxed':
            if mdef == '200c':
                kappa = 0.60
                a0 = 2.14
                a1 = 2.63
                b0 = 1.69
                b1 = 6.36
                c_alpha = 0.37
            elif mdef == 'vir':
                kappa = 1.22
                a0 = 2.52
                a1 = 1.87
                b0 = 2.13
                b1 = 4.19
                c_alpha = -0.017
            elif mdef == '500c':
                kappa = 0.38
                a0 = 1.44
                a1 = 3.41
                b0 = 2.86
                b1 = 2.99
                c_alpha = 0.42 
                
    dlnsigma = _dln_sigma_dlnR(M, z, kappa, cosmo)
    n_eff = -2 * dlnsigma -3

    def _g(x):
        return math.log(1 + x) - x / (1 + x)
    
    def _G(x, n_eff):
        return x / _g(x) ** ((5 + n_eff)/6)
        
    func = Nc.GrowthFunc.new()
    func.prepare(cosmo)
    dD = Nc.GrowthFunc.eval_deriv(func, cosmo, z)
    D = Nc.GrowthFunc.eval(func, cosmo, z)
    alpha_eff = -dD * (1.0 + z) / D

    s = sigmafunc(M, z, cosmo)
    nu = 1.68647 / s

    A_neff = a0 * (1 + a1 * (n_eff + 3))
    B_neff = b0 * (1 + b1 * (n_eff + 3))
    C_aeff = 1 - c_alpha * (1 - alpha_eff)

    y = A_neff / nu * (1 + nu **2 / B_neff)

    def _G_inv(y, n_eff):
        if isinstance(y, np.ndarray):
            i = 0
            x_rt = np.array([])
            while i < len(y):
                def f(x):
                    x_value = _G(x, n_eff) - y[i]
                    x_value = np.asarray(x_value)
                    return x_value[i]
                x = brentq(f, a=0.05, b=200)
                x_rt = np.append(x_rt, x)
                i = i + 1

            return x_rt
            
        else:
            def f(y):
                func = lambda x: _G(x, n_eff) - y
                x_root = brentq(func, a=0.05, b=200)
                return x_root
            
            x_roots = f(y)
            return x_roots

    c = C_aeff * _G_inv(y, n_eff)

    return c    
