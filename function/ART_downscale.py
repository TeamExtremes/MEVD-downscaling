import scipy
import numpy as np
import xarray as xr
import scipy.optimize
from scipy.special import gamma
from scipy.integrate import dblquad, nquad
from scipy.optimize import curve_fit, minimize, fsolve

def Quantile_manual_general(Tr, N, C, W):
    """
    Manual methid to compute quantiles for given return periods and parameters.
    Works whether N, C, W are provided as scalars or as 1D arrays.
    
    Parameters:
        Tr : array-like
            List/array of return periods (e.g., [10, 20, 50, 100]).
        N  : scalar or array-like
            Number of wet days per year.
        C  : scalar or array-like
            Scale parameter.
        W  : scalar or array-like
            Shape parameter.
            
    Returns:
        QQ : ndarray
            If N, C, W are arrays (with length > 1), returns an array of shape (len(N), len(Tr))
            where each row corresponds to one set of parameters.
            If N, C, W are scalars, returns a 1D array of length len(Tr).
    """
    # Ensure that N, C, W are numpy arrays.
    N = np.atleast_1d(N)
    C = np.atleast_1d(C)
    W = np.atleast_1d(W)
    
    # Initialize the output array with shape (len(N), len(Tr))
    QQ = np.zeros((len(N), len(Tr)))
    
    # Loop over each return period
    for idx, tr in enumerate(Tr):
        annual_non_exceed_prob = 1 - 1/tr  # scalar
        daily_non_exceed_prob = annual_non_exceed_prob ** (1 / N)  # broadcasted over N
        inside_log = 1 - daily_non_exceed_prob
        quantiles = C * (-np.log(inside_log)) ** (1 / W)
        QQ[:, idx] = quantiles  # each column corresponds to a return period
    
    # If the original parameters were scalars, return a 1D array instead of 2D.
    if QQ.shape[0] == 1:
        return QQ[0, :]
    else:
        return QQ

def compute_beta(WET_MATRIX_EXTRA, origin, target, new_spatial_scale, tscales_INTER):
    pos_xmin = np.argmin(np.abs(origin[0] - new_spatial_scale))
    pos_tmin = np.argmin(np.abs(origin[1] - tscales_INTER))
    pwet_origin = WET_MATRIX_EXTRA[pos_tmin, pos_xmin]

    pos_xmin = np.argmin(np.abs(target[0] - new_spatial_scale))
    pos_tmin = np.argmin(np.abs(target[1] - tscales_INTER))
    pwet_target = WET_MATRIX_EXTRA[pos_tmin, pos_xmin]

    beta = pwet_origin / pwet_target
    return beta

def str_exp_fun(x, d0, mu):
    '''
    Stretched exponential rainfall correlation function
    from eg Habib and krajewski (2003)
    or Villarini and Krajewski (2007)
    with 2 parameters d0, mu (value as x = 0 is 1)
    '''
    x = np.asarray(x) # transform to numpy array
    is_scalar = False if x.ndim > 0 else True # create flag for output
    x.shape = (1,)*(1-x.ndim) + x.shape # give it dimension 1 if scalar
    myfun = np.exp( -(x/d0)**mu)
    myfun = myfun if not is_scalar else myfun[0]
    return  myfun

def epl_fun(x, epsilon, alpha):
    '''
    Marco's Exponential kernel + Power law tail
    (autocorrelation) function with exponential nucleus and power law decay
    for x> epsilon - 2 parameters
    see Marani 2003 WRR for more details
    '''
    x = np.asarray(x) # transform to numpy array
    is_scalar = False if x.ndim > 0 else True # create flag for output
    x.shape = (1,)*(1-x.ndim) + x.shape # give it dimension 1 if scalar
    m = np.size(x)
    myfun = np.zeros(m)
    for ii in range(m):
        if x[ii] < 10e-6:
            myfun[ii] = 1
        elif x[ii] < epsilon:
            myfun[ii] = np.exp(-alpha*x[ii]/epsilon)
        else:
            myfun[ii] = (epsilon/(np.exp(1)*x[ii]))**alpha
    myfun = myfun if not is_scalar else myfun[0]
    return  myfun

def myfun_sse(xx, yobs, parhat, L, acf):
    xx = np.asarray(xx)
    myacf = lambda x, y, parhat: myacf_2d(x, y, parhat, acf=acf)
    # Ty = np.array([L, 0., L, 0.])
    sse = 0
    m = np.size(xx)
    for ii in range(m):
        myx = xx[ii]
        Tx = np.array([np.abs(L - myx), myx, L + myx, myx])
        # L2 = [L, L]
        # res = corla_2d(L2, L2, parhat, myacf, Tx, Ty, err_min=1e-5)
        # faster option: - same result (optimized)
        res = fast_corla_2d(parhat, myacf, Tx, L, err_min=1e-2)
        sse = sse + (res - yobs[ii]) ** 2
        # sse = sse + ((res - yobs[ii])/yobs[ii]) ** 2 # norm
    # print('sse', sse)
    # print('parhat =', parhat)
    return sse

def myacf_2d(x, y, parhat, acf): #default acf = 'str'
    '''########################################################################
    set of 2D autocorrelation functions
    INPUTS::
        x,y = variables of the ACF
        parhat = set of the two parameters of the ACF
        acf: which ACF. possible choices:
            acf = 'str': stretched exponential ACF,
                WITH PARAMETERS: scale d0 and shape mu
            acf = 'mar': Marani 2003 exponential kernel + power law tail
                WITH PARAMETERS: transition point epsilon and shape alpha
    OUTPUTS::
        value of the ACF at a point
    ########################################################################'''
    d = np.sqrt(x**2 + y**2)
    if acf == 'str': # par d0, mu0
        d0 = parhat[0]
        mu = parhat[1]
        return np.exp( -(d/d0)**mu)
    elif acf == 'mar': # par:: epsilon, alpha
        # print('it actually is a Power Law')
        epsilon = parhat[0]
        alpha   = parhat[1]
        return epl_fun(d, epsilon , alpha)

def fast_corla_2d(par_acf, myacf, Tx, L, err_min=1e-2):
    nab_1 = nabla_2d(par_acf, myacf, L, Tx[0], err_min=err_min)
    nab_2 = nabla_2d(par_acf, myacf, L, Tx[1], err_min=err_min)
    nab_3 = nabla_2d(par_acf, myacf, L, Tx[2], err_min=err_min)
    nab_den = nabla_2d(par_acf, myacf, L, L, err_min=err_min)
    if np.abs(nab_den) < 10e-6: # to avoid infinities
        # print('correcting - inf value')
        nab_den = 10e-6
    covla = 2*(nab_1 -2*nab_2 + nab_3)/(4*nab_den)
    # print('parhat =', par_acf)
    return covla

def nabla_2d(par_acf, myacf, T1, T2, err_min = 1e-2):
    '''########################################################################
    EQUATION 13
    Compute the variance function as in Vanmarcke's book.
    INPUTS::
        par_acf = tuple with the parameters of the autocorr function (ACF)
        myacf = ACF in 1,2,or 3 dimensions, with parameters in par_acf
        T1 = 1st dimension of averaging area
        T2 = 2nd dim of the averaging area'''
    if (T1 == 0) or (T2 == 0):
        print('integration domain is zero')
        return 0.0 # integral is zero in this case.
    else:
        fun_XY = lambda x ,y: (T1 - x ) *(T2 - y ) * myacf(x ,y, par_acf)
        myint, myerr = nquad(fun_XY, [[0. ,T1], [0. ,T2]])
        # if myint != 0.0:
        #     rel_err = myerr /myint
        # else:
        #     rel_err = 0
        # if rel_err > err_min:
        #     print('varfun ERROR --> Numeric Integration scheme does not converge')
        #     print('int rel error = ', rel_err)
        #     sys.exit("aargh! there are errors!") # to stop code execution
        return 4.0 * myint

def vrf(L, L0, par_acf, acf):
    '''-------------------------------------------------------------
    compute the variance reduction factor
    between scales L (large) and L0 (small)
    defined as Var[L]/Var[L0]
    INPUT:
        L [km]
        L0 [Km]
        parhat = tuple with ACF parameters (eg epsilon, alpha)
        acf='mar' type of acf (mar or str available)
    OUTPUT:
        gam, variance reduction factor
        ---------------------------------------------------------'''
    def myacf(x, y, parhat, acf):
        if acf == 'str':  # par d0, mu0
            return str_exp_fun(np.sqrt(x ** 2 + y ** 2), parhat[0], parhat[1])
        elif acf == 'mar':  # par:: epsilon, alpha
            return epl_fun(np.sqrt(x ** 2 + y ** 2), parhat[0], parhat[1])
        else:
            print('vrf ERROR: insert a valid auto correlation function')
    # compute variance reduction factor
    fun_XY = lambda x, y: (L - x) * (L - y) * myacf(x, y, par_acf, acf)
    fun_XY0 = lambda x, y: (L0 - x) * (L0 - y) * myacf(x, y, par_acf, acf)
    # its 2D integral a-la Vanmarcke
    int_XY, abserr   = dblquad(fun_XY,  0.0, L,  lambda x: 0.0, lambda x: L)
    int_XY0, abserr0 = dblquad(fun_XY0, 0.0, L0, lambda x: 0.0, lambda x: L0)

    # gam  = 4/L**4 * int_XY # between scale L and a point
    # gam = (L0 / L) ** 4 * (int_XY / int_XY0)  # between scales L and L0
    if L0 == 0:
        gam = 4/L**4 * int_XY
    else:
        gam = (L0 / L)** 4 * (int_XY / int_XY0) 
    
    return gam

def down_wei(Ns, Cs, Ws, L, L0, beta, par_acf, acf):
    ''' -----------------------------------------------------------------------
    Downscale Weibull parameters from grid cell scale to a subgrid scale:
    compute the downscaled weibull parameters from a large scale L
    to a smaller scale L0, both in Km

    INPUT ::
        Ns, Cs, Ws (original parmaters at scale L - may be scalars or arrays)
        L [km] large scale
        L0 [km] small scale
        gams = ratio between the mean number of wet days at scales L and L0
        (should be larger than 1)
        par_acf: set of paramaters for the acf (tuple)

    OPTIONAL ARGUMENTS ::
        acf: what acf. default is 'str'. others available:
            'str'  ->  for 2p stretched exonential (Krajewski 2003)
            'mar'  ->  for 2p power law with exp kernel (Marani 2003)

        N_bias:: correction to N if comparing satellite and gauges
        default is equal to zero (no correction)

    OUTPUT::
        Nd, Cd, Wd (downscaled parameters)
        gam = variance reduction function
        fval = function value at the end of numerical minimization
    ---------------------------------------------------------------------'''
    Ns = np.asarray(Ns)  # check if scalar input - should be the same for N,C,W
    Cs = np.asarray(Cs)
    Ws = np.asarray(Ws)
    # the three parameter mush have same shape - I only check one here
    is_scalar = False if Cs.ndim > 0 else True
    Ns.shape = (1,) * (1 - Ns.ndim) + Ns.shape
    Cs.shape = (1,) * (1 - Cs.ndim) + Cs.shape
    Ws.shape = (1,) * (1 - Ws.ndim) + Ws.shape
    m = Cs.shape[0]  # length of parameter arrays = number of blocks=
    gam = vrf(L, L0, par_acf, acf=acf)
    print(f'Gamma value: {gam}')

    # prob wet:: correct satellite N adding the average difference
    pws = np.mean(Ns) / 365.25
    Wd = np.zeros(m)
    Cd = np.zeros(m)
    Nd = np.zeros(m)
    for ii in range(m):
        cs = Cs[ii]
        ws = Ws[ii]
        rhs = (1/(gam*beta)) * (((2*ws*gamma(2 / ws))/((gamma(1/ws))**2)) + (gam-1)*pws)
        wpfun = lambda w: (2*w*gamma(2 / w)/(gamma(1/w))**2) - rhs

        res = fsolve(wpfun, 0.1, full_output=True,xtol=1e-06, maxfev=10000)
        
        Wd[ii] = res[0]
        info = res[1]
        fval = info['fvec']
        if fval > 1e-5:
            print('warning - downscaling function:: '
                    'there is something wrong solving fsolve!')
        Cd[ii] = (beta * Wd[ii]) * (cs / ws) * (gamma(1 / ws) / gamma(1 / Wd[ii]))
        Nd[ii] = int( np.rint( Ns[ii] / beta))

    # If Nd, Cd, Wd are a collection (example, list or array) and not a scalar, 
    # return all collection.
    Nd = Nd if not is_scalar else Nd[0]
    Cd = Cd if not is_scalar else Cd[0]
    Wd = Wd if not is_scalar else Wd[0]

    return Nd, Cd, Wd, gam, fval

def down_wei_beta_alpha(Ns, Cs, Ws, beta, gam):
    Ns = np.asarray(Ns)  # check if scalar input - should be the same for N,C,W
    Cs = np.asarray(Cs)
    Ws = np.asarray(Ws)
    # the three parameter mush have same shape - I only check one here
    is_scalar = False if Cs.ndim > 0 else True
    Ns.shape = (1,) * (1 - Ns.ndim) + Ns.shape
    Cs.shape = (1,) * (1 - Cs.ndim) + Cs.shape
    Ws.shape = (1,) * (1 - Ws.ndim) + Ws.shape
    m = Cs.shape[0]  # length of parameter arrays = number of blocks=

    # prob wet:: correct satellite N adding the average difference
    pws = np.mean(Ns) / 365.25
    Wd = np.zeros(m)
    Cd = np.zeros(m)
    Nd = np.zeros(m)
    for ii in range(m):
        rhs = (1/(gam*beta)) * (((2*Ws[ii]*gamma(2 / Ws[ii]))/((gamma(1/Ws[ii]))**2)) + (gam-1)*pws)
        wpfun = lambda w: (2*w*gamma(2 / w)/(gamma(1/w))**2) - rhs

        res = fsolve(wpfun, 0.1, full_output=True,xtol=1e-06, maxfev=10000)
        
        Wd[ii] =  res[0][0]
        info = res[1]
        fval = info['fvec']
        if fval > 1e-5:
            print('warning - downscaling function:: '
                    'there is something wrong solving fsolve!')
            # print(Cs[ii], Ws[ii], beta, gam)
        Cd[ii] = (beta * Wd[ii]) * (Cs[ii] / Ws[ii]) * (gamma(1 / Ws[ii]) / gamma(1 / Wd[ii]))
        Nd[ii] = int( np.rint( Ns[ii] / beta))

    # If Nd, Cd, Wd are a collection (example, list or array) and not a scalar, 
    # return all collection.
    Nd = Nd if not is_scalar else Nd[0]
    Cd = Cd if not is_scalar else Cd[0]
    Wd = Wd if not is_scalar else Wd[0]

    return Nd, Cd, Wd

def down_wei_beta_alpha_update(Ns, Cs, Ws, beta, gam):
    '''
    The difference with down_wei_beta_alpha is the use of try-except for errors,
    In the case solve dont found the solution, we use the original ws value.
    '''
    Ns = np.asarray(Ns)  # check if scalar input - should be the same for N,C,W
    Cs = np.asarray(Cs)
    Ws = np.asarray(Ws)
    # the three parameter mush have same shape - I only check one here
    is_scalar = False if Cs.ndim > 0 else True
    Ns.shape = (1,) * (1 - Ns.ndim) + Ns.shape
    Cs.shape = (1,) * (1 - Cs.ndim) + Cs.shape
    Ws.shape = (1,) * (1 - Ws.ndim) + Ws.shape
    m = Cs.shape[0]  # length of parameter arrays = number of blocks=

    # prob wet:: correct satellite N adding the average difference
    pws = np.mean(Ns) / 365.25
    Wd = np.zeros(m)
    Cd = np.zeros(m)
    Nd = np.zeros(m)
    for ii in range(m):
        cs = Cs[ii]
        ws = Ws[ii]
        rhs = (1/(gam*beta)) * (((2*ws*gamma(2 / ws))/((gamma(1/ws))**2)) + (gam-1)*pws)
        wpfun = lambda w: (2*w*gamma(2 / w)/(gamma(1/w))**2) - rhs

        try:
            res = fsolve(wpfun, 0.1, full_output=True, xtol=1e-06, maxfev=10000)
            Wd[ii] =  res[0][0]
            info = res[1]
            fval = info['fvec']
            
            if np.abs(fval) > 1e-5:
                raise ValueError("Fsolve no convergió adecuadamente")

        except (RuntimeError, ValueError) as e:
            print(f"Warning: Problema en fsolve para ii={ii}. Se usará un valor por defecto.")
            Wd[ii] = ws  # Usa ws como aproximación
        
        info = res[1]
        fval = info['fvec']
        if fval > 1e-5:
            print('warning - downscaling function:: '
                    'there is something wrong solving fsolve!')
            # print(Cs[ii], Ws[ii], beta, gam)
        Cd[ii] = (beta * Wd[ii]) * (cs / ws) * (gamma(1 / ws) / gamma(1 / Wd[ii]))
        Nd[ii] = int( np.rint( Ns[ii] / beta))

    # If Nd, Cd, Wd are a collection (example, list or array) and not a scalar, 
    # return all collection.
    Nd = Nd if not is_scalar else Nd[0]
    Cd = Cd if not is_scalar else Cd[0]
    Wd = Wd if not is_scalar else Wd[0]

    return Nd, Cd, Wd

def mev_fun(y, pr, N, C, W):
    ''' MEV distribution function, to minimize numerically
    for computing quantiles
    Updated version, to include accounting for dry years with 0 events'''
    nyears = N.size
    # mev0f = numzero + np.sum( ( 1-np.exp(-(y/Cn)**Wn ))**Nn) - nyears*pr
    mev0f = np.sum( ( 1-np.exp(-(y/C)**W ))**N) - nyears*pr
    return mev0f

# =========================================================================================================
# Quantiles 
# =========================================================================================================

def mev_quant(Fi, x0, N, C, W, thresh=1):
    # Ensure Fi is an array
    Fi = np.asarray(Fi)

    # Check if Fi is a scalar
    is_scalar = Fi.ndim == 0

    # Reshape Fi to handle scalars consistently
    Fi.shape = (1,) * (1 - Fi.ndim) + Fi.shape
    m = np.size(Fi)
    quant = np.zeros(m)
    flags = np.ones(m, dtype=bool)  # Flag for the convergence of numerical solver

    for ii in range(m):
        # Define the function to solve
        myfun = lambda y: mev_fun(y, Fi[ii], N, C, W)
        
        # Use fsolve to find the root
        res = scipy.optimize.fsolve(myfun, x0, full_output=True)
        quant[ii] = res[0] if np.ndim(res[0]) == 0 else res[0].item()  # Ensure res[0] is scalar
        info = res[1]
        fval = info['fvec']
        
        # Check for convergence issues
        if fval > 1e-5:
            print('ERROR - fsolve does not work - change x0')
            flags[ii] = False

    # Add the threshold to the quantile
    quant += thresh

    # Return as scalar if the input was scalar
    quant = quant[0] if is_scalar else quant
    flags = flags[0] if is_scalar else flags

    return quant, flags

def mev_quant_update(Fi, x0, N, C, W, thresh=1):
    Fi = np.asarray(Fi)
    is_scalar = Fi.ndim == 0
    Fi.shape = (1,) * (1 - Fi.ndim) + Fi.shape
    m = np.size(Fi)
    
    quant = np.zeros(m)
    flags = np.ones(m, dtype=bool)  # Flag de convergencia

    for ii in range(m):
        myfun = lambda y: mev_fun(y, Fi[ii], N, C, W)

        # Intenta resolver con x0 inicial
        res = scipy.optimize.fsolve(myfun, x0, full_output=True)
        solution = res[0] if np.ndim(res[0]) == 0 else res[0].item()
        fval = res[1]['fvec']
        
        # Si la solución no es buena, probar diferentes x0
        if np.abs(fval) > 1e-5:
            # print(f'⚠️ WARNING: fsolve falló para ii={ii}, probando otros x0')
            x0_list = [x0 * 0.5, x0 * 2, 1.0, 10.0]  # Lista de valores iniciales alternativos
            
            for new_x0 in x0_list:
                res = scipy.optimize.fsolve(myfun, new_x0, full_output=True)
                solution = res[0] if np.ndim(res[0]) == 0 else res[0].item()
                fval = res[1]['fvec']
                
                if np.abs(fval) <= 1e-5:
                    # print(f'✅ Solución encontrada con x0 = {new_x0}')
                    break
            else:
                # print(f'❌ ERROR: No se encontró solución para ii={ii} con ningún x0')
                flags[ii] = False  # Marcar que esta solución falló
        
        quant[ii] = solution

    # Añadir el umbral a los cuantiles
    quant += thresh

    # Convertir a escalar si la entrada era escalar
    quant = quant[0] if is_scalar else quant
    flags = flags[0] if is_scalar else flags

    return quant, flags

def pre_quantiles(data_in, Tr, lat, lon, dic_in, thresh):
    Fi = 1 - 1/Tr
    
    QUANTILE = np.zeros([len(Tr), len(lat), len(lon)])*np.nan

    for i in range(len(lat)):
        for j in range(len(lon)):
            data_tmp = data_in[dic_in['SC']][:,i,j].values
            x0 = 9.0*np.nanmean(data_tmp)
            if np.isnan(data_tmp).sum() == len(data_tmp):
                continue
            else:
                quant = mev_quant_update(Fi, x0, 
                                data_in[dic_in['WD']][:,i,j].values, 
                                data_in[dic_in['SC']][:,i,j].values, 
                                data_in[dic_in['SH']][:,i,j].values,
                                thresh=thresh)[0]
                QUANTILE[:,i,j] = quant

    return QUANTILE

def pre_quantiles_array(N, C, W, Tr, lat, lon, thresh):
    '''
    New version of pre_quantiles
    Compute quantiles using weibull parameter arrays (N, C, W)
    '''
    Tr = np.array(Tr)
    Fi = 1 - 1/Tr
    
    QUANTILE = np.zeros([len(Tr), len(lat), len(lon)])*np.nan

    for i in range(len(lat)):
        for j in range(len(lon)):
            data_tmp = C[:,i,j]
            x0 = 9.0*np.nanmean(data_tmp)
            if np.isnan(data_tmp).sum() == len(data_tmp):
                continue
            else:
                quant, flag = mev_quant_update(Fi, x0, 
                                N[:,i,j], 
                                C[:,i,j], 
                                W[:,i,j],
                                thresh=thresh)
                QQ = np.where(flag, quant, np.nan)
                QUANTILE[:,i,j] = QQ

    return QUANTILE

def quantile_correction(obs, model):
    """
    Perform quantile mapping to correct the model data based on observed data distribution.

    Parameters:
    - obs: 1D array of observed data
    - model: 1D array of model data

    Returns:
    - corrected_model: 1D array of corrected model data
    """
    # Remove NaN values if present
    obs = obs[~np.isnan(obs)]
    model = model[~np.isnan(model)]

    # Compute the sorted observed data and model data
    obs_sorted = np.sort(obs)
    model_sorted = np.sort(model)

    # Compute the quantiles
    obs_cdf = np.linspace(0, 1, len(obs_sorted))
    model_cdf = np.linspace(0, 1, len(model_sorted))

    # Interpolate the observed CDF values at the model's CDF positions
    corrected_model = np.interp(
        np.interp(model, model_sorted, model_cdf),  # Interpolate model values to their CDF
        obs_cdf, obs_sorted  # Interpolate the model's CDF to the observed data
    )

    return corrected_model

def gamma_manual(Ns, Cs, Ws, L, L0, par_acf, acf):
    Ns = np.asarray(Ns)  # check if scalar input - should be the same for N,C,W
    Cs = np.asarray(Cs)
    Ws = np.asarray(Ws)
    # the three parameter mush have same shape - I only check one here
    is_scalar = False if Cs.ndim > 0 else True
    Ns.shape = (1,) * (1 - Ns.ndim) + Ns.shape
    Cs.shape = (1,) * (1 - Cs.ndim) + Cs.shape
    Ws.shape = (1,) * (1 - Ws.ndim) + Ws.shape
    gam = vrf(L, L0, par_acf, acf=acf)
    return gam

# =========================================================================================================
# Weibull fit 
# =========================================================================================================

def wei_fit(sample):
    ''' fit a 2-parameters Weibull distribution to a sample
    by means of Probability Weighted Moments (PWM) matching (Greenwood 1979)
    using only observations larger than a value 'threshold' are used for the fit
    -- threshold without renormalization -- it assumes the values below are
    not present. Default threshold = 0
    INPUT:: sample (array with observations)
            threshold (default is = 0)
    OUTPUT::
    returns dimension of the sample (n) (only values above threshold)
    N represent the number of observations > threshold
    Weibull scale (c) and shape (w) parameters '''
    sample = np.asarray(sample) # from list to Numpy array
    wets   = sample[sample > 0.0]
    x      = np.sort(wets) # sort ascend by default
    M0hat  = np.mean(x)
    M1hat  = 0.0
    n      = x.size # sample size
    for ii in range(n):
        real_ii = ii + 1
        M1hat   = M1hat + x[ii]*(n - real_ii)
    M1hat = M1hat/(n*(n-1))
    c     = M0hat/gamma( np.log(M0hat/M1hat)/np.log(2)) # scale par
    w     = np.log(2)/np.log(M0hat/(2*M1hat)) # shape par
    return  n, c, w

def wei_fit_update(sample):
    sample = np.asarray(sample) # from list to Numpy array
    x      = np.sort(sample) # sort ascend by default
    M0hat  = np.mean(x)
    M1hat  = 0.0
    n      = x.size # sample size
    for ii in range(n):
        real_ii = ii + 1
        M1hat   = M1hat + x[ii]*(n - real_ii)
    M1hat = M1hat/(n*(n-1))
    c     = M0hat/gamma( np.log(M0hat/M1hat)/np.log(2)) # scale par
    w     = np.log(2)/np.log(M0hat/(2*M1hat)) # shape par
    return  n, c, w

def wei_fit_pwm(sample, threshold = 0): 
    ''' fit a 2-parameters Weibull distribution to a sample 
    by means of Probability Weighted Moments (PWM) matching (Greenwood 1979)
    using only observations larger than a value 'threshold' are used for the fit
    -- threshold without renormalization -- it assumes the values below are 
    not present. Default threshold = 0    
    INPUT:: sample (array with observations) threshold (default is = 0)
    OUTPUT::
    returns dimension of the sample (n) (only values above threshold)
    Weibull scale (c) and shape (w) parameters '''    
    sample = np.asarray(sample) # from list to Numpy array
    wets   = sample[sample > threshold]
    x      = np.sort(wets) # sort ascend by default
    M0hat  = np.mean(x)
    M1hat  = 0.0
    n      = x.size # sample size
    for ii in range(n): 
        real_ii = ii + 1
        M1hat   = M1hat + x[ii]*(n - real_ii) 
    M1hat = M1hat/(n*(n-1))
    c     = M0hat/gamma( np.log(M0hat/M1hat)/np.log(2)) # scale par
    w     = np.log(2)/np.log(M0hat/(2*M1hat)) # shape par
    return  n, c, w

def wei_fit_pwm_cens(sample, threshold = 0): 
    ''' fit a 2-parameters Weibull distribution to a sample 
    by means of censored Probability Weighted Moments (CPWM) - Wang, 1999
    only observations larger than a value 'threshold' are used for the fit
    but the probability mass of the observations below threshold is accounted for.
    compute the first two PWMs
    ar and br are linear comb of each other, perfectly equivalent
    I use censoring on the br as proposed by Wang 1990
    so that I am censoring the lower part of the distribution
    Default threshold = 0
    INPUT:: sample (array with observations) threshold (default is = 0)
    OUTPUT::
    returns numerosity of the sample (n) (only values above threshold)
    Weibull scale (c) and shape (w) parameters '''    
    sample = np.asarray(sample) # from list to Numpy array
    wets   = sample[sample > 0]
    x      = np.sort(wets) # sort ascend by default
    b0  = 0.0
    b1  = 0.0
    n      = x.size # sample size
    for ii in range(n): 
        real_ii = ii + 1
        if x[ii]>threshold:
            b1=b1+x[ii]*(real_ii-1)
            b0=b0+x[ii]
    b1=b1/(n*(n-1))
    b0=b0/n
    # obtain ar=Mrhat  as linear combination of the first two br
    M0hat = b0
    M1hat = b0 - b1
    c     = M0hat/gamma( np.log(M0hat/M1hat)/np.log(2)) # scale par
    w     = np.log(2)/np.log(M0hat/(2*M1hat)) # shape par
    return  n, c, w

def fit_yearly_weibull_update(xdata, thresh, maxmiss=36):
    '''
    This Function computes the Weibull fit parameters for each year in the data.
    Return NCW matrix [WET_DAYS (N), SCALE (S), SHAPE (W), YEARS]
    '''
    OBS_min = 366 - maxmiss
    years = np.unique(xdata.time.dt.year.values)
    years_num = np.size(years)

    NCW = np.zeros((years_num, 4))
    NOBS = np.zeros(years_num)

    for i, yy in enumerate(years):
        sample = xdata.sel(time=str(yy))
        NOBS[i] = len(sample[sample>=0])

        if NOBS[i] < OBS_min:
            print('Not enough data')
            NCW[i,:] = np.array([np.nan, np.nan, np.nan, yy])

        else:
            # excesses = sample[sample > thresh]
            excesses = sample[sample > thresh] - thresh
            
            Ni = np.size(excesses)
            if Ni == 0:
                NCW[i,:] = np.array([np.nan, np.nan, np.nan, yy])
            elif Ni == 1:
                NCW[i,:] = np.array([np.nan, np.nan, np.nan, yy])
            else:
                NCW[i,0:3] = wei_fit_update(excesses)
                NCW[i,0:3] = wei_fit(excesses)
                NCW[i,3] = yy

    return NCW

def weibull_year_parameters(DATA_in, lat_c, lon_c, thresh, maxmiss):
    lats = DATA_in['lat'].data
    lons = DATA_in['lon'].data

    i_ = np.where(lats==lat_c)[0][0]
    j_ = np.where(lons==lon_c)[0][0]

    IMERG_pixel_1dy = DATA_in['PRE'][:,i_,j_].data
    IMERG_pixel_1dy_xr = xr.DataArray(
                IMERG_pixel_1dy, 
                coords={'time':DATA_in['time'].values}, 
                dims=('time'))

    WEIBULL_YEAR = fit_yearly_weibull_update(
                    IMERG_pixel_1dy_xr, 
                    thresh=thresh,
                    maxmiss=maxmiss)

    return WEIBULL_YEAR

def down_year_parameters(N, C, W, BETA, GAMMA):
    yy, la, lo = N.shape
    Nd = np.zeros([yy, la, lo])
    Cd = np.zeros([yy, la, lo])
    Wd = np.zeros([yy, la, lo])
    
    for i in range(la):
        for j in range(lo):
            if np.isnan(BETA[i,j]) == True:
                Nd[:,i,j] = np.nan
                Cd[:,i,j] = np.nan
                Wd[:,i,j] = np.nan
            else:
                # Nd_, Cd_, Wd_ = down_wei_beta_alpha_update(N[:,i,j], C[:,i,j], W[:,i,j], BETA[i,j], GAMMA[i,j])
                Nd_, Cd_, Wd_ = down_wei_beta_alpha(N[:,i,j], C[:,i,j], W[:,i,j], BETA[i,j], GAMMA[i,j])
                Nd[:,i,j] = Nd_
                Cd[:,i,j] = Cd_
                Wd[:,i,j] = Wd_
    return Nd, Cd, Wd
