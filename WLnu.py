import numpy as np
import scipy as sp
import math
from scipy.integrate import quad
from functools import partial
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.special import gamma
import pandas as pd
from classy import Class  
from sys import exit

H0 = 1/2997.92458 # this is a global variable (Hubble constant) due to depends on h


"""
    CLASS MODULE
"""

def class_module(M_nu):
    #Omega_i = w_i/h² , w_i: omega_i    """Fixed values: CosmoParams"""
    omega_b = 0.0223;        #Baryons
    h = 0.6711                #H0/100
    omega_ncdm = M_nu/93.14 # 0.00;   #massive neutrinos (for now)
    omega_cdm = 0.1128126 - omega_ncdm;      #CDM  Original  (el del .dat)
    # omega_cdm = 0.125;      #CDM   nuevo
    #omega_k = 0 -> default <-

    A_s = 2.7391e-9; #A_s = np.exp(logA_s)/(10**10);  #  
    n_s = 0.9667; 
    z_pk = 0.0;               #z evaluation

    "Computing pkl_cb using CLASS"
    Kmin = 0.000015 ;  Kmax = 2.0 #UNITS: 1/Mpc  # ***In nb allmodules Pk.dat had 890 values of k from 0.00001 to 521.88
        
    params = {
                'output':'mPk',
                'omega_b':omega_b,
                'omega_cdm':omega_cdm,
                'omega_ncdm':omega_ncdm, 
                'h':h,
                'A_s':A_s,
                'n_s':n_s,
                'P_k_max_1/Mpc':Kmax,
                'z_max_pk':10.,   #Default value is 10
                #'N_eff':3.046,
                'N_ur':2.046,    #2.046 #massless neutrinos 
                'N_ncdm':1 #massive neutrinos species
    }
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    #Specify k
    k = np.logspace(np.log10(Kmin*h), np.log10(Kmax*h), num = 591) #Mpc^-1

    #Extract pkl_cb (cb: CDM + baryons) 
    Plin = np.array([cosmo.pk_cb(ki, z_pk) for ki in k])

    inputpkT = np.array([k, Plin])

    "Extrapolating the pkl_cb"
    kcutmax = Kmax; kmax = 800;
    inputpkT = ExtrapolateHighkLogLog(inputpkT, kcutmax, kmax)

    return inputpkT



"""
    MODULES from allmodules.ipynb
"""

# required functions
def integrand(OmegaM, z): # i.e., 1/H(z')
    return 1/(H0*np.sqrt(OmegaM*(1+z)**3 + (1-OmegaM))) # from Friedmann eq

################### module itself ###################

def background(OmegaM, zT):

    ### Integration (trapeze) ###
    delta_z = zT[1] - zT[0]
    Nz = len(zT)
    start_time = time.time()

    chi_x = np.zeros(Nz) # is actually zT
    chi_y = np.zeros(Nz) # for the new method but is still the same chi_y at the end

    xp = xA = xB = 0.0

    z = zT[0]
    xA = integrand(OmegaM, z)
    chi_x[0], chi_y[0] = z, xA*delta_z
    z_prev = z

    for i in range(1,Nz):
        z = zT[i]
        xB = integrand(OmegaM, z)
        xp = xp + 0.5*(xA + xB)*delta_z
        chi_x[i], chi_y[i] = z, xp
        xA = xB
        z_prev = z

    end_time = time.time()
    print('The time used to calculate all the chi[z] values (integrate) is:', end_time - start_time, 's. \n')

    zMax = zT[-1]
    
    chi_y[0] = 0 # handly making the first point (0,0)
    chiOfz_T = np.concatenate((zT.reshape(-1,1),chi_y.reshape(-1,1)), axis = 1)
    chiOfz = CubicSpline(zT,chi_y)
    chi_inter = chiOfz(zT) #use interpolated function returned by "CubicSpline"
    # plot_radialcomovil(zT,chi_inter) # plot of the interpolation of chiOfz_T 
    
    chi_Max = (chiOfz_T[-1])[1]
    chi_min = (chiOfz_T[0])[1]
    #print('\n There are',len(chiOfz_T), "chi(z) values from chi_min=chi(z=",zT[0], ") =", chi_min, " Mpc/h to chi_Max=chi(z=", zT[-1], ") =",chi_Max, "Mpc/h")
   
    zOfchi_T = np.concatenate((chi_y.reshape(-1,1),zT.reshape(-1,1)), axis = 1)
    zOfchi = CubicSpline(chi_y,zT) # plots
    z_interp = zOfchi(chi_y) # plots
    #plot_zOfchi(chi_y, z_interp, chi_Max)
    
    aOfchi_y = 1/(1+zT)
    aOfchi_T = np.concatenate((chi_y.reshape(-1,1),aOfchi_y.reshape(-1,1)), axis = 1)
    aOfchi = CubicSpline(chi_y,aOfchi_y) #scale factor function a(chi) given by interpolation
    a_chi = aOfchi(chi_y)
    #plot_aOfchi(chi_y, a_chi, chi_Max)
    
    chi_Maxvalue=np.where(chi_y == chi_Max)[0][0]
    chi_minvalue=np.where(chi_y == chi_min)[0][0]

    a_min = a_chi[chi_Maxvalue] # min. value of "a" corresponding to chi_Max
    a_Max = a_chi[chi_minvalue] # Max. value of "a" corresponding to chi_min
    # print("\n a_min =", a_min, ";  a_Max =", a_Max)
    return chiOfz_T, zOfchi_T, aOfchi_T

############ MODULE Linear Growth D_+(a) ##############
# required functions for module
### Solving diff. eq. ### # defining functions f1 and f2(eta)
def f1(eta, Om_M):
    return 2.0-3.0/(2.0*(1.0+((1.0-Om_M)/Om_M)*math.exp(3.0*eta)))

def f2 (eta, Om_M):
    return 3.0/(2.0*(1.0+((1.0-Om_M)/Om_M)*math.exp(3.0*eta)))

def S(Om_M, x,eta):
    return [x[1],-f1(eta, Om_M)*x[1]+f2(eta, Om_M)*x[0]]


def Module_linG(Om_M, aOfchi_T):
    eta_ini = -8
    eta_fin = 0 # notice today a=1, eta=0
    Dplus_i = math.exp(eta_ini) # initial condition
    dDplus_i = math.exp(eta_ini) # initial condition
    eta_T = np.linspace(-8.0,0,81, endpoint = True)

    Dplus_sol, dDplus_sol = odeint(partial(S, Om_M), [Dplus_i, dDplus_i], eta_T).T

    #array of chi(z) values
    x_Dplus_a = np.exp(eta_T)
    y_Dplus_a = Dplus_sol/Dplus_sol[-1]

    DplusOfa = CubicSpline(x_Dplus_a,y_Dplus_a, extrapolate=True)
    a = np.linspace(0.005,1,1000, endpoint = True)
    DplusOfa_interp = DplusOfa(a)

    a_chi = aOfchi_T[:,1]
    DplusOfchi = CubicSpline(aOfchi_T[:,0], DplusOfa(a_chi), extrapolate= True)
    DplusOfchi_y = DplusOfchi(aOfchi_T[:,0])

    DplusOfa_T = np.concatenate((x_Dplus_a.reshape(-1,1),y_Dplus_a.reshape(-1,1)), axis = 1)
    DplusOfchi_T = np.concatenate((aOfchi_T[:,0].reshape(-1,1),DplusOfchi_y.reshape(-1,1)), axis = 1)

    return DplusOfa_T, DplusOfchi_T


###############################################
##########   MODULE: LENS EFFICIENCY ##########

# defining g_L and q_L using Dirac Delta
def gLDiracDelta(chi, chiBin):
    return (1-chi/chiBin)*np.heaviside(chiBin-chi,0)

def qDiracDelta(chi, chiBin, Om_M, aOfchi):
    return 3/2*(H0**2 *Om_M)*(chi/aOfchi(chi))*gLDiracDelta(chi, chiBin)

# It says W_input but in this case W=DD, so the input in anycase is actually np.heaviside
# module itself
def Module_lens_eff(zT,aOfchi_T, Om_M): # remember chi_y is aOfchi_T[:,0] (from the back output)
    aOfchi_y = 1/(1+zT)   # *** aOfchi_y == aOfchi_T[:,1]
    chiOfz = CubicSpline(zT,aOfchi_T[:,0])
    aOfchi = CubicSpline(aOfchi_T[:,0],aOfchi_y) # aOfchi_y == aOfchi_T[:,1]

    zBin = 1.0
    chiBin = chiOfz(zBin)
    chimax = 3000
    #chimax_lensEff = chimax # we use now the real Xmax in qT
    chiBin_lensEff = chiBin
    sizeOfchiT = 100
    
    # making table of (i-1)*chimax/(sizeOfchiT-1) from 1 to the size 100
    chiT = np.zeros(100) #defining empty table
    for i in range(sizeOfchiT):
        chiT[i] = (i)*chimax/(sizeOfchiT-1) # filling with the values described above
    
    # Computing output chimaxinqT_lensEff
    qT = np.zeros(100)
    for i in range(len(chiT)):
        qT[i]= qDiracDelta(chiT[i],chiBin, Om_M, aOfchi)

    # OUTPUT (apart from chimax_lensEff, chiBin_lensEff)
    qT_lensEff = np.concatenate((chiT.reshape(-1,1),qT.reshape(-1,1)), axis = 1)

    indices_for_almost_nullqT = np.where(qT < 1/1000*qT[1])
    # these are the indeces where qT 'vanishes'

    index_for_chimax = indices_for_almost_nullqT[0][1] - 1
    # therefore, the max. value of chi before qT ~ 0 is
    chimaxinqT_lensEff = chiT[index_for_chimax]

    return qT_lensEff, chimaxinqT_lensEff, chiBin_lensEff

########## MODULE of CONVERGENCE POWER SPECTRUM ##########

### functions required for the module
def chimin_func(ell,kMax): # max and min values of chi(ell)
    return ell/kMax

def chiMax_func(ell,kmin):
    return ell/kmin

def PddLinear(ell,chi, DplusOfchi, pkl):
    return DplusOfchi(chi)**2*pkl(ell/chi)

def power(ell,chi, DplusOfchi, pkl):
    return PddLinear(ell,chi, DplusOfchi, pkl)

######### Extrapolation from Modules_RealSpace.py code #########
"""
This part of the code focuses on the extrapolation of the inputpkT 
"""
def LinearRegression(inputxy):
    xm = np.mean(inputxy[0])
    ym = np.mean(inputxy[1])
    Npts = len(inputxy[0])
   
    SS_xy = np.sum(inputxy[0]*inputxy[1]) - Npts*xm*ym
    SS_xx = np.sum(inputxy[0]**2) - Npts*xm**2
    m = SS_xy/SS_xx
    b = ym - m*xm
    return (m, b)

def Extrapolate(inputxy, outputx):   
    m, b = LinearRegression(inputxy)
    outxy = [(outputx[ii], m*outputx[ii]+b) for ii in range(len(outputx))]
    return np.array(np.transpose(outxy))


def ExtrapolateHighkLogLog(inputT, kcutmax, kmax):
    cutrange = np.where(inputT[0]<= kcutmax)
    inputcutT = np.array([inputT[0][cutrange], inputT[1][cutrange]])
    listToExtT = inputcutT[0][-6:]
    tableToExtT = np.array([listToExtT, inputcutT[1][-6:]])
    delta = np.log10(listToExtT[2])-np.log10(listToExtT[1])
    lastk = np.log10(listToExtT[-1])
   
    logklist = [];
    while (lastk <= np.log10(kmax)):
        logklistT = lastk + delta;
        lastk = logklistT
        logklist.append(logklistT)
    logklist = np.array(logklist)
   
    sign = np.sign(tableToExtT[1][1])
    tableToExtT = np.log10(np.abs(tableToExtT))
    logextT = Extrapolate(tableToExtT, logklist)   
    output = np.array([10**logextT[0], sign*10**logextT[1]])
    output = np.concatenate((inputcutT, output), axis=1)
    return output

def plot_convergencePS(ell_T, Ckappa_PS_T):
    plt.loglog(ell_T, Ckappa_PS_T, 'r-', label=r' $P^{3D}_\delta = P_{linear}(k=\frac{\ell}{\chi})$')
    font1 = {'family':'serif','color':'darkred','size':20}
    font2 = {'family':'serif','color':'black','size':22}
    plt.title(r'Galaxy convergence power spectrum', fontdict = font1)
    plt.xlabel(r"$ \ell$", fontdict = font2)
    plt.ylabel(r"$\ell (\ell +1)/2\pi \ C_\kappa (\ell)$ ", fontdict = font2)
    plt.grid()
    plt.legend(loc='right',bbox_to_anchor=(1.7, 0.5),fontsize = 'xx-large')
    plt.rcParams["figure.figsize"] = (7,5)
    plt.show()

############# Module itself ##############

def Module_Convergence_PS(inputpkT, Om_M, zT, DplusOfa_T, aOfchi_T, chimaxinqT_lensEff): # ***Note we're using here chimaxinqT_lensEff and not the default chimax_qTchimax=3000
    # note that in this case pkT has k & pk not as columns but as rows
    k = inputpkT[0,:] # i.e., k is the first row (array with all the columns) 
    pk = inputpkT[1,:] # and pk is the second col.
    pkl = CubicSpline(k,pk)

    kmin = inputpkT[0][0]
    kMax = inputpkT[0][-1]
    #print('The min. and max. values of k are, respect., kmin =', kmin, 'and kMax =', kMax,'.')

    Nell = 120 #number of ell values
    ellmin = 1
    ellMax = 100000
    delta = math.log10(ellMax/ellmin)/(Nell-1)

    ell_T = np.zeros(Nell)
    for i in range(Nell):
        ell_T[i] = 10**(math.log10(ellmin) + delta*(i))

    #print('There are', len(ell_T), 'log-spaced ell points, between ell_min=', min(ell_T), 'and ell_Max=', max(ell_T), '.')

    ## all this comes from the D+ module (ODE)
    DplusOfa = CubicSpline(DplusOfa_T[:,0],DplusOfa_T[:,1], extrapolate=True)
    a_chi = aOfchi_T[:,1]
    DplusOfchi = CubicSpline(aOfchi_T[:,0], DplusOfa(a_chi), extrapolate= True)
    
    chiMax = chimaxinqT_lensEff # Note this is different from chi_Max of first module
    sizeOfchiT = 100
    chiT = np.zeros(100) # this was defined previously but has to be done again in each module
    for i in range(sizeOfchiT):
        chiT[i] = (i)*chiMax/(sizeOfchiT-1) # filling with the values described above
    # **** NOTE: we're used above chimaxinqT_lensEff ~ 2300 instead of default 3000
    
    chiOfz = CubicSpline(zT,aOfchi_T[:,0])
    aOfchi = CubicSpline(aOfchi_T[:,0],aOfchi_T[:,1])
    zBin = 1.0
    chiBin = chiOfz(zBin)

    qT = np.zeros(100) # defining again in this new module
    for i in range(len(chiT)):
        qT[i]= qDiracDelta(chiT[i],chiBin, Om_M, aOfchi)

    q = CubicSpline(chiT, qT)
    Nchi = 100
    Ckappa_T = np.zeros((len(ell_T),2))

    # runtime count
    start_ck = time.time()
    ones_vect = np.ones(Nchi) # this is a vector of dimension Nchi full of ones
    m_Nchis = np.arange(Nchi)

    for l in range(len(ell_T)):
        ell = ell_T[l] #choosing every particular value of ell
        chimin_ell = chimin_func(ell, kMax) #functions, not val.
        chiMax_eval = chiMax_func(ell, kmin)
        chiMax_ell = min(chiMax,chiMax_eval)
        #defining the max value as the min. between chi_Max (value) and chimax(ell) (evaluated function)
        
        delta_chi = (chiMax_ell - chimin_ell)/(Nchi-1)
        
        chi_Table = chimin_ell*ones_vect + delta_chi*m_Nchis
        
        pkappa, pkappa_B = 0, 0
        chi_A = chi_Table[0]
        pkappa_A = (q(chi_A)*q(chi_A)*power(ell,chi_A, DplusOfchi, pkl))/(chi_A**2)
        
        for n in range(1,len(chi_Table)):
            
            chi_B = chi_Table[n]
            pkappa_B = (q(chi_B)*q(chi_B)*power(ell,chi_B, DplusOfchi, pkl))/(chi_B**2)
            delta_chi = chi_B - chi_A
            
            pkappa = pkappa + (pkappa_A + pkappa_B)/2*delta_chi
            chi_A = chi_B
            pkappa_A = pkappa_B
        
        Ckappa_T[l] = [ell,pkappa]
            
    # showing the time used to compute this calculation

    end_ck = time.time()
    print('The time used to compute the Convergence PS integral ("handly") was ',end_ck - start_ck, 's')

    Ck_ell = Ckappa_T[:,0]
    Ck_pkappa = Ckappa_T[:,1]
    
    Ckappa_linear = CubicSpline(Ck_ell, Ck_pkappa)

    Ckappa_PS_T = ell_T*(ell_T+1)/(2*np.pi)*Ckappa_linear(ell_T) # FOR PLOT
    
    plot_convergencePS(ell_T, Ckappa_PS_T)  # PLOTTING

    return Ckappa_T, chiT, qT
    



##################################################################################
########## 5) MODULE of CORRELATION FUNCTIONS xi+- ##########

### functions required for the module
def arcmin_to_rad(thethaArcMin):   # conversion from rad to arcmin
    return thethaArcMin*math.pi/(180*60)

def xiPlusf_T(theta, xiPlus_product, cm_T, Am_T, Im0_T, am_T):  # 'f' stands for 'function'
    for i in range(len(cm_T)):
        xiPlus_product[i] = cm_T[i]*Am_T[i]*Im0_T[i]*theta**(-am_T[i]-1)
    return np.sum(xiPlus_product)

def xiMinusf_T(theta, xiMinus_product, cm_T, Am_T, Im4_T, am_T):
    for i in range(len(cm_T)):
        xiMinus_product[i] = cm_T[i]*Am_T[i]*Im4_T[i]*theta**(-am_T[i] - 1)
    return np.sum(xiMinus_product)

def plot_XiFFT(t_xiplot, xi_P, xi_M):
    plt.plot(t_xiplot,xi_P, "-", label=r'$\xi_+$ (lin)')
    plt.plot(t_xiplot,xi_M, "-", label=r'$\xi_-$ (lin)')
    font1 = {'family':'serif','color':'darkred','size':16}
    font2 = {'family':'serif','color':'black','size':16}
    plt.legend(loc='right',bbox_to_anchor=(1.4, 0.5),fontsize = 'xx-large')
    plt.title(r'Correlation functions (linear order)', fontdict = font1)
    plt.xlabel(r"$\theta$ [arcmin]",fontdict = font2)
    plt.ylabel(r"$\theta \xi_{+,-}(\theta)$",fontdict = font2)
    # plt.ylim(0,5)
    plt.xscale('log')
    plt.show()


################### MODULE itself ####################
def Module_FFT(inputpkT, chiOfzT_backg, chimaxinqT_lensEff, DplusOfa_T, aOfchi_T, chiT, qT):
    Nd = 60
    interval = math.log10(arcmin_to_rad(200)/arcmin_to_rad(3))/(Nd-1)

    tangle_T = np.zeros(Nd)
    for i in range(Nd):
        tangle_T[i] = 10**(math.log10(arcmin_to_rad(3))+i*interval)

    N_fftlog =128 # even number chosen of the form 2^n 
    kmin_fft = 1e-4
    kMax_fft = 10
    nu_bias = -1.3

    int_fft = math.log(kMax_fft/kmin_fft)/(N_fftlog-1)

    kT_fft = np.zeros(N_fftlog)
    for i in range(N_fftlog):
        kT_fft[i] = kmin_fft*math.exp(i*int_fft)

    k = inputpkT[0,:] # i.e., k is the first row (array with all the columns) 
    pk = inputpkT[1,:] # and pk is the second col.
    pkl = CubicSpline(k,pk)

    toFFT_T = np.zeros(N_fftlog)
    for i in range(N_fftlog):
        toFFT_T[i] = pkl(kT_fft[i])*(kT_fft[i]/kmin_fft)**(-nu_bias)

    etam_fft_T = np.zeros(N_fftlog+1, dtype=np.complex_)
    for i in range(N_fftlog+1):
        etam_fft_T[i] = nu_bias + 2*math.pi*1j*(i - N_fftlog/2)*(N_fftlog-1)/(math.log(kMax_fft/kmin_fft)*(N_fftlog))

    pre_cm_T = np.fft.fft(toFFT_T, norm = "forward") # norm = "forward" is important to get the right fourier parameters

    cm_T = np.zeros(N_fftlog+1, dtype=np.complex_)

    for i in range(N_fftlog+1):
        if i-N_fftlog/2 < 0: # de i=0 a 63    ---> dan los pre de 64 a 1
            cm_T[i] = kmin_fft**(-etam_fft_T[i])*np.conj(pre_cm_T[-i + N_fftlog//2])
        else:   # de i=64 a 128 (129 ya no lo toma en cuenta) ---> dan los pre de 0 a 64
            cm_T[i] = kmin_fft**(-etam_fft_T[i])*pre_cm_T[i - N_fftlog//2]
            
    cm_T[0] = cm_T[0]/2
    cm_T[-1] = cm_T[-1]/2

    result = np.concatenate((cm_T.reshape(-1,1),etam_fft_T.reshape(-1,1)), axis = 1)

    chiMax_backg = chiOfzT_backg[-1][1]
    chimin_backg = chiOfzT_backg[0][1]

    Nchi_fft = 150
    chimin_fft = max(0.0001, chimin_backg)
    chiMax_fft = min(chimaxinqT_lensEff, chiMax_backg) # instead of default chimax_lensEff = 300
    deltachi_fft = math.log10(chiMax_fft/chimin_fft)/(Nchi_fft-1)

    chiT_fft = np.zeros(Nchi_fft)
    for i in range(Nchi_fft):
        chiT_fft[i] = 10**(math.log10(chimin_fft) + deltachi_fft*i)

    ## all this comes from the D+ module (ODE)
    DplusOfa = CubicSpline(DplusOfa_T[:,0],DplusOfa_T[:,1], extrapolate=True)
    a_chi = aOfchi_T[:,1]
    DplusOfchi = CubicSpline(aOfchi_T[:,0], DplusOfa(a_chi), extrapolate= True)
   
    q = CubicSpline(chiT, qT)

    Am_T = np.zeros(len(etam_fft_T),dtype=np.complex_)
    start_am = time.time()

    for j in range(len(etam_fft_T)):   #
        etam = etam_fft_T[j]
        am = 1 + etam
        
        Am = 0
        Am_B = 0
        chifft_A = chiT_fft[0]
        Am_A = q(chifft_A)*q(chifft_A)*DplusOfchi(chifft_A)*DplusOfchi(chifft_A)*chifft_A**(-am-1)
        
        for k in range(len(chiT_fft)):
            chifft_B = chiT_fft[k]
            Am_B = q(chifft_B)*q(chifft_B)*DplusOfchi(chifft_B)*DplusOfchi(chifft_B)*chifft_B**(-am-1)
            deltachi_am = chifft_B - chifft_A
            Am = Am + (Am_A + Am_B)/2*deltachi_am
            chifft_A = chifft_B
            Am_A = Am_B
            
        Am_T[j] = Am

    end_am = time.time()
    print("The time used to calculate A_m terms is",end_am - start_am, 's.')

    am_T = 1 + etam_fft_T

    Im0_T = np.zeros(len(etam_fft_T), dtype=np.complex_)
    for i in range(len(etam_fft_T)):
        Im0_T[i] = (2**(-1 + am_T[i])*gamma(1/2*(1+am_T[i])))/(math.pi*gamma(1/2*(1-am_T[i])))

    Im4_T = np.zeros(len(etam_fft_T),dtype=np.complex_)
    for i in range(len(etam_fft_T)):
        Im4_T[i] = (2**(-1 + am_T[i])*gamma(1/2*(5+am_T[i])))/(math.pi*gamma(1/2*(5-am_T[i])))

    xiPlus_product = np.zeros(len(cm_T), dtype=np.complex_)
    real_xiP_oftangle = np.zeros(len(tangle_T))

    for i in range(len(tangle_T)):
        real_xiP_oftangle[i] = np.real(xiPlusf_T(tangle_T[i], xiPlus_product, cm_T, Am_T, Im0_T, am_T))

    xiPlus_T = np.concatenate((tangle_T.reshape(-1,1),real_xiP_oftangle.reshape(-1,1)), axis = 1)
    xiPlus = CubicSpline(tangle_T,real_xiP_oftangle, extrapolate = True)


    xiMinus_product = np.zeros(len(cm_T), dtype=np.complex_)
    real_xiM_oftangle = np.zeros(len(tangle_T))

    for i in range(len(tangle_T)):
        real_xiM_oftangle[i] = np.real(xiMinusf_T(tangle_T[i], xiMinus_product, cm_T, Am_T, Im4_T, am_T))

    xiMinus_T = np.concatenate((tangle_T.reshape(-1,1),real_xiM_oftangle.reshape(-1,1)), axis = 1)
    xiMinus = CubicSpline(tangle_T,real_xiM_oftangle, extrapolate = True)
    
    t_xiplot = np.linspace(3,200, num=1000, endpoint=True)
    xi_P = (10**4)*t_xiplot*xiPlus(t_xiplot*math.pi/(180*60)) # just in order to plot
    xi_M = (10**4)*t_xiplot*xiMinus(t_xiplot*math.pi/(180*60))

    plot_XiFFT(t_xiplot, xi_P, xi_M)

    return xiPlus_T, xiMinus_T, xi_P, xi_M, tangle_T



##################### 6) MODULE: Direct integration #####################
####################################################################


def intPlus(theta, ell1, Ckappa_linear):
    return 1/(2*math.pi)*ell1*sp.special.jv(0,ell1*theta)*Ckappa_linear(ell1)

def intMinus(theta, ell1, Ckappa_linear):
    return 1/(2*math.pi)*ell1*sp.special.jv(4,ell1*theta)*Ckappa_linear(ell1)


def xiP_plot_direct(t, xiPlus_direct):
    return (10**4)*t*xiPlus_direct(t*math.pi/(180*60))[:,1] # we specify we want the value from its second column (all the rows)

def plot_direct_integ(t_xiplot,xi_P_direct, xi_M_direct):
    plt.plot(t_xiplot,xi_P_direct, "-", label=r'$\xi_+$ (lin)')
    plt.plot(t_xiplot,xi_M_direct, "-", label=r'$\xi_-$ (lin)')

    font1 = {'family':'serif','color':'darkred','size':16}
    font2 = {'family':'serif','color':'black','size':16}

    plt.legend(loc='right',bbox_to_anchor=(1.4, 0.5),fontsize = 'xx-large')
    plt.title(r'Correlation functions (direct method)', fontdict = font1)
    plt.xlabel(r"$\theta$ [arcmin]",fontdict = font2)
    plt.ylabel(r"$\theta \xi_{+,-}(\theta)$",fontdict = font2)
    # plt.ylim(0,5)
    plt.xscale('log')
    plt.show()


################ Module itself #############

def Direct_integration(Ckappa_T, tangle_T):

    Ck_ell = Ckappa_T[:,0]
    Ck_pkappa = Ckappa_T[:,1]    
    Ckappa_linear = CubicSpline(Ck_ell, Ck_pkappa)

    ellmin = 1
    ellMax = 100000
    xiPlus_y_direct = np.zeros(len(tangle_T))
    
    for i in range(len(tangle_T)):
        xiPlus_y_direct[i] = quad(lambda ell: intPlus(tangle_T[i],ell, Ckappa_linear), ellmin, ellMax)[0]

    xiPlus_T_direct = np.concatenate((tangle_T.reshape(-1,1),xiPlus_y_direct.reshape(-1,1)), axis = 1)

    xiMinus_y_direct = np.zeros(len(tangle_T))

    for i in range(len(tangle_T)):
        xiMinus_y_direct[i] = quad(lambda ell: intMinus(tangle_T[i],ell, Ckappa_linear), ellmin, ellMax)[0]

    xiMinus_T_direct = np.concatenate((tangle_T.reshape(-1,1),xiMinus_y_direct.reshape(-1,1)), axis = 1)

    xiPlus_direct = CubicSpline(tangle_T,xiPlus_T_direct, extrapolate=True)
    xiMinus_direct = CubicSpline(tangle_T,xiMinus_T_direct, extrapolate=True)

    t_xiplot = np.linspace(3,200, num=1000, endpoint=True)

    xi_P_direct = xiP_plot_direct(t_xiplot, xiPlus_direct)
    xi_M_direct = xiP_plot_direct(t_xiplot, xiMinus_direct)

    plot_direct_integ(t_xiplot, xi_P_direct,xi_M_direct)



##################### 7) MODULE: NONLINEAR REGIME #####################
####################################################################

"""
                P L O T S
"""

def plot_plin_v_Ploop(k_all, pk_lin, pk_1loop, pk_13, pk_22):
    plt.loglog(k_all, pk_lin, label=r'$P_L$ (linear)', color='green')
    plt.loglog(k_all, np.abs(pk_13 + pk_22), label=r'$P_{1-loop}$')
    plt.loglog(k_all, pk_lin + np.abs(pk_13 + pk_22), label=r'$P_{NL} = P_L + P_{1-loop}$', color='red')


    font1 = {'family':'serif','color':'darkred','size':14}
    font2 = {'family':'serif','color':'black','size':12}

    plt.legend(loc='right',bbox_to_anchor=(1.4, 0.5),fontsize = 'x-large')
    plt.rcParams["figure.figsize"] = (5.5,4)
    plt.title(r'Linear and non-linear contributions to Power Spectrum', fontdict = font1)
    plt.xlabel(r"$k$ [h/Mpc]",fontdict = font2)
    plt.ylabel(r"$P(k)$  [(Mpc/h)$^3$]",fontdict = font2)
    # plt.xlim(1e-4,10)
    # plt.ylim(0.1,5e5)
    plt.show()

### functions required for the module

def arcmin_to_rad(thethaArcMin):   # conversion from rad to arcmin
    return thethaArcMin*math.pi/(180*60)

def xiPlusf_nl(theta, xiPlus_product_nl, cm_nl, Am_nl, Im0_nl, am_nl):  # 'f' stands for 'function'
    for i in range(len(cm_nl)):
        xiPlus_product_nl[i] = cm_nl[i]*Am_nl[i]*Im0_nl[i]*theta**(-am_nl[i]-1)
    return np.sum(xiPlus_product_nl)

def xiMinusf_nl(theta, xiMinus_product_nl, cm_nl, Am_nl, Im4_nl, am_nl):
    for i in range(len(cm_nl)):
        xiMinus_product_nl[i] = cm_nl[i]*Am_nl[i]*Im4_nl[i]*theta**(-am_nl[i] - 1)
    return np.sum(xiMinus_product_nl)


def plot_XiFFT_nl(t_xiplot, xi_P_nl, xi_M_nl):
    plt.plot(t_xiplot,xi_P_nl, "-", label=r'$\xi_+$ (loop)')
    plt.plot(t_xiplot,xi_M_nl, "-", label=r'$\xi_-$ (loop)')

    plt.legend(loc='right',bbox_to_anchor=(1.4, 0.5),fontsize = 'x-large')
    font1 = {'family':'serif','color':'darkred','size':14}
    font2 = {'family':'serif','color':'black','size':12}
    
    plt.title(r'Correlation functions (1-loop PS corrections)', fontdict = font1)
    plt.xlabel(r"$\theta$ [arcmin]",fontdict = font2)
    plt.ylabel(r"$\theta \xi_{+,-}(\theta)$",fontdict = font2)
    # plt.ylim(-2,5)
    plt.xscale('log')
    plt.show()

def plot_XiFFT_BOTH(t_xiplot, xi_P_nl, xi_M_nl, xi_P, xi_M):
    plt.plot(t_xiplot,xi_P, "--", label=r'$\xi_+$',color='red')
    plt.plot(t_xiplot,xi_M, "--", label=r'$\xi_-$',color='green')
    plt.plot(t_xiplot,xi_P+xi_P_nl, "-", label=r'$\xi_+$ nonlinear',color='red')
    plt.plot(t_xiplot,xi_M+xi_M_nl, "-", label=r'$\xi_-$ nonlinear',color='green')

    font1 = {'family':'serif','color':'darkred','size':14}
    font2 = {'family':'serif','color':'black','size':12}

    plt.legend(loc='right',bbox_to_anchor=(1.55, 0.5),fontsize = 'x-large')

    plt.title(r'Shear correlation functions (nonlinear order)', fontdict = font1)
    plt.xlabel(r"$\theta$ [arcmin]",fontdict = font2)
    plt.ylabel(r"$\theta \xi_{+,-}(\theta)$",fontdict = font2)

    plt.xlim(3,)
    plt.ylim(-.5,14)
    plt.xscale('log')

    plt.show()


################ Module itself ##################

def Module_nonlin(inputpkT, chiOfzT_backg, chimaxinqT_lensEff, DplusOfa_T, aOfchi_T, chiT, qT, xi_P, xi_M, tangle_T):

    k_class = inputpkT[0][:]
    print("There are", k_class.shape[0], " values of k, with \n k_min= ", min(k_class), "& k_max = ", max(k_class))

    pk_allT = pd.read_table('./pk_loop.dat', header=None) # pknl_mass00z0
    # the pandas input data had the columns headers " 1.k[h/Mpc], 2.Pklinear  3.P22  4.P13 "

        # converting from pd to np.arrays
    k_all  = pk_allT.iloc[:,[0]][0].to_numpy() 
    pk_lin = pk_allT.iloc[:,[1]][1].to_numpy() # linear ref PS 
    pk_22  = pk_allT.iloc[:,[2]][2].to_numpy() # non linear ref PS: 1-loop 22
    pk_13  = pk_allT.iloc[:,[3]][3].to_numpy() # non linear ref PS: 1-loop 13

    pk_l = CubicSpline(pk_allT[0], pk_allT[1]) # INTERPOLATION: getting the linear PS
    pk_1loop = CubicSpline(pk_allT[0], pk_allT[2] + pk_allT[3]) # getting the nonlinear PS part: 1-loop
    pk_class = CubicSpline(inputpkT[0], inputpkT[1])


    #defining the kmin & kMax values as the first & last elements in the array, respect.
    kmin_nl = k_all[0]
    kMax_nl = k_all[-1]
    #kMax_nl = 10

    """
        APPROACH: $P^{appr}_{1-loop}$
        using $P_L$ and $P_{1-loop}$ of reference obtained by input datafile (.dat)
    """
    pl_ref = pk_l(inputpkT[0])
    ploop_ref = pk_1loop(inputpkT[0])

    """
        Finally, with the new information corresponding to P_L(k) (inputpkT by Class)
        we compute P_1loop^appr with the approach in the equation above
    """

    quotient_plin = (inputpkT[1]/pl_ref)**2
    pk_1loop_appr = np.multiply(quotient_plin, ploop_ref)

    """
        FFTLog RUTINE FOR 1-LOOP CONTRIBUTION
    """

    kmax_nl = 100 # different from the k max. of .dat, this is for FFTLog rutine only
    pkloop_appr = CubicSpline(k_class, pk_1loop_appr) # interpolation of the pk_nl, is a function


    ###### compute cm_nl ######
    # Internal params (local variables) again:
    # Nd = 60
    # interval = math.log10(arcmin_to_rad(200)/arcmin_to_rad(3))/(Nd-1)
    # tangle_T = np.zeros(Nd)
    # for i in range(Nd):
    #     tangle_T[i] = 10**(math.log10(arcmin_to_rad(3))+i*interval)

    N_fftlog =128 # even number chosen of the form 2^n
    nu_bias = -1.3
    Nchi_fft = 150
    # Recalling the fisrt min. & max. value of chi in this code
    chiMax_backg = chiOfzT_backg[-1][1] # same chi_Max (7519.61)
    chimin_backg = chiOfzT_backg[0][1]  # same chi_min (0.0)

    chimin_fft = max(0.0001, chimin_backg)
    chiMax_fft = min(chimaxinqT_lensEff, chiMax_backg) # instead of default chimax_lensEff = 300
    deltachi_fft = math.log10(chiMax_fft/chimin_fft)/(Nchi_fft-1)

    # Defining necessary arrays
    int_nl = math.log(kmax_nl/kmin_nl)/(N_fftlog-1)

    kT_nl = np.zeros(N_fftlog)
    for i in range(N_fftlog):
        kT_nl[i] = kmin_nl*math.exp(i*int_nl)

    # array to apply FFT numpy function
    toFFT_nl = np.zeros(N_fftlog)
    for i in range(N_fftlog):
        toFFT_nl[i] = pk_1loop(kT_nl[i])*(kT_nl[i]/kmin_nl)**(-nu_bias)

    # defining an array for values of eta as an complex number
    etam_nlT = np.zeros(N_fftlog+1, dtype=np.complex_)
    for i in range(N_fftlog+1):
        etam_nlT[i] = nu_bias + 2*math.pi*1j*(i - N_fftlog/2)*(N_fftlog-1)/(math.log(kmax_nl/kmin_nl)*(N_fftlog))

    # Using the Discrete Fourier Transform from numpy
    pre_cm_nl = np.fft.fft(toFFT_nl, norm = "forward")    # parameters a=0, b=-1 are already in the definition
    # For a=-1, i.e., 1/n we make: norm = "forward"

    cm_nl = np.zeros(N_fftlog+1, dtype=np.complex_)

    for i in range(N_fftlog+1):
        if i-N_fftlog/2 < 0: # from i=0 to 63    ---> give the 'pre_cm' from 64 to 1
            cm_nl[i] = kmin_nl**(-etam_nlT[i])*np.conj(pre_cm_nl[-i + N_fftlog//2])
        else:   # from i=64 to 128 (129 is no longer considered) ---> give the 'pre_cm' from 0 to 64
            cm_nl[i] = kmin_nl**(-etam_nlT[i])*pre_cm_nl[i - N_fftlog//2]

    cm_nl[0] = cm_nl[0]/2
    cm_nl[-1] = cm_nl[-1]/2

    #result_nl = np.concatenate((cm_nl.reshape(-1,1),etam_nlT.reshape(-1,1)), axis = 1)


    chiT_nl = np.zeros(Nchi_fft)
    for i in range(Nchi_fft):
        chiT_nl[i] = 10**(math.log10(chimin_fft) + deltachi_fft*i)

    ## all this comes from the D+ module (ODE)
    DplusOfa = CubicSpline(DplusOfa_T[:,0],DplusOfa_T[:,1], extrapolate=True)
    a_chi = aOfchi_T[:,1]
    DplusOfchi = CubicSpline(aOfchi_T[:,0], DplusOfa(a_chi), extrapolate= True)
    q = CubicSpline(chiT, qT)

    # INTEGRATING
    Am_nl = np.zeros(len(etam_nlT),dtype=np.complex_)

    start = time.time()

    for j in range(len(etam_nlT)):
        etam = etam_nlT[j]
        am = 1 + etam
        
        Am = 0
        Am_B = 0
        chifft_A = chiT_nl[0]
        Am_A = q(chifft_A)*q(chifft_A)*(DplusOfchi(chifft_A))**4*chifft_A**(-am-1)
        
        for k in range(len(chiT_nl)):
            chifft_B = chiT_nl[k]
            Am_B = q(chifft_B)*q(chifft_B)*(DplusOfchi(chifft_B))**4*chifft_B**(-am-1)
            deltachi_am = chifft_B - chifft_A
            Am = Am + (Am_A + Am_B)/2*deltachi_am
            chifft_A = chifft_B
            Am_A = Am_B
            
        Am_nl[j] = Am

    end = time.time()
    print("The time used to execute this is",end - start, 's.')

    am_nl = 1 + etam_nlT

    Im0_nl = np.zeros(len(etam_nlT), dtype=np.complex_)
    for i in range(len(etam_nlT)):
        Im0_nl[i] = (2**(-1 + am_nl[i])*gamma(1/2*(1+am_nl[i])))/(math.pi*gamma(1/2*(1-am_nl[i])))

    Im4_nl = np.zeros(len(etam_nlT),dtype=np.complex_)
    for i in range(len(etam_nlT)):
        Im4_nl[i] = (2**(-1 + am_nl[i])*gamma(1/2*(5+am_nl[i])))/(math.pi*gamma(1/2*(5-am_nl[i])))

# computing A_m's
    xiPlus_product_nl = np.zeros(len(cm_nl), dtype=np.complex_)
    # creating array (t.angle, xi_+-(t.ang))
    real_xiP_nl = np.zeros(len(tangle_T)) # real part

    for i in range(len(tangle_T)):
        real_xiP_nl[i] = np.real(xiPlusf_nl(tangle_T[i], xiPlus_product_nl, cm_nl, Am_nl, Im0_nl, am_nl))

    # now putting them togheter (x is already known)
    xiPlus_nl = np.concatenate((tangle_T.reshape(-1,1),real_xiP_nl.reshape(-1,1)), axis = 1)

    # interpolating
    xiPlus_nl = CubicSpline(tangle_T,real_xiP_nl, extrapolate = True)


    # same for Minus component (Im4T)
    xiMinus_product_nl = np.zeros(len(cm_nl), dtype=np.complex_)

    real_xiM_nl = np.zeros(len(tangle_T))

    for i in range(len(tangle_T)):
        real_xiM_nl[i] = np.real(xiMinusf_nl(tangle_T[i], xiMinus_product_nl, cm_nl, Am_nl, Im4_nl, am_nl))

    xiMinus_nl = np.concatenate((tangle_T.reshape(-1,1),real_xiM_nl.reshape(-1,1)), axis = 1)

    # interpolating
    xiMinus_nl = CubicSpline(tangle_T,real_xiM_nl, extrapolate = True)

    t_xiplot = np.linspace(3,200, num=1000, endpoint=True)
    xi_P_nl = (10**4)*t_xiplot*xiPlus_nl(t_xiplot*math.pi/(180*60))
    xi_M_nl = (10**4)*t_xiplot*xiMinus_nl(t_xiplot*math.pi/(180*60))


    plot_XiFFT_nl(t_xiplot, xi_P_nl, xi_M_nl)
    

    plot_XiFFT_BOTH(t_xiplot, xi_P_nl, xi_M_nl, xi_P, xi_M)

    return xiPlus_nl, xiMinus_nl



"""
    ### CONVERGENCE PS and 2PCF XI+- OUTPUTS (nonlin) (function of all modules) ###
"""

def allmodules(OmegaM_in, zTableChi_in, inputpkT, method):
    # if (method != 1) or (method != 2):
        # print("Choose a method (1: FFTLog, 2: Direct integration)")
        # exit()

    chiOfzT_backg, zOfchiT_backg, aOfchiT_backg = background(OmegaM_in, zTableChi_in) #Mod. backg
    DplusOfa_linG, DplusOfchi_linG = Module_linG(OmegaM_in, aOfchiT_backg) #Mod. Dplus
    qT_lensEff, chimaxinqT_lensEff, chiBin_lensEff = Module_lens_eff(zTableChi_in, aOfchiT_backg, OmegaM_in) #Mod. lens eff
    CkappaT_kappaPS, chiT_out, qT_out = Module_Convergence_PS(inputpkT, OmegaM_in, zTableChi_in, DplusOfa_linG, aOfchiT_backg, chimaxinqT_lensEff)

    xiPlusT_FFT_output, xiMinusT_FFT_output, xi_P_output, xi_M_output, tangle_T_output = Module_FFT(inputpkT, chiOfzT_backg, chimaxinqT_lensEff, DplusOfa_linG, aOfchiT_backg, chiT_out, qT_out)

    if method == 1:
        start = time.time()
        xiPlus_nl, xiMinus_nl = Module_nonlin(inputpkT, chiOfzT_backg, chimaxinqT_lensEff, DplusOfa_linG, aOfchiT_backg,
                                            chiT_out, qT_out, xi_P_output, xi_M_output, tangle_T_output)
        end = time.time()
        print("Time used (FFTLog): ", (end - start), "s \n \n \n")

    elif method == 2:
        start = time.time()
        Direct_integration(CkappaT_kappaPS, tangle_T_output)
        end = time.time()
        print("Time used (direct): ", (end - start), "s \n \n \n")

    else:
        print(":>")
