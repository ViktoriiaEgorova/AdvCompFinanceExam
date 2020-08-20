import numpy as np
import pandas as pd
import ACF
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats
import copy
from operator import itemgetter
from config import get_input_parms, loadConfig
import sys
from sys import argv
from sys import stdout as cout
from math import *


def heston(sigma_square, k, theta, eta, S0, r, delta_t, rho, N):
    S = np.ones((len(delta_t) + 1, 1))  # Where are stored the values of the asset

    n1 = int(np.round(np.sqrt(N)))

    n2 = int(np.round(np.sqrt(N)))

    for _ in range(0, n1):

        nu = np.hstack((np.array([sigma_square]), np.empty(len(delta_t))))

        temp_S = np.vstack((np.ones(n2) * S0, np.empty((len(delta_t), n2))))

        for j, dt in enumerate(delta_t):
            flag = nu[j] + k * (theta - nu[j]) * dt + eta * np.sqrt(nu[j] * dt) * np.random.normal(0, 1, 1)

            nu[j + 1] = max(0, flag)

            temp_S[j + 1] = temp_S[j] * np.exp((r - nu[j] / 2) * dt + np.sqrt(nu[j] * dt) * (
                        rho * np.random.normal(0, 1, n2) + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, n2)))

        S = np.hstack((S, temp_S))
    return S[:, 1:]


def empirical_cdf(array):
    array = np.sort(array)
    ecdf = np.empty((len(array) , 2))
    n = len(array)
    for i,x in enumerate(array):
        ecdf[i,0] = x
        ecdf[i,1] = len(array[0:i+1])/n
    return ecdf


def usage():
    print("Usage: $> python3 ESAME [options]")
    print("Options parameters:")
    print("    %-24s: this output" %("--help"))
    print("    %-24s: input data file with the rates (compulsory)" %("-in"))
    print("    %-24s: initial value of the underlying" %("-S0"))
    print("    %-24s: option strike" %("-strike"))
    print("    %-24s: option maturity" %("-T"))
    print("    %-24s: VaR time" %("-tM"))
    print("    %-24s: Starting value of the volatility" %("-s"))
    print("    %-24s: Black and Sholes volatility" %("-sBS"))
    print("    %-24s: Length of the time steps" %("-dt"))
    print("    %-24s: NUmber of simulations (2^n n = 20 by default)" %("-N"))
    print("    %-24s: k: the speed of adjustment of the Heston model" %("-k"))
    print("    %-24s: theta: the mean reverting factor" %("-theta"))
    print("    %-24s: eta: the eta of the Heston model" %("-eta"))
    print("    %-24s: rho: the correlations of the two Brownian motions" %("-rho"))
    print("    %-24s: Number of bins for the histogram of the profit and loss function" %("-bins"))
    print("    %-24s: seed: fixing the seed (default 100)" %("-seed"))

def run(args):
    #Default values
    
    N  = 22
    seed = 100

    np.random.seed(seed)

    sigma_BS = .24
    
    #Heston parameters
    sigma = .0635 # initial value for Heston
    delta_t = 1/12
    k = .1153
    theta = .0240
    eta = .01
    rho = .2125
    
    #Option data
    tM = .5
    T = 1.13 #Option maturity
    tM = np.round(tM,6)
    T = np.round(T,6)
    strike = 1.03
    S0 = 1

    #######THE CHECKING PART########

    parms = get_input_parms(args) #Per prendere i parametri da linea di comando
    n_bins = 1000 #Number of bins for the histogram of the profit and loss function

    try:
        op = parms["help"]
        usage()
        return
    except KeyError:
        pass

    inpt = parms["in"]
    PAR = loadConfig(inpt)

    try:
        S0 = float(parms["S0"])
    except KeyError:
        pass
    try:
        strike = float(parms["strike"])
    except KeyError:
        pass
    try:
        T = float(parms["T"])
    except KeyError:
        pass
    try:
        tM = float(parms["tM"])
    except KeyError:
        pass
    try:
        sigma = float(parms["s"])
    except KeyError:
        pass
    try:
        delta_t = float(parms["dt"])
    except KeyError:
        pass
    try:
        N = int(parms["N"])
    except KeyError:
        pass
    try:
        k = float(parms["k"])
    except KeyError:
        pass
    try:
        theta = float(parms["theta"])
    except KeyError:
        pass
    try:
        eta = float(parms["eta"])
    except KeyError:
        pass
    try:
        rho = float(parms["rho"])
    except KeyError:
        pass
    try:
        sigma_BS = float(parms["sBS"])
    except KeyError:
        pass
    try:
        n_bins = int(parms["bins"])
    except KeyError:
        pass
    try:
        seed = int(parms["seed"])
    except KeyError:
        pass

    ################################

    N = 2**N
    np.random.seed(seed)

    curve = PAR.curve
    final_TM = ACF.preparingStructure(curve, delta_t, T, tM)

    # COMPUTING THE PRICE OF THE PUT WITH BS
    index_T, index_tM, BS_put, BS_Call, P_0_T = ACF.put_call_price(final_TM, T, tM, S0, strike, sigma_BS)

    #MARTINGALE CHECK UP TO THE MATURITY OF THE OPTION

    time_steps_T = final_TM[:index_T+1,4] #Selecting the time steps from 0 to T. They will be needed for verifing the martingale property

    maturities = np.cumsum(time_steps_T)

    layout1 = "{0:>15}{1:>15}{2:>15}"
    layout2 = "{0:>15}{1:>15}"
    layout3 = "{0:>15}{1:>15}{2:>15}{3:>15}"
    layout4 = "{0:>24}{1:>24}{2:>24}{3:>24}{4:>24}"

    Asset_Heston_trj_T = heston(sigma, k, theta, eta, S0, 0, time_steps_T, rho, N)

    martingale = np.round(np.mean(Asset_Heston_trj_T, axis = 1),6) #Computing the mean along each row

    MC_error = np.sqrt((np.mean(Asset_Heston_trj_T ** 2, axis = 1) - np.mean(Asset_Heston_trj_T, axis = 1) ** 2)/N)

    print("Maringale check for the trajectories of the GBM used for the underlying")
    print()
    print(layout3.format('maturity',"Mean","Abs error","MC error"))
    print()
    for i,m in enumerate(martingale):
        if i != 0:
            print(layout3.format(maturities[i-1], martingale[i], np.absolute(np.round(S0-martingale[i],6)), 3*np.round(MC_error[i], 6)))
        else:
            print(layout3.format( 0, martingale[i], np.absolute(np.round(S0-martingale[i],6)), 3*np.round(MC_error[i], 6)))
    print()
    print()

    #ERROR ANALYSIS OF THE MONTE CARLO SIMULATION FOR PUT AND CALL OPTIONS

    MC_put_T = strike*P_0_T - Asset_Heston_trj_T[-1]
    MC_put_T[MC_put_T < 0] = 0
    MC_put_price_T = np.mean(MC_put_T) #Put price obtained with the MC method

    square_payoff = np.mean(MC_put_T**2)

    error = np.sqrt((square_payoff - MC_put_price_T**2)/N)

    print(layout4.format("MC put price","Theoretical price",'N of simulations',"Absolute error","Error"))
    print(layout4.format(np.round(MC_put_price_T, 6) , BS_put,N, np.absolute(MC_put_price_T-BS_put), error))
    print()
    
    #SIMULATION OF THE PAYOFF AT 6 MONTHS FROM NOW, WITH THE GBM MODEL

    P_0_tM = final_TM[index_tM,1] #Discount factor P(0,tM)

    P_tM_T = np.prod(final_TM[index_tM:index_T+1,3]) #Discount factor P(tM,T)

    time_steps_tM = final_TM[:index_tM+1,4] #Selecting the time steps from 0 to tM (VaR evaluation date). They will be needed for simulating the stochastic processes up to tM

    Asset_Heston_trj_tM = heston(sigma, k, theta, eta, S0, 0, time_steps_tM, rho, N) #Martingale from 0 to tM with the GBM
    
    Heston_put_payoff_tM = strike*P_tM_T - Asset_Heston_trj_tM[-1]/final_TM[index_tM,1] #Computing the put payoff at tM

##    Heston_call_payoff_tM = Asset_Heston_trj_tM[-1]/final_TM[index_tM,1] - strike*P_tM_T #Computing the call payoff at tM
##
##    Heston_call_payoff_tM[Heston_call_payoff_tM < 0] = 0 #Distribution of the call option payoff at time tM

    Heston_put_payoff_tM[Heston_put_payoff_tM < 0] = 0 #Distribution of the put option payoff at time tM

    Heston_put_NU = BS_put - Heston_put_payoff_tM*P_0_tM #Distribution of the profit and loss function

    Heston_q10 = np.round(np.quantile(Heston_put_NU, 1 - .1),6) #quantile at 90%
    Heston_q5 = np.round(np.quantile(Heston_put_NU, 1 - .05),6) #quantile at 95%
    Heston_q1 = np.round(np.quantile(Heston_put_NU, 1 - .01),6) #quantile at 99%

    #PRINTING THE RESULTS

    print()
    print("P(0,"+str(T)+") = "+str(P_0_T))
    print("P(0,"+str(tM)+") = "+str(P_0_tM))
    print("P("+str(tM)+","+str(T)+")"+" = " + str(P_tM_T))
    print()
    
    print("Put option price at time 0: ", str(BS_put))
    print()
    print(layout1.format("Quantile","VaR","Portfolio loss"))
    print(layout1.format("10%",Heston_q10, "{:.4f}%".format((np.round(np.quantile(Heston_put_NU, 1 - .1)/BS_put,  6))*100) ))
    print(layout1.format("5%",Heston_q5,   "{:.4f}%".format((np.round(np.quantile(Heston_put_NU, 1 - .05)/BS_put, 6))*100) ))
    print(layout1.format("1%",Heston_q1,   "{:.4f}%".format((np.round(np.quantile(Heston_put_NU, 1 - .01)/BS_put, 6))*100) ))
    print()
    print()
    print("Moments of the distribution of the loss function")
    print()
    print(layout3.format('Mean','Standard dev','Skewness','Kurtosis'))
    print()
    print(layout3.format(np.round(np.mean(Heston_put_NU),6), np.round(np.std(Heston_put_NU),6), np.round(scipy.stats.skew(Heston_put_NU),6), np.round(scipy.stats.kurtosis(Heston_put_NU),6)))
    print()
    print("Moments of the distribution of the underlying at "+str(tM))
    print()
    print(layout3.format(np.round(np.mean(Asset_Heston_trj_tM[-1]),6), np.round(np.std(Asset_Heston_trj_tM[-1]),6), np.round(scipy.stats.skew(Asset_Heston_trj_tM[-1]),6), np.round(scipy.stats.kurtosis(Asset_Heston_trj_tM[-1]),6)))
    print()

    #PLOTS

    plt.plot(np.hstack((np.array([0]),maturities)), np.std(Asset_Heston_trj_T, axis = 1))
    plt.xlabel("Time")
    plt.ylabel('Standar dev')
    plt.title("Evolution of the standard deviation (Heston)")
    plt.savefig("Evolution_of_the_standard_deviation(Hest).pdf")
    plt.show()

    plt.plot(np.hstack((np.array([0]),maturities)) ,Asset_Heston_trj_T[:,0:20])
    plt.xlabel("Time")
    plt.ylabel("Asset price")
    plt.title("Asset trajectories for the whole life of the contract (Hest)")
    plt.savefig("Asset_trajectories_for_the_whole_life_of_the_contract(Hest).pdf")
    plt.show()

    plt.hist(Asset_Heston_trj_tM[-1], bins = 1000, density = True)
    plt.title("Density of the asset with Heston at tm = "+str(tM))
    plt.savefig("(Hest)Density_of_the_asset_with_Heston_at_tm="+str(tM)+".pdf")
    plt.show()
    
    plt.hist(Heston_put_NU, bins = n_bins, density = True)
    plt.title('Empirical pdf of the PL func (Heston)')
    plt.savefig("Empirical_pdf_of_the_PL_func(Hest).pdf")
    plt.show()

    ecdf_Heston_NU = empirical_cdf(Heston_put_NU) #This is an array with two columns where in the first one there are the values of the rv and in the second the associated prob
    
    plt.plot(ecdf_Heston_NU[:,0],ecdf_Heston_NU[:,1],'b')
    plt.plot(Heston_q10, .9, 'ro')
    plt.plot(Heston_q5, .95, 'go')
    plt.plot(Heston_q1, .99, 'yo')
    plt.legend(["ecdf",'quantile 90%','quantile 95%','quantile 99%'])
    plt.ylabel('P(X < x)')
    plt.xlabel('x')
    plt.title("Empirical probability function of the PL func (Heston)")
    plt.savefig("Empirical_probability_function _of_the_PL_func(Hest).pdf")
    plt.show()

##    plt.plot(np.hstack((np.array([0]),maturities)), S0 + 1.96*np.std(Asset_Heston_trj_T, axis = 1), 'r')
##    plt.plot(np.hstack((np.array([0]),maturities)), S0 - 1.96*np.std(Asset_Heston_trj_T, axis = 1), 'r')
##    plt.plot(np.hstack((np.array([0]),maturities)), martingale, 'b')
##    plt.title("Martingale check with 95% of accuracy")
##    plt.xlabel("Maturity")
##    plt.show()


if __name__ == "__main__":
    run(sys.argv)


    
