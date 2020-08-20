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

def usage():
    print("Usage: $> python3 ESAME [options]")
    print("Options parameters:")
    print("    %-24s: input_file: the input file holding interest rates")
    print("    %-24s: this output" %("--help"))
    print("    %-24s: r: interest rate" %("-r"))
    print("    %-24s: initial value of the underlying (1 by default)" %("-S0"))
    print("    %-24s: option strike (1.03 by default)" %("-strike"))
    print("    %-24s: option maturity (1.13 by default)" %("-T"))
    print("    %-24s: GBM volatility (sigma square is 0.24 by default)" %("-s"))
    print("    %-24s: Length of the time steps (1/12 by default)" %("-dt"))
    print("    %-24s: Number of simulations (2^n n = 20 by default)" %("-N"))
    print("    %-24s: seed: fixing the seed (default 100)" %("-seed"))


def run(args):
    #Default values
    
    N  = 22  #Simulation of BS
    seed = 100

    np.random.seed(seed)
    
    #GBM and Black and Sholes parameters
    sigma = .24 
    delta_t = 1/12
    r = 0
    #Option data
    tM = .5
    T = 1.13 #Option maturity
    tM = np.round(tM,6)
    T = np.round(T,6)
    strike = 1.03
    S0 = 1

    #########THE CHECKING PART############

    parms = get_input_parms(args) #Per prendere i parametri da linea di comando


    try:
        op = parms["help"]
        usage()
        return
    except KeyError:
        pass
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
        seed = int(parms["seed"])
    except KeyError:
        pass
    try:
        r = float(parms["r"])
    except KeyError:
        pass

    layout1 = "{0:>15}{1:>15}{2:>15}"
    layout2 = "{0:>15}{1:>15}"
    layout3 = "{0:>24}{1:>24}{2:>24}{3:>24}"
    layout4 = "{0:>24}{1:>24}{2:>24}{3:>24}{4:>24}"

    N = 2**N
    np.random.seed(seed)

    n = int(T/delta_t)  # Number of periods

    maturities = np.cumsum(np.ones(n)*delta_t) #Array with the maturities

    if T not in maturities:
        maturities = np.hstack((maturities, np.array([T])))
        maturities = np.sort(maturities)

    steps = np.ones(n)*delta_t #Time intervals

    P_0_T = ACF.continuous_spot_discount_factor(r,T) #The discount factor

    GBM_trj = ACF.GBM(S0, 0, sigma, steps,N) #Underlying trajectories

    martingale = np.mean(GBM_trj, axis = 1) #Martingale check for the underlying

    put_payoff = strike*P_0_T - GBM_trj[-1]
    put_payoff[put_payoff < 0] = 0
    put_price = np.mean(put_payoff) #MC put price

    call_payoff = GBM_trj[-1] - strike*P_0_T
    call_payoff[call_payoff < 0] = 0
    call_price = np.mean(call_payoff) #MC call price

    MC_Asset_error = 3 * np.sqrt((np.mean(GBM_trj ** 2, axis = 1) - martingale ** 2)/N) #MC error of the underlying

    MC_put_error = 3 * np.sqrt( (np.mean(put_payoff ** 2) - put_price ** 2) / N) #MC error of the put option

    MC_call_error = 3 * np.sqrt( (np.mean(call_payoff ** 2) - call_price ** 2) / N) #MC error of the put option

    #Theoretical option prices

    theo_put = ACF.BS_EuropeanPut(S0, strike, T, r, sigma) #Theoretical put price with BS
    theo_call = ACF.BS_EuropeanCall(S0, strike, T, r, sigma) #Theoretical call price with BS

    #Printing part

    print("{0:>20}{1:>20}".format("Interest rate","Discount factor"))
    print("{0:>20}{1:>20}".format(r ,np.round(P_0_T, 6)))
    print()
    print("Number of simulations: "+str(N))
    print()
    print("Maringale check for the underlying")
    print()
    print(layout3.format("Maturity","Mean","Absolute error","MC error"))
    print()
    for i in range(0,len(maturities)):
        print(layout3.format( np.round( maturities[i], 6), np.round( martingale[i], 6), np.absolute(np.round(S0-martingale[i], 6)), np.round(MC_Asset_error[i], 9) ))

    print()
    print("Put option")
    print()
    print(layout4.format("Maturity", "Theoretical price", "MC price","Absolute error", "MC error" ) )
    print(layout4.format(T , np.round(theo_put , 6), np.round(put_price , 6),np.round(np.absolute(theo_put - put_price) ,6) ,np.round(MC_call_error , 6) ))
    print()

    print()
    print("Call option")
    print()
    print(layout4.format("Maturity", "Theoretical price", "MC price", "Absolute error", "MC error" ) )
    print(layout4.format(T , np.round(theo_call , 6), np.round(call_price , 6),np.round(np.absolute(theo_call - call_price) ,6) ,np.round(MC_call_error , 6) ))
    print()

if __name__ == "__main__":
    run(sys.argv)
