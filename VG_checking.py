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
    print("    %-24s: this output" %("--help"))
    print("    %-24s: initial value of the underlying (1 by default)" %("-S0"))
    print("    %-24s: option strike (1.03 by default)" %("-strike"))
    print("    %-24s: option maturity (1 by default)" %("-T"))
    print("    %-24s: r constan interest rate (0 by default)" %("-r"))
    print("    %-24s: Length of the time steps (1/12 by default)" %("-dt"))
    print("    %-24s: NUmber of simulations (2^n n = 21 by default)" %("-N"))
    print("    %-24s: eta: the eta of the Variance Gamma model (0.1664 by default)" %("-eta"))
    print("    %-24s: theta: the theta of the Variance Gamma model (-0.7678 by default)" %("-theta"))
    print("    %-24s: nu: the nu of the Variance Gamma model (0.0622 by default)" %("-nu"))
    print("    %-24s: seed: fixing the seed (default 100)" %("-seed"))


def run(args):
    #Default values
    
    N  = 21

    seed = 100

    np.random.seed(seed)

    #Variance Gamma parameters

    eta = 0.1664
    theta =  -.7678
    nu = 0.0622

    r = 0

    #Option data
    T = 1 #Option maturity
    strike = 1.03
    S0 = 1
    delta_t = 1/12

    #######THE CHECKING PART########

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
        eta = float(parms["eta"])
    except KeyError:
        pass
    try:
        theta = float(parms["theta"])
    except KeyError:
        pass
    try:
        nu = float(parms["nu"])
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

    np.random.seed(seed)

    N = 2**N #Number of MC simulations

    n = int(T/delta_t)  # Number of periods

    maturities = np.cumsum(np.ones(n)*delta_t) #Array with the maturities

    if T not in maturities:
        maturities = np.hstack((maturities, np.array([T])))
        maturities = np.sort(maturities)

    steps = np.ones(n)*delta_t #Time intervals

    P_0_T = ACF.continuous_spot_discount_factor(r,T) #The discount factor

    VG_trj = ACF.VarianceGamma_process(S0, eta, theta, nu, steps, N) #Underlying trajectories

    martingale = np.mean(VG_trj, axis = 1) #Martingale check for the underlying

    put_payoff = strike*P_0_T - VG_trj[-1]
    put_payoff[put_payoff < 0] = 0
    put_price = np.mean(put_payoff) #MC put price

    call_payoff = VG_trj[-1] - strike*P_0_T
    call_payoff[call_payoff < 0] = 0
    call_price = np.mean(call_payoff) #MC call price

    MC_Asset_error = 3 * np.sqrt((np.mean(VG_trj ** 2, axis = 1) - martingale ** 2)/N) #MC error of the underlying

    MC_put_error = 3 * np.sqrt( (np.mean(put_payoff ** 2) - put_price ** 2) / N) #MC error of the put option

    MC_call_error = 3 * np.sqrt( (np.mean(call_payoff ** 2) - call_price ** 2) / N) #MC error of the put option

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
    print(layout1.format("Maturity", "Price", "MC error" ) )
    print(layout1.format(T , np.round(put_price , 6), np.round(MC_put_error , 6) ))
    print()

    print()
    print("Call option")
    print()
    print(layout1.format("Maturity", "Price", "MC error" ) )
    print(layout1.format(T , np.round(call_price , 6), np.round(MC_call_error , 6) ))
    print()
    

    
if __name__ == "__main__":
    run(sys.argv)

    

    
