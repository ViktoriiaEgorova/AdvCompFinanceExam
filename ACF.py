import numpy as np
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt
import scipy.stats as si

def continuous_spot_discount_factor(r,t):
    return np.exp(-t*r)

def continuous_spot_rate(t,P):
    return (-1/t)*np.log(P)

def simple_spot_rate(c_rate):
    return np.exp(c_rate)-1
    

def Correlated_MultiNormal(mu, cov_m, N=1): #Generating correlated multivariate Gaussian random variables
    """mu = mean column array, cov_m = variance-covariance matrix\nN = number of realization for each component of the random vector"""
    P = np.linalg.cholesky(cov_m) #Cholesky decomposition of the variance covarinace matrix
    std_normal = np.random.normal(0,1,(len(mu),N))
    values = mu + np.linalg.multi_dot((P,std_normal))
    return values #return an array in which each row is a component of the random vector and the columns are the realization

def GBM(S0, mu, sigma, dt, N):
    """ALL THE INPUTS MUST BE AN ARRAY\ndt=array of the steps, n=number of paths"""
    St0 = np.ones(N)*S0
    final = np.empty((len(dt),N))
    final = np.vstack((St0,final))
    for i,delta in enumerate(dt):
        Stn = final[i]*np.exp((mu-(sigma**2)/2)*delta+sigma*np.sqrt(delta)*np.random.normal(0,1,(N)))
        final[i+1] = Stn
    return final #Each column is a different path

def heston(sigma_square, k, theta, eta, S0, r, delta_t, rho, N):
    S = np.ones((len(delta_t)+1,1)) #Where are stored the values of the asset

    n1 = int(np.round(np.sqrt(N)))

    n2 = int(np.round(np.sqrt(N)))

    for _ in range(0,n1):

        nu = np.hstack((np.array([sigma_square]),np.empty(len(delta_t))))

        temp_S = np.vstack((np.ones(n2)*S0,np.empty((len(delta_t),n2))))

        for j,dt in enumerate(delta_t):

            flag = nu[j] + k*(theta-nu[j])*dt + eta*np.sqrt(nu[j]*dt)*np.random.normal(0,1,1)

            nu[j+1] = max(0,flag)
            
            temp_S[j+1] = temp_S[j]*np.exp((r-nu[j]/2)*dt+np.sqrt(nu[j]*dt)*(rho*np.random.normal(0,1,n2) + np.sqrt(1 - rho**2)*np.random.normal(0,1,n2)))
            
        S = np.hstack((S,temp_S))
    return S[:,1:]


def VarianceGamma_process(S0, eta, theta, nu, delta_t, N):
    """delta_t = array of time steps N = number o simulations"""
    S = np.vstack((np.ones(N)*S0,np.empty((len(delta_t),N))))
    for i,dt in enumerate(delta_t):
        gamma = np.random.gamma(dt/nu, nu, N)
        Z = theta * gamma + eta * np.sqrt(gamma) * np.random.normal(0,1,N)
        S[i+1] = S[i] * np.exp((dt/nu) * np.log(1 - nu * theta - (nu * eta ** 2) * .5) + Z)
    return S
 
def rate_finder(t, TS):
    """TS must be an array"""
    dates = list(TS[:,0].flatten())
    if t in dates:
        return dates.index(t)
    else:
        return -1

def forward_zcb(TS):
    v = np.array([TS[0]])
    f1 = TS[:-1]
    f2 = TS[1:]
    forward = f2/f1
    forward = np.reshape(forward, (len(forward),1))
    forward = np.vstack((v,forward))
    return forward

def TIME_STEPS(maturities):
    t1 = maturities[:-1]
    t2 = maturities[1:]
    deltas = t2 - t1
    return np.reshape(deltas, (len(deltas) , 1))


def preparingStructure(curve, delta_t, T, tM):

    term_structure = []  # In the first column there are the maturities in year fraction, in the second there are the ZCB prices and in the third one there are the interest rates
    for item in curve:
        v1 = np.round(item[0], 7)
        v2 = continuous_spot_discount_factor(item[1], item[0])
        v3 = item[1]
        term_structure.append([v1, v2, v3])
    term_structure = np.array(term_structure)

    # PREPARING THE TERM STRUCTURE

    n = int(np.round(term_structure[-1, 0] / delta_t))
    ZCB_mties = np.round(np.cumsum(np.ones(n) * delta_t), 6)  # ZCB maturities
    ZCB_mties = np.reshape(ZCB_mties, (len(ZCB_mties), 1))
    DT = []
    for i, v in enumerate(ZCB_mties):
        DT.append([float(v), np.nan])
    DT = np.array(DT)

    for i, t in enumerate(term_structure[:, 0]):
        if t in ZCB_mties:
            index = rate_finder(t, ZCB_mties)
            DT[index, 1] = np.log(term_structure[i, 1])
        else:
            DT = np.vstack((DT, np.array([t, np.log(term_structure[i, 1])])))
    DT = np.vstack((np.array([0, 0]), DT))
    if T not in DT:
        DT = np.vstack((DT, np.array([np.round(T, 6), np.nan])))
    if tM not in DT:
        DT = np.vstack((DT, np.array([np.round(tM, 6), np.nan])))
    DT = list(DT)
    DT = sorted(DT, key=itemgetter(0))
    DT = np.array(DT)
    df = pd.DataFrame(DT[:, 1], index=DT[:, 0])
    df = df.interpolate()  # So I"ve got the interpolated values of the log of the ZCB

    ts1 = np.reshape(np.array(df.index), (len(np.array(df.index)), 1))
    ts2 = np.reshape(np.array(np.array(df[0])), (len(np.array(df.index)), 1))

    final_TM = np.hstack((ts1, ts2))
    final_TM[:, 1] = np.round(np.exp(final_TM[:, 1]), 6)  # <-- The complete interpolated term structure

    rates = np.round(-np.log(final_TM[1:, 1]) / final_TM[1:, 0], 6)
    rates = np.hstack((np.array([0]), rates))
    final_TM = np.hstack((final_TM, np.reshape(rates, (len(rates), 1))))
    final_TM = final_TM[1:, :]
    final_TM[:, 0] = np.round(final_TM[:, 0], 6)
    forward_discount = forward_zcb(final_TM[:, 1])
    final_TM = np.hstack((final_TM,
                          forward_discount))  # I've added a new column with the forward prices of the zcb whose maturity is a time step
    deltas = TIME_STEPS(np.hstack((np.array([0]), final_TM[:, 0])))
    final_TM = np.hstack((final_TM, deltas))  # I've added a new column with the all the time steps from a time to the folliwing one
    return final_TM


def BS_EuropeanCall(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))

    return call

def BS_EuropeanPut(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))

    return put




def put_call_price(final_TM, T, tM, S0, strike, sigma):
    # Index corresponding to the maturity T
    index_T = rate_finder(T, final_TM)
    # Index corresponding to tM (when you want to evaluate the position)
    index_tM = rate_finder(tM, final_TM)
    # Put option price computed with the BS pricing formula
    BS_put = BS_EuropeanPut(S0, strike, final_TM[index_T, 0],final_TM[index_T, 2], sigma)
    # Call option price computed with the BS pricing formula
    BS_Call = BS_EuropeanCall(S0, strike, final_TM[index_T, 0], final_TM[index_T, 2], sigma)
    P_0_T = final_TM[index_T, 1]
    return index_T, index_tM, BS_put, BS_Call, P_0_T




#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------

##S0 = 2.7
##N = 3
##delta_t = np.ones(11)*(1/12)
##mu = np.array([0])
##sigma = np.array([1])
##jmu = np.array([0])
##jsigma = np.array([1])
##Lambda = 3
##
##
##
##y1 = Poisson_process(2,np.ones(30)*0.5,N=1)
##y2 = Poisson_process(3,np.ones(30)*0.5,N=1)
##y3 = Poisson_process(3.5,np.ones(30)*0.5,N=1)
##plt.step(np.arange(0,len(y1)) ,y1)
##plt.step(np.arange(0,len(y2)) ,y2)
##plt.step(np.arange(0,len(y3)) ,y3)
##plt.show()
