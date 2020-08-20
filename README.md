# AdvCompFinanceExam
Analysis of the performance of Value at Risk (VaR) analysis on a long position in put option (plain vanilla, european) for three different stochastic processes: Geometric Brownian Motion (GBM) model, Heston stochastic volatilty model and Variance Gamma (VG) model. Trajectories of stochastic processes are generated using the Monte Carlo approach.
The current price of the option comes from the Black and Scholes (BS) formula.

The solution for the exercise is separated of three different files for different stochastic processes (files "VaR GBM.py", "VaR Heston.py", "VaR VG.py") for greater convenience with working with parameters. There is the same algorithm for every code:
- Set the parameters (either default or given from the command line). In particular, the term structure is imported from the file "ir.py".
- Prepare the right form of term structure by interpolating. Here we use the file "ACF.py" (the file with such a
common functions).
- Based on the assumption that market prices are Black and Scholes prices, we compute the option price.
- Depending on the model, we generate trajectories (up to the maturity of the option T and up to a VaR time tM).
- Compute the price of put option.
- Compute the P&L function and simulate probability density function of it.
- Print all necessary results.

So as the output in our program we have all the numerical results printed, such as a martingale property check (in order to know if the computations can be reliable), results for Monte Carlo simulation (including the absolute error of MC price of a put comparing to the theoretical price and an Error computed using the formula (2.9) from lecture
notes), VaR results with a loss from the position (in %) specified, and parameters of distribution (mean, standard deviation, skewness, kurtosis) of both P&L function and underlying for better understanding. After these numerical results we have histogram (empirical density function) of the P&L function and cumulative probability function plotted. Also there are three files ("GBM checking.py ", "Heston checking.py ", "VG checking.py ") for testing the martingale
property and reliability of the Monte Carlo simulat
