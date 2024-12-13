import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

from scipy.signal import convolve

def euler_integration(X, Y, interval):
    """Calculate the integral of two signals X*Y over a given interval."""
    mask = ~np.isnan(X) & ~np.isnan(Y)
    product = X[mask] * Y[mask]
    scale = (interval[-1] - interval[0]) / len(interval)
    return scale * np.sum(product)


def conv(X, Y, interval):
    """Calculate the convolution of two signals X and Y over a given interval."""
    Z = np.zeros_like(X)
    for t in range(1, len(X)):
        #mask = ~np.isnan(X[:t]) & ~np.isnan(Y[:t])
        #product = X[mask] * Y[mask]
        scale = (interval[t] - interval[0]) / len(interval[:t])
        Z[t] = scale * np.sum(X[:t] * Y[:t])
    return Z


def conv_lin_kernel_delta_x_j(DeltaX_stimulated_node, *params):
    """Estimate response based on convolution of linear kernels."""
    resolution = 500
    t_eval = np.linspace(0, 20, resolution)

    N = 3  # do not consider the stimulated neuron
    M = 3  # number of exponentials per kernel
    k = 3  # number of parameters per exponential, C_m, gamma_m0 and gamma_m1

    C_gammas = np.array(params).reshape(N, M, k)

    estimated_x_i_from_convolutions = np.zeros((N, resolution))
    for i in range(N):
        for m in range(M):
            exp_terms = np.exp(-C_gammas[i, m, 1] * (t_eval[:, None] - t_eval[None, :])) * np.exp(-C_gammas[i, m, 2] * (t_eval[:, None] - t_eval[None, :]))
            for t in range(1, resolution):
                estimated_x_i_from_convolutions[i, t] += euler_integration(C_gammas[i, m, 0] * exp_terms[t, :t], DeltaX_stimulated_node[:t], t_eval[:t])
    
    return estimated_x_i_from_convolutions.flatten()

def exp_kernel_func(t, *params):
    # exponentials combinations with two parameters each, amplitude, decay const
    exponential_combination = np.zeros(len(t))
    for p_idx in range(0,len(params),3):
        exponential_combination += params[p_idx]* conv(np.exp(-params[p_idx+1] * t), np.ones(len(t)), t)
    return exponential_combination*np.heaviside(t, 1.0)

def fit_exp_kernel(t, x, y):
    def convolved_model(t, *params):
        # Convolution function: convolves x with the kernel
        resolution = len(t)
        kernel = exp_kernel_func(t, *params)
        convolved_signal = np.zeros(resolution)
        for t_idx in range(1,resolution):
            convolved_signal[t_idx] = euler_integration(x[:t_idx], kernel[(t_idx-1)::-1], t[:t_idx])
        return convolved_signal
        
    initial_guess = np.random.rand(1*3) # be multiple of two, otherwise change func exp_kernel_func
    
    # Fit the parameters a and b to minimize the difference between fit_func and the actual data y
    params_opt, params_cov = curve_fit(convolved_model, t, y, p0=initial_guess)
    
    return params_opt, params_cov

def sum_conv_g_ij_delta_x_j(nonlinear_term, *params):
    """Estimate response based on convolution of nonlinear terms."""
    resolution = 500
    t_eval = np.linspace(0, 20, resolution)
    N = 4

    A = np.reshape(params, (N, N))

    estimated_x_i_from_convolutions = np.zeros((N, resolution))
    for i in range(N):
        exp_terms = np.exp(-(t_eval[:, None] - t_eval[None, :]))
        for t in range(1, resolution):
            A[i, i] = 0.0
            for j in range(N):
                estimated_x_i_from_convolutions[i, t] += euler_integration(A[i, j] * exp_terms[t, :t], nonlinear_term[i * N + j, :t], t_eval[:t])

    return estimated_x_i_from_convolutions.flatten()


def nonlinear_equation_original(t, X, N, A):
    """Nonlinear ODE system without external stimulus."""
    dX = np.zeros_like(X)
    for i in range(N):
        sum_term = np.sum(A[i] * (1 - X[i]) * X) - A[i, i] * (1 - X[i]) * X[i]
        dX[i] = sum_term - X[i]
    return dX


def nonlinear_equation_stimuli(t, X, N, A, stim_neurons, stimuli):
    """Nonlinear ODE system with external stimulus on node 3 between time 2 and 8."""
    dX = np.zeros_like(X)
    for i in range(N):
        sum_term = np.sum(A[i] * (1 - X[i]) * X) - A[i, i] * (1 - X[i]) * X[i]
        
        stim=0.0
        if (i in stim_neurons) and (2 < t < 8):
            stim = stimuli
        else:
            stim = 0.0
            
        dX[i] = sum_term - X[i] + stim
    return dX


if __name__ == "__main__":
    N = 4  # Number of nodes
    A = np.array([[0, 1,   0,   0], 
                  [0, 0,   1,   0], 
                  [0, 0,   0,   1], 
                  [0, 0,   0,   0]])
    X0 = np.random.rand(N)  # Initial random state for nodes
    resolution = 500
    tolerance = 1e-8
    max_iterations = 100
    t_span = (0, 20)
    t_eval = np.linspace(t_span[0], t_span[1], resolution)

    # Solve without stimulus
    for iteration in tqdm(range(max_iterations), desc="Iterating"):
        sol = solve_ivp(lambda t, X: nonlinear_equation_original(t, X, N, A), t_span, X0, t_eval=t_eval)
        change = np.linalg.norm(sol.y[:, -1] - sol.y[:, -2])
        if change < tolerance:
            break
        X0 = sol.y[:, -1]  # Update initial condition
    Xeq = X0  # Final equilibrium state
    
    N_trials   = 8
    DeltaX     = np.zeros((N_trials, N, resolution))
    DeltaX_avr = np.zeros((N, resolution))
    
    # solve trial by trial
    for trial_idx in range(N_trials):
        sol = solve_ivp(lambda t, X: nonlinear_equation_stimuli(t, X, N, A, [3], np.array([0.01+0.2*trial_idx])), t_span, np.zeros(N), t_eval=t_eval)
        DeltaX[trial_idx, :, :] = sol.y + np.random.normal(0, 0.03, (N, resolution))  # eq is zero

    for i in range(N):
        for t in range(resolution):
            DeltaX_avr[i, t] = np.mean(DeltaX[:, i, t])
    """    
    #Plot the time evolution of the system
    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(sol.t, DeltaX[i], label=f'Node {i+1}')
    plt.title('Time Evolution of the System for Each Node')
    plt.xlabel('Time')
    plt.ylabel('Node State')
    plt.legend()
    plt.grid(True)
    #plt.show()
    """

    fitted_curve_linear = np.zeros((N_trials,resolution))
    fitted_lin_params, _ = fit_exp_kernel(t_eval,  DeltaX_avr[3, :], DeltaX_avr[0, :])

    for trial_idx in tqdm(range(N_trials), desc="Iterating (trials)"):
        for t_idx in range(1,len(t_eval)):
            fitted_curve_linear[trial_idx, t_idx] = euler_integration(DeltaX[trial_idx, 3, :t_idx], exp_kernel_func(t_eval[(t_idx-1)::-1], *fitted_lin_params), t_eval[:t_idx])
            
    nonlinear_term = np.zeros((N_trials, N*N, resolution))
    fitted_curve_nonlinear =  np.zeros((N_trials, N*resolution))
    # Initial guess for parameters
    A_guess = np.ones((N, N)) * 0.5
    np.fill_diagonal(A_guess, 0.0)  # Start diagonal elements as 0, expected not self connection
    A_guess = A_guess.flatten()
    for trial_idx in tqdm(range(N_trials), desc="Iterating (trials)"):
        ## Compute nonlinear term # maybe it is more efficient to compute outside the fit function because it would not be computed several times (it does not need parameters)
        for i in range(N):
            for j in range(N):
                nonlinear_term[trial_idx, (i*N+j), :] = (1 - Xeq[i])*(1 - (DeltaX[trial_idx, i, :] / (1 - Xeq[i])))*DeltaX[trial_idx, j, :]

        """  
        fitted_nonlin_params, _ = curve_fit(
            sum_conv_g_ij_delta_x_j,
            nonlinear_term[trial_idx, :, :],
            DeltaX[trial_idx, :, :].flatten(),
            p0=A_guess,
            bounds=(-1, 1),
            method='trf'
        )
        """
        fitted_nonlin_params = A.flatten() # do not fit the parameters, use the known values from the model

        fitted_curve_nonlinear[trial_idx, :] = sum_conv_g_ij_delta_x_j(nonlinear_term[trial_idx, :, :], fitted_nonlin_params)
        
        #A_guess = fitted_nonlin_params
        print(fitted_nonlin_params)
    """
    # Solve for different stimulation
    stim_neurons = [3, 2] #stimulated neurons
    sol = solve_ivp(lambda t, X: nonlinear_equation_stimuli(t, X, N, A, stim_neurons), t_span, Xeq, t_eval=t_eval)
    DeltaX = sol.y + np.random.normal(0, 0.05, (N, resolution)) - np.tile(Xeq.reshape(-1, 1), (1, len(sol.t)))
    #remove node
    #DeltaX[1, :] = np.zeros(resolution) + np.random.normal(0, 0.05, (resolution))
    """
    
    # Plot fitted results
    colors = plt.cm.viridis(np.linspace(0, 1, N_trials))
    plt.figure(figsize=(12, 10))
    for trial_idx in range(N_trials):
        plt.plot(t_eval, fitted_curve_linear[trial_idx, :], label=f'Linear Kernels Fit #{trial_idx}', color='r', linestyle='-')
        plt.scatter(t_eval, DeltaX[trial_idx, 0, :], label='Observed', color=colors[trial_idx], alpha=0.5)
        plt.plot(t_eval, fitted_curve_nonlinear[trial_idx, 0 * resolution:(0 + 1) * resolution], label=f'NEGF Fit trial #{trial_idx}', color='k')
        
    
    plt.ylabel(f'Response Node State')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time')
    plt.show()

    
