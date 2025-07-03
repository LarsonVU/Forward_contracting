import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_variance(v, covariance_matrix):
    v = np.array(v)
    return np.dot(v, np.dot(covariance_matrix, v.T))

def compute_expectation(v, E_Y):
    return np.dot(v, E_Y)

def compute_lambda_mu(E_Y, covariance_matrix, beta, m):
    sigma_inv = np.linalg.inv(covariance_matrix)
    A = np.vstack([E_Y, beta])
    A_sigma_inv = A @ sigma_inv @ A.T
    rhs = np.array([m, 1])
    lambda_mu = 2 * np.linalg.inv(A_sigma_inv) @ rhs
    return lambda_mu

def compute_mu(covariance_matrix, beta):
    beta = np.array(beta)
    sigma_inv = np.linalg.inv(covariance_matrix)
    denominator = beta @ sigma_inv @ beta.T
    mu = -2 / denominator
    return mu

def compute_lambda_mu_given_v(v, E_Y, covariance_matrix, beta):
    # Compute g
    # Filter indices where v > 1e-8 and E_Y != 0
    positive_indices = np.where((v > 1e-8) & (E_Y != 0))[0]

    g = (2 * v @ covariance_matrix)[positive_indices]

    A = [E_Y[positive_indices], -beta[positive_indices]]
    A = np.vstack(A).T  # shape (m, 2)

    # Solve least squares
    x, residuals, rank, s = np.linalg.lstsq(A, g, rcond=None)

    lambda_mu = x
    #print(f"Lambda: {lambda_mu[0]}, Mu: {lambda_mu[1]}")
    #print(f"Lambda_nonegative {lambda_mu[2:]}")
    return lambda_mu

def compute_v_unconstrained(covariance_matrix, beta):
    sigma_inv = np.linalg.inv(covariance_matrix)
    numerator = np.dot(beta, sigma_inv)
    denominator = np.dot(np.dot(beta, sigma_inv), np.transpose(beta))
    v = numerator / denominator
    return v

def compute_v_constrained(E_Y, covariance_matrix, beta, lambda_mu):
    sigma_inv = np.linalg.inv(covariance_matrix)
    lambda_, mu = lambda_mu
    # Form the vector [lambda, mu]
    lm_vec = np.array([lambda_, mu])
    # Form the matrix [E[Z], beta]
    ez_beta = np.vstack([E_Y, beta])
    # Compute w
    v = 0.5 * (lm_vec @ ez_beta) @ sigma_inv
    return v

def compute_given_active_constraints(E_Y, covariance_matrix, active_set, m):
    total_components = len(E_Y)
    number_b_components = total_components // 2

    beta = []
    E_Y_prime = E_Y
    Cov_prime = covariance_matrix

    # Build beta
    beta = []
    for i in range(total_components):
        if i not in active_set:
            if i < number_b_components:
                beta.append(1)
            else:
                beta.append(0)

    # Delete from E_Y_prime and Cov_prime
    to_delete = sorted([i for i in range(total_components) if i in active_set], reverse=True)
    for i in to_delete:
        E_Y_prime = np.delete(E_Y_prime, i)
        Cov_prime = np.delete(Cov_prime, i, axis=0)
        Cov_prime = np.delete(Cov_prime, i, axis=1)

    C1 = beta
    c1 = np.ones(1)
    C1_rank = np.linalg.matrix_rank(C1)
    C1c1 = np.hstack([C1, c1])
    C1c1_rank = np.linalg.matrix_rank(C1c1)

    if C1_rank != C1c1_rank:
        #print("Infeasible: Constraints cannot be satisfied (rank deficiency).")
        return None, None, None

    v_unconstrained = compute_v_unconstrained(Cov_prime, beta)

    expected_value = compute_expectation(v_unconstrained, E_Y_prime)
    if expected_value > m:
        v = np.zeros(total_components)
        inactive_indices = [i for i in range(total_components) if i not in active_set]
        v[inactive_indices] = v_unconstrained
        variance = compute_variance(v, covariance_matrix)
        return v, expected_value, variance
    else:
        # Rank check for feasibility
        C2 = np.vstack([E_Y_prime, beta])
        c2 = np.array([m, 1])
        C2_rank = np.linalg.matrix_rank(C2)
        C2c2 = np.hstack([C2, c2.reshape(-1, 1)])
        C2c2_rank = np.linalg.matrix_rank(C2c2)

        if C2_rank != C2c2_rank:
            #print("Infeasible: Constraints cannot be satisfied (rank deficiency).")
            return None, None, None

        v_constrained = compute_v_constrained(E_Y_prime, Cov_prime, beta, compute_lambda_mu(E_Y_prime, Cov_prime, beta, m))
        v = np.zeros(total_components)
        inactive_indices = [i for i in range(total_components) if i not in active_set]
        v[inactive_indices] = v_constrained
        return v, compute_expectation(v, E_Y), compute_variance(v, covariance_matrix)
            
def compute_active_set(covariance_matrix, E_Y, m, verbose=True):
    # Start with no active constraints:
    beta = []
    total_components = len(E_Y)
    number_b_components = total_components // 2
    beta = np.zeros(total_components)
    beta[:number_b_components] = 1  # Set the first half to 1 (for yield constraint)
    
    active_set = set() #set(i for i in range(total_components) if v_start[i] == 0)  # Start with the first forward contract active
    iter =0
    while iter<20:
        iter+=1
        if verbose == True:
            shown_indices = [x + 1 for x in active_set]
            print(f"Current active set: {shown_indices}")

        v, expected_value, variance = compute_given_active_constraints(E_Y, covariance_matrix, active_set, m)
        if v is None:
            if verbose == True:
                print(f"Active set {active_set} does not satisfy expectation condition {m}.")
            return None, None, None, None, None
        lambda_mu = compute_lambda_mu_given_v(v,E_Y, covariance_matrix, beta)

        if verbose == True:
            print(f"Lambda mu: {lambda_mu}")

        lambdas = (2 * covariance_matrix @ v.T - lambda_mu[0] * E_Y + lambda_mu[1] * beta)

        if verbose == True:
            print(f"Lambdas: {np.round(lambdas, 2)}")
            print(f"Current values: {np.round(v, 2)}")
            print(f"Expected value: {expected_value}")

        if any(v < 0):
            candidates = [idx for idx in range(total_components) if v[idx] < 0]
            active_set.add(min(candidates, key=lambda idx: v[idx]))  # Add the component with the highest value
            if verbose == True:
                    print(f"Adding component {min(candidates, key=lambda idx: v[idx])+1} to active set, with {v[min(candidates, key=lambda idx: v[idx])]}.")
            
        else:
            # Remove the last added component from the active set
            if any(lambdas[[idx for idx in active_set]]< 0):
                candidates = [idx for idx in active_set if lambdas[idx] < 0]
                active_set.remove(min(candidates, key=lambda idx:  lambdas[idx]))
                if verbose == True:
                    print(f"Removing component {min(candidates, key=lambda idx: lambdas[idx])+1} from active set, with {lambdas[min(candidates, key=lambda idx: lambdas[idx])]}.")
            else:
                break
    
    if verbose == True:
        shown_indices = [x + 1 for x in active_set]
        print(f"Final active set: {shown_indices}")
        print(f"Final expected value: {expected_value}")
        print(f"Final variance: {variance}")
        print(f"Final v: {np.round(v, 2)}")
        print(f"Final lambda mu: {lambda_mu}")
    return active_set, v, expected_value, variance, lambda_mu

def compute_frontier(Range, E_Y, covariance_matrix):
    """
    Compute the Pareto frontier for a given set of parameters.
    """
    v_values = []
    variances = []
    expectations = []

    for i in range(1, Range):
        M_value = i
        active_set, v, exp, var, lambda_mu = compute_active_set(covariance_matrix, E_Y,  M_value, verbose=False)
        v_values.append(v)
        variances.append(var)
        expectations.append(exp)

    return v_values, variances, expectations

def plot_frontier(expectations_list, variances_list, labels=None, folder="pareto_frontiers"):
    """
    Plot one or more Pareto frontiers (variance vs expectation).
    If only one valid (non-None) point is present, plot it as a dot.

    Args:
        expectations_list (list of lists): Each element is a list/array of expectations for a frontier.
        variances_list (list of lists): Each element is a list/array of variances for a frontier.
        labels (list of str, optional): Labels for each frontier.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    if not isinstance(expectations_list[0], (list, tuple, pd.Series, np.ndarray)):
        # Single frontier, wrap in list
        expectations_list = [expectations_list]
        variances_list = [variances_list]

    for idx, (expectations, variances) in enumerate(zip(expectations_list, variances_list)):
        # Remove None values
        valid_points = [(e, v) for e, v in zip(expectations, variances) if e is not None and v is not None]
        label = labels[idx] if labels and idx < len(labels) else f"Frontier {idx+1}"
        real_points = {
            tup for tup in set(valid_points)
            if not any(np.isnan(x) for x in tup)
        }
        if len(real_points) == 1:
            e, v = valid_points[0]
            ax1.plot(e, v, 'o', label=label)
        elif len(valid_points) > 1:
            e_arr, v_arr = zip(*valid_points)
            ax1.plot(e_arr, v_arr, label=label)

    ax1.set_xlabel("Expectation ($)")
    ax1.set_ylabel(r"Variance ($\$^2$)")
    ax1.tick_params(axis='y')
    plt.title("Pareto Frontiers")
    ax1.legend()
    plt.grid()
    fig.tight_layout()
    plt.savefig(f"{folder}PF_mp.png", dpi=300)
    plt.show()


def plot_percentage_planted(v_values, num_crops):
    """
    Plot the percentage of planted crops.
    """
    for i in range(len(num_crops)):
        plt.plot(range(len(v_values)), [w[i] for w in v_values], label=f"Crop {i+1} Planted")
    plt.xlabel("Index")
    plt.ylabel("Percentage Planted")
    plt.title("Percentage of Planted Crops")
    plt.legend()
    plt.grid()
    plt.show()

def plot_forward_contracts(v_values, num_crops):
    """
    Plot the forward contracts.
    """
    percentage_contracts = [v[num_crops + i] / (v[i] *E_Y[i] ) for w in v_values for i in range(num_crops)]
    for i in range(num_crops):
        plt.plot(range(len(v_values)), percentage_contracts, label=f"Forward Contract {i+1}")
    plt.xlabel("Index")
    plt.ylabel("Forward Contract Value")
    plt.title("Forward Contracts")
    plt.legend()
    plt.grid()
    plt.show()

def plot_planted_and_forward_side_by_side(expectations, v_values, num_crops, means , folder="Figures/case1", product_types=['Improved', 'Hybrid', 'Local']):
    """
    Plot percentage planted and forward contracts side by side.
    """
    x =  [expectations[j] for j, v in enumerate(v_values) if v is not None]

    filtered_v_values = [v for v in v_values if v is not None]

    # Prepare planted percentages and forward contract percentages
    planted_percentages = [
        [v[i] for v in filtered_v_values]
        for i in range(num_crops)
    ]

    percentage_contracts = [
        [
            v[num_crops + i] / (means[i]) 
            for v in v_values if v is not None
        ]
        for i in range(num_crops)
        ]
    
    # Find the earliest deviation index across all crops for planted percentages
    plot_start = float('inf')
    for q_values in planted_percentages:
        q_values = np.array(q_values)
        initial_value = q_values[0]
        deviation_idx = np.argmax(q_values != initial_value)
        if np.all(q_values == initial_value):
            this_start = 0
        else:
            this_start = max(0, deviation_idx - 200)
        if plot_start > this_start:
            plot_start = this_start

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot percentage planted
    for i in range(num_crops):
        ax1.plot(
            x[plot_start:],
            np.array(planted_percentages[i][plot_start:]),
            label=f"Crop {product_types[i]} Planted"
        )
    ax1.set_xlabel("Required Expectation ($)")
    ax1.set_ylabel("Percentage Planted (%)")
    ax1.set_title(r"Percentage of land use per crop")
    ax1.legend()
    ax1.grid()

    # Plot forward contracts
    for i in range(num_crops):
        ax2.plot(
            x[plot_start:],
            np.array(percentage_contracts[i][plot_start:]),
            label=f"Forward Contract {product_types[i]}"
        )
    ax2.set_xlabel("Required Expectation ($)")
    ax2.set_ylabel(r"Forward Contract Size $q_i^*$ (% of $\mathbb{E}[Q_i]$)")
    ax2.set_title("Forward Contract sizes")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.savefig(f"{folder}q_mp.png", dpi=300)
    plt.show()

def collect_crop_statistics(yield_csv="crop_yield.csv", cov_csv="covariance_matrix_P_Q_neg_PQ.csv"):
    yields = pd.read_csv(yield_csv, low_memory=False)
    covariance_matrix = pd.read_csv(cov_csv, low_memory=False).drop(columns=['Unnamed: 0'])
    covariance_matrix = np.array(covariance_matrix)

    # Preprocess data
    yields = yields[(yields['M-q706_cropSP'] < 10000) & (yields['M-q706_cropSP'] > 0)].copy()  # Remove outliers
    yields = yields[yields['C-q306_cropLarestAreaAcre'] > 0]
    acres = yields['C-q306_cropLarestAreaAcre'].median()
    hectares = acres / 2.47105  # Convert acres to hectares
    yields['median_farm_yield_quintale'] = yields['L-tonPerHectare'] * 10  # Convert to quintale per hectare
    rupy_to_dollar = 0.012
    yields['dollar_spot_price'] = yields['M-q706_cropSP'] * rupy_to_dollar  # Convert to dollar
    yields['price_quantity'] = yields['dollar_spot_price'] * yields['median_farm_yield_quintale']  # Price times quantity

    yields_improved = yields[yields['D-q409_varType'] == 'Improved']
    yields_hybrid = yields[yields['D-q409_varType'] == 'Hybrid']
    yields_local = yields[yields['D-q409_varType'] == 'Local']

    mean_price_improved = yields_improved['dollar_spot_price'].mean()
    mean_price_hybrid = yields_hybrid['dollar_spot_price'].mean()
    mean_price_local = yields_local['dollar_spot_price'].mean()

    mean_yield_improved = yields_improved['median_farm_yield_quintale'].mean()
    mean_yield_hybrid = yields_hybrid['median_farm_yield_quintale'].mean()
    mean_yield_local = yields_local['median_farm_yield_quintale'].mean()

    mean_Q = [mean_yield_improved, mean_yield_hybrid, mean_yield_local]

    mean_prices = [mean_price_improved, mean_price_hybrid, mean_price_local]
    mean_Q = [mean_yield_improved, mean_yield_hybrid, mean_yield_local]

    PQ_improved = yields_improved['price_quantity'].mean()
    PQ_hybrid = yields_hybrid['price_quantity'].mean()
    PQ_local = yields_local['price_quantity'].mean()

    PQ_values = [PQ_improved, PQ_hybrid, PQ_local]

    return {
        "covariance_matrix": covariance_matrix,
        "mean_prices": mean_prices,
        "mean_Q": mean_Q,
        "PQ_values": PQ_values,
    }

def compute_E_Y(mean_prices, PQ_values, price_multipliers):
    mean_fP = [mean_prices[i] * price_multipliers[i] - mean_prices[i] for i in range(len(mean_prices))]
    mean_PQ = list(PQ_values) + mean_fP
    return np.array(mean_PQ)

if __name__ == "__main__":
    stats = collect_crop_statistics()
    covariance_matrix = stats["covariance_matrix"]
    mean_prices = stats["mean_prices"]
    mean_Q = stats["mean_Q"]
    PQ_values = stats["PQ_values"]


    # Example usage for 1.1 E[P]
    price_multipliers = [1.1, 1.1, 1.1]
    E_Y = compute_E_Y(mean_prices, PQ_values, price_multipliers)
    active_set, v, expected_value, variance, lambda_mu = compute_active_set(covariance_matrix, E_Y, 3000)
    print()

    v_values1, variances1, expectations1 = compute_frontier(2000, E_Y, covariance_matrix)
    plot_planted_and_forward_side_by_side([i for i in range(2000)], v_values1, len(mean_Q), mean_Q, folder="Figures/case1")

    # Example usage for 0.9 E[P]
    price_multipliers = [0.9, 0.9, 0.9]
    E_Y = compute_E_Y(mean_prices, PQ_values, price_multipliers)
    active_set, v, expected_value, variance, lambda_mu = compute_active_set(covariance_matrix, E_Y, 3000)
    print()

    v_values2, variances2, expectations2 = compute_frontier(2000, E_Y, covariance_matrix)
    plot_planted_and_forward_side_by_side([i for i in range(2000)], v_values2, len(mean_Q), mean_Q, folder="Figures/case2")

    # Example usage for different E[P]
    price_multipliers = [1.1, 1.0, 0.9]
    E_Y = compute_E_Y(mean_prices, PQ_values, price_multipliers)
    active_set, v, expected_value, variance, lambda_mu = compute_active_set(covariance_matrix, E_Y, 3000)
    print()

    v_values3, variances3, expectations3 = compute_frontier(2000, E_Y, covariance_matrix)
    plot_planted_and_forward_side_by_side([i for i in range(2000)], v_values3, len(mean_Q), mean_Q, folder="Figures/case3")
    
    expectations = [expectations1, expectations2, expectations3]
    variances = [variances1, variances2, variances3]
    labels = [r'$f=1.1 E[P]$', r'$f=0.9 E[P]$', r'$f =[1.1, 1.0, 0.9] E[P]$']
    plot_frontier(expectations, variances, folder="Figures/combined", labels=labels)

    # Example not used in the paper
    price_multipliers = [1, 1, 1]
    E_Y = compute_E_Y(mean_prices, PQ_values, price_multipliers)
    active_set, v, expected_value, variance, lambda_mu = compute_active_set(covariance_matrix, E_Y, 1100)
    print()

    v_values4, variances4, expectations4 = compute_frontier(2000, E_Y, covariance_matrix)
    plot_planted_and_forward_side_by_side([i for i in range(2000)], v_values4, len(mean_Q), mean_Q, folder="Figures/case4")
