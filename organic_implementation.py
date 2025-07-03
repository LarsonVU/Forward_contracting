import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def compute_variance(v, covariance_matrix):
    v = np.array(v)
    return np.dot(v, np.dot(covariance_matrix, v.T))

def compute_expectation(v, E_Y):
    return np.dot(v, E_Y)

def compute_langrangian(covariance_matrix, C_i, c_i):
    sigma_inv = np.linalg.inv(covariance_matrix)
    lambda_ = -2 * c_i @ np.linalg.inv(C_i @ sigma_inv @ C_i.T)
    return lambda_

def compute_v_given_C(covariance_matrix, C_i, c_i):
    sigma_inv = np.linalg.inv(covariance_matrix)
    lambda_ = compute_langrangian(covariance_matrix, C_i, c_i)
    v= -0.5 * lambda_ @ C_i @ sigma_inv
    v = np.array(v).flatten()
    return v, lambda_


def compute_given_active_constraints(E_Y, covariance_matrix, active_set_n, active_set_r, m, r_values, verbose =False):
    total_components = len(E_Y)
    number_b_components = total_components // 2

    beta = [1 if i < number_b_components else 0 for i in range(total_components)]
    # Construct C1 and C2 using vstack

    # Build R' matrix for active set A_r
    n = number_b_components
    A_r = list(active_set_r) if active_set_r else []
    R = np.zeros((len(A_r),2 * n))
    for idx, i in enumerate(A_r):
        R[idx, i] = -r_values[i]
        R[idx, n+i] = 1

    # Build N matrix for active set A_n
    n = number_b_components
    A_n = list(active_set_n) if active_set_n else []
    N = np.zeros((len(A_n), 2 * n))
    for col, i in enumerate(A_n):
        N[col, i] = -1
    
    C1 = np.vstack([R, N, beta])
    C2 = np.vstack([R, N, beta, E_Y])

    c1 = np.zeros(C1.shape[0])
    c1[-1] =1
    c2 = np.append(c1,m)

    # Rank check for feasibility
    C1_rank = np.linalg.matrix_rank(C1)
    C1c1 = np.hstack([C1, c1.reshape(-1, 1)])
    C1c1_rank = np.linalg.matrix_rank(C1c1)

    if C1_rank != C1c1_rank:
        if verbose:
            print("Infeasible: Constraints cannot be satisfied (rank deficiency).")
        return None, None, None, None
    else:
        v, lambda_ = compute_v_given_C(covariance_matrix, C1, c1)

    expected_value = compute_expectation(v, E_Y)
    if compute_expectation(v, E_Y) > m:
        variance = compute_variance(v, covariance_matrix)
        return v, expected_value, variance, lambda_
    else:
        C2_rank = np.linalg.matrix_rank(C2)
        C2c2 = np.hstack([C2, c2.reshape(-1, 1)])
        C2c2_rank = np.linalg.matrix_rank(C2c2)

        if C2_rank != C2c2_rank:
            if verbose:
                print("Infeasible: Constraints with expectation cannot be satisfied (rank deficiency).")
            return None, None, None, None
        else:
            v, lambda_ = compute_v_given_C(covariance_matrix, C2, c2)
            return v, compute_expectation(v, E_Y), compute_variance(v, covariance_matrix), lambda_
                
def compute_active_set(covariance_matrix, E_Y, m, r_values, numerically_stable = True, verbose=True):
    # Start with no active constraints:
    total_components = len(E_Y)
    b_components = total_components // 2

    active_set_r = set()  # Set for r_i constraints
    active_set_n = set()  # Set for n_i constraints
    iter = 0 
    while True:
        iter +=1
        if iter>100:
            print("The algorithm cycles, returning infeasible, make sure the covariance matrix is of full rank")
            return None, None, None, None, None
        if verbose == True:
            shown_indices_r = [x + 1 for x in active_set_r]
            shown_indices_n = [x + 1 for x in active_set_n]
            print(f"Active set credibility: {shown_indices_r}")
            print(f"Active set non-negativity: {shown_indices_n}")

        v, expected_value, variance, lambda_ = compute_given_active_constraints(E_Y, covariance_matrix, active_set_n, active_set_r, m, r_values, verbose)
        if v is None:
            if verbose == True:
                print(f"Active sets credibility {active_set_r} combined with {active_set_n} does not satisfy expectation condition {m}.")
            return None, None, None, None, None

        if verbose == True:
            print(f"Lambda: {lambda_}")

        lambdas_n = lambda_[:len(active_set_n)]
        lambdas_r = lambda_[len(active_set_n):len(active_set_n) + len(active_set_r)]

        if verbose == True:
            print(f"Lambdas non negative: {np.round(lambdas_n, 2)}")
            print(f"Lambdas credibility: {np.round(lambdas_r, 2)}")
            print(f"Current values: {np.round(v, 2)}")

        # Check for violated constraints
        violations = []
        # v_i < 0 for i in 0..2n-1
        for i in range(total_components):
            if v[i] < -0.01  and i not in active_set_n:
                violations.append((v[i], i, 'n'))
        # r_j v_j - v_{n+j} < 0 for j in 0..n-1

        for j in range(b_components):
            val = r_values[j] * v[j] - v[b_components + j]
            if val < -0.01 and j not in active_set_r:
                violations.append((val, j, 'r'))

        if verbose:
            n_violations = [(val, idx) for val, idx, kind in violations if kind == 'n']
            r_violations = [(val, idx) for val, idx, kind in violations if kind == 'r']
            if n_violations:
                print(f"Non-negativity violations (index, value): {[(idx, float(val)) for val, idx in n_violations]}")
            if r_violations:
                print(f"Credibility violations (index, value): {[(idx, float(val)) for val, idx in r_violations]}")

        if violations:
            # Find the most violated constraint (minimum value)
            min_violation = min(violations, key=lambda x: x[0])
            violation_value, idx, kind = min_violation
            if kind == 'n':
                active_set_n.add(idx)
                if verbose:
                    print(f"Adding index {idx} to A_n (non-negativity constraint violated).")
                    print(f"Current value: {v[idx]}")
            else:
                active_set_r.add(idx)
                if verbose:
                    print(f"Adding index {idx} to A_r (credibility constraint violated).")
                    print(f"With violation value: {violation_value}")


        else:
            # Remove the last added component from the active set if any lambda is negative
            all_lambdas = list(lambdas_n) + list(lambdas_r)
            all_active = list(active_set_n) + list(active_set_r)
            negative_lambdas = []
            for idx, val in zip(all_active, all_lambdas):
                if idx in active_set_r:
                    # Only consider removal for r if the corresponding element is nonzero
                    if not numerically_stable:
                        b_components = len(E_Y) // 2
                        val_r = min(v[idx], v[b_components + idx])
                        if val < 0 and val_r > 1e-3:
                            negative_lambdas.append((idx, val))
                    else:
                        if val < 0:
                            negative_lambdas.append((idx, val))
                else:
                    if val < 0:
                        negative_lambdas.append((idx, val))
            if negative_lambdas:
                idx_to_remove, _ = min(negative_lambdas, key=lambda x: x[1])
                if idx_to_remove in active_set_n:
                    active_set_n.remove(idx_to_remove)
                    if verbose:
                        print(f"Removing component {idx_to_remove} from A_n (non-negativity), lambda negative.")
                elif idx_to_remove in active_set_r:
                    active_set_r.remove(idx_to_remove)
                    if verbose:
                        print(f"Removing component {idx_to_remove} from A_r (credibility), lambda negative.")
            else:
                break
    
    if verbose == True:
        shown_indices_r = [x + 1 for x in active_set_r]
        shown_indices_n = [x + 1 for x in active_set_n]
        print(f"Final active set (credibility): {shown_indices_r}")
        print(f"Final active set (non-negativity): {shown_indices_n}")
        print(f"Final values: {np.round(v, 2)}")
        print(f"Final expectation: {expected_value}")
        print(f"Final variance: {variance}")
        print(f"Final lambda: {np.round(lambda_, 2)}")
    active_set = [active_set_r, active_set_n]
    return active_set, v, expected_value, variance, lambda_

def compute_frontier(Range, E_Y, covariance_matrix, r_valuesm, numerically_stable = True):
    """
    Compute the Pareto frontier for a given set of parameters.
    """
    v_values = []
    variances = []
    expectations = []

    for i in range(1, Range):
        M_value = i
        active_set, v, exp, var, lambda_ = compute_active_set(covariance_matrix, E_Y,  M_value, r_values, numerically_stable=numerically_stable, verbose=False)
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
    if len(expectations_list) == 1:
        plt.title("Pareto Frontier")
    else:
        plt.title("Pareto Frontiers")
    ax1.legend()
    plt.grid()
    fig.tight_layout()
    plt.savefig(f"{folder}PF_mp_organic.png", dpi=300)
    plt.show()


def plot_planted_and_forward_side_by_side(expectations, v_values, num_crops, yield_means, folder = "Figures/", product_types=['Improved', 'Hybrid', 'Local', 'Organic']):
    """
    Plot percentage planted and forward contracts side by side, starting from the first deviation.
    """
    # Compute x-axis values (expectations) and filter out None v_values
    x = [expectations[j] for j, v in enumerate(v_values) if v is not None]
    filtered_v_values = [v for v in v_values if v is not None]

    # Prepare planted percentages and forward contract percentages
    planted_percentages = [
        [v[i] for v in filtered_v_values]
        for i in range(num_crops)
    ]
    percentage_contracts = [
        [
            v[num_crops + i] / yield_means[i] #(v[i] * yield_means[i])
            #if abs(v[i] * yield_means[i]) > 1e-10 else np.nan
            for v in filtered_v_values
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
            this_start = max(0, deviation_idx - 100)
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
    ax2.set_ylabel(r"Forward Contract Size  $q_i^*$ (% of $ \mathbb{E}[Q_i]$)")
    ax2.set_title("Forward Contract sizes")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.savefig(f"{folder}q_mp_organic.png", dpi=300)
    plt.show()

def sample_q_organic(yields_local, mu_Z=0.94, sigma_Z=0.0702, N_samples=100000, random_state=None):
    """
    Samples Q_organic = Z_i * Q_local, where:
    - Z_i ~ Normal(mu_Z, sigma_Z^2)
    - Q_local is sampled empirically from yields_local['median_farm_yield_quintale']
    
    Parameters:
        yields_local (pd.DataFrame): DataFrame with 'median_farm_yield_quintale' column
        mu_Z (float): Mean of Z_i (conversion ratio)
        sigma_Z (float): Std deviation of Z_i
        N_samples (int): Number of samples to generate
        random_state (int or None): Seed for reproducibility

    Returns:
        pd.Series: Sampled Q_organic values
    """
    rng = np.random.default_rng(seed=random_state)

    # Sample from empirical distribution of Q_local
    q_local_values = yields_local['median_farm_yield_quintale'].values
    q_local_sample = rng.choice(q_local_values, size=N_samples, replace=True)

    # Sample Z_i from normal distribution
    z_samples = rng.normal(loc=mu_Z, scale=sigma_Z, size=N_samples)

    # Compute Q_organic
    q_organic = z_samples * q_local_sample

    return pd.Series(q_organic, name="Q_organic_sample")


def collect_crop_statistics(yield_csv="crop_yield.csv", cov_csv="covariance_matrix_P_Q_neg_PQ.csv", verbose=False):
    yields = pd.read_csv(yield_csv, low_memory=False)

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

    EXPECTATION_DIFFERENCE = 0.94
    VARIANCE = 0.0702**2
    
    mean_price_organic = mean_price_local
    mean_yield_organic= mean_yield_local * EXPECTATION_DIFFERENCE

    mean_prices = [mean_price_improved, mean_price_hybrid, mean_price_local,mean_price_organic]
    mean_Q = [mean_yield_improved, mean_yield_hybrid, mean_yield_local, mean_yield_organic]

    PQ_improved = yields_improved['price_quantity'].mean()
    PQ_hybrid = yields_hybrid['price_quantity'].mean()
    PQ_local = yields_local['price_quantity'].mean()
    PQ_organic = yields_local['price_quantity'].mean() * EXPECTATION_DIFFERENCE

    PQ_values = [PQ_improved, PQ_hybrid, PQ_local, PQ_organic]

    if verbose:
        print("Mean Price Local:", mean_price_local)
        print("Mean Price Organic:", mean_price_organic)
        print("Mean PQ Local:", PQ_local)
        print("Mean PQ Organic:", PQ_organic)

    covariance_matrix = pd.read_csv(cov_csv, low_memory=False).drop(columns=['Unnamed: 0'])
    covariance_matrix = np.array(covariance_matrix)

    cov_matrix_extended = np.zeros((8, 8))

    # Mapping from old to new indices
    old_to_new = [0, 1, 2, 4, 5, 6]
    # Fill in the new matrix
    for i_old, i_new in enumerate(old_to_new):
        for j_old, j_new in enumerate(old_to_new):
            cov_matrix_extended[i_new, j_new] = covariance_matrix[i_old, j_old]

    # Add new rows and columns for organic
    # First the rows
    for i in range(len(cov_matrix_extended)):
        # First the rows
        cov_matrix_extended[i, 3] = cov_matrix_extended[i, 2] * EXPECTATION_DIFFERENCE
        cov_matrix_extended[i, 7] = cov_matrix_extended[i, 6]
            
        # Then the columns
        cov_matrix_extended[3, i] = cov_matrix_extended[i, 3]
        cov_matrix_extended[7, i] = cov_matrix_extended[i, 7]
        
        
    # Finally, set the diagonal for organic
    cov_matrix_extended[3, 3] = VARIANCE* covariance_matrix[2, 2] + VARIANCE * PQ_local**2 +  covariance_matrix[2, 2]  * EXPECTATION_DIFFERENCE**2
    cov_matrix_extended[7, 7] = cov_matrix_extended[6,6]
    if verbose:
        print("Organic PQ variance:", cov_matrix_extended[3, 3])
        print("Organic P variance:", cov_matrix_extended[7, 7])
        print("Local PQ variance:", cov_matrix_extended[2, 2])
        print("Local P variance:", cov_matrix_extended[6, 6])

    # Create invertiblity 
    cov_matrix_extended = cov_matrix_extended + np.eye(cov_matrix_extended.shape[0]) * 1e-10

    yields_organic = sample_q_organic(yields_local, random_state=42)

    return {
        "covariance_matrix": cov_matrix_extended,
        "mean_prices": mean_prices,
        "mean_Q": mean_Q,
        "PQ_values": PQ_values,
        "yield_types": {
            'Improved': yields_improved['median_farm_yield_quintale'],
            'Hybrid': yields_hybrid['median_farm_yield_quintale'],
            'Local': yields_local['median_farm_yield_quintale'],
            'Organic': yields_organic
        }
    }


def compute_E_Y(mean_prices, PQ_values, price_multipliers):
    """
    Compute E_Y vector for the model.

    Args:
        mean_prices (list or array): Mean prices for each crop [improved, hybrid, local].
        PQ_values (list or array): Mean price*quantity for each crop [improved, hybrid, local].
        price_multipliers (list or array): Multipliers for each crop's price (e.g., [1.1, 1, 0.9]).

    Returns:
        np.ndarray: The E_Y vector.
    """
    mean_fP = [mean_prices[i] * price_multipliers[i] - mean_prices[i] for i in range(len(mean_prices))]
    mean_PQ = list(PQ_values) + mean_fP
    return np.array(mean_PQ)

def compute_r_values(yield_types, alphas, verbose=False):
    """
    Given a dict of yield series and a list of alphas, return r_values (quantiles).
    """
    r_values = []
    for (crop, data), alpha in zip(yield_types.items(), alphas):
        quantile = data.quantile(1 - alpha)
        r_values.append(quantile)

    if verbose:
        print("Quantiles for each crop type (r_i):")
        for crop, r in zip(yield_types.keys(), r_values):
            print(f"{crop}: {r}")
    return r_values

def plot_cdf(distributions: dict, title="Empirical CDFs", save_as=None):
    plt.figure(figsize=(8, 5))

    for name, yield_series in distributions.items():
        data = np.sort(pd.Series(yield_series).dropna())
        cdf = np.arange(1, len(data) + 1) / len(data)
        plt.step(data, cdf, where='post', label=name)
        #plt.plot(data, cdf, marker='.', linestyle='none', label=name)

    plt.xlabel('Yield (quintal)')
    plt.ylabel('CDF')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    if save_as:
        plt.savefig(f"Figures/{save_as}.png")
    
    plt.show()

def plot_organic_farming_solutions(variable_values, v_values, expectations, variances, name = "undefined", variable_name = "Forward price premiums"):
    """
    Plot the amount of organic farming (index 3 in v_values) for:
    - minimal variance solution
    - median variance solution
    - maximum expectation solution

    Args:
        v_values (list of arrays): List of solution vectors.
        expectations (list): List of expectations.
        variances (list): List of variances.
    """
    # Filter out None solutions
    minimal_variance_values = []
    median_values = []
    max_expectation_values =[]
    average_values = []

    for i, v in enumerate(v_values):
        variances_i = variances[i]

        # Create a list of (index, value) pairs excluding None values
        valid_pairs = [(idx, val) for idx, val in enumerate(variances_i) if val is not None]

        # Find the pair with minimal variance value
        min_idx, min_val = min(valid_pairs, key=lambda x: x[1])

        # Similarly for median, but you must handle medians carefully since median might be None or non-existing
        unique_variances = [val for val in set(variances_i) if val is not None]
        median_variance = np.median(unique_variances)

        # Find index of median_variance (first occurrence)
        median_idx = median_idx = min(
            (idx for idx, val in enumerate(variances_i) if val is not None),
            key=lambda idx: abs(variances_i[idx] - median_variance)
        )

        # For max expectation (similar to min)
        expectations_i = expectations[i]
        valid_exp_pairs = [(idx, val) for idx, val in enumerate(expectations_i) if val is not None]
        max_exp_idx, max_exp_val = max(valid_exp_pairs, key=lambda x: x[1])
        

        # Now get corresponding v values
        minimal_variance_value = v[min_idx][3]
        median_value = v[median_idx][3]
        max_expectation_value = v[max_exp_idx][3]
        set_values = set([val[3] for val in v if val is not None])
        average_value = np.mean(list(set_values))

        # Append or do whatever you need
        minimal_variance_values.append(minimal_variance_value)
        median_values.append(median_value)
        max_expectation_values.append(max_expectation_value)
        average_values.append(average_value)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    names = ["Risk Averse", "Risk Neutral", "Risk Loving", "Average allocation"]
    for i, v in enumerate([minimal_variance_values, median_values, max_expectation_values, average_values]):
        ax1.plot(variable_values, v, label=names[i])

    ax1.set_xlabel(variable_name)
    ax1.set_ylabel("Organic Farming land allocation (%)")
    ax1.set_title(f"Organic Farming land allocation: {variable_name}")
    ax1.legend()
    ax1.grid()

    plt.savefig(f"Figures/different_levels{name}.png")  # Or replace with your preferred filename
    plt.show()

if __name__ == "__main__": 
    stats = collect_crop_statistics(verbose=False)
    covariance_matrix = stats["covariance_matrix"]
    mean_prices = stats["mean_prices"]
    mean_Q = stats["mean_Q"]
    PQ_values = stats["PQ_values"]
    yield_types = stats["yield_types"]

    selected = {k: yield_types[k] for k in ['Organic' ,'Local']}
    #plot_cdf(selected, save_as="empirical_cdf")

    # Example usage for 1.1 E[P]
    price_multipliers = [1, 1, 1, 1]
    E_Y = compute_E_Y(mean_prices, PQ_values, price_multipliers)

    alphas = [0.5, 0.5, 0.5, 0.5]  # as defined above
    r_values = compute_r_values(yield_types, alphas, verbose=True)
    active_set, v, expected_value, variance, lambda_ = compute_active_set(covariance_matrix, E_Y, 0, r_values)
    print()


    v_values, variances, expectations = compute_frontier(2000, E_Y, covariance_matrix, r_values)
    plot_planted_and_forward_side_by_side([i for i in range(2000)], v_values, len(E_Y) // 2, mean_Q, folder="Figures/base_case")

    import credibility_model as cred
    pm2 = [1,1,1]
    alpha2 =[0.5, 0.5, 0.5]
    selected = {k: yield_types[k] for k in ['Improved', 'Hybrid' ,'Local']}
    r_values2= compute_r_values(selected, alpha2)
    stats2 = cred.collect_crop_statistics()
    covariance_matrix2 = stats2["covariance_matrix"]
    mean_prices2 = stats2["mean_prices"]
    PQ_values2 = stats2["PQ_values"]
    yield_types2 = stats2["yield_types"]

    E_Y2 = cred.compute_E_Y(mean_prices2, PQ_values2, pm2)
    v_values_no, variances_no, expectations_no = cred.compute_frontier(2000, E_Y2, covariance_matrix2, r_values2)

    expectations = [ expectations, expectations]
    variances = [ variances, variances_no]
    labels = [ 'With Organic', 'Without Organic']

    plot_frontier(expectations, variances,folder="Figures/base_case_", labels=labels)

    #Compute frontiers for different organic price multipliers
    organic_price_factors = np.arange(0.95, 1.15, 0.005)
    frontier_v_values =[]
    frontier_expectations = []
    frontier_variances = []

    for f in organic_price_factors:
        print("Computing frontier for: ",f," times expected spot price" )
        pm = [1, 1, 1, f]
        E_Y_f = compute_E_Y(mean_prices, PQ_values, pm)
        v_values, variances_f, expectations_f = compute_frontier(2000, E_Y_f, covariance_matrix, r_values)
        frontier_v_values.append(v_values)
        frontier_expectations.append(expectations_f)
        frontier_variances.append(variances_f)

    plot_organic_farming_solutions(organic_price_factors, frontier_v_values, frontier_expectations, frontier_variances, name="f_values")

    #Also compute for alpha differences
    price_multipliers = [1, 1, 1, 1]
    E_Y = compute_E_Y(mean_prices, PQ_values, price_multipliers)


    credibility_levels = np.arange(0, 1, 0.02)
    frontier_v_values =[]
    frontier_expectations = []
    frontier_variances = []

    for a in credibility_levels:
        print("Computing frontier for credibility level: ",a )
        alphas = [0.5, 0.5, 0.5, a]  # as defined above
        r_values = compute_r_values(yield_types, alphas, verbose=False)
        v_values, variances_f, expectations_f = compute_frontier(2000, E_Y, covariance_matrix, r_values, numerically_stable = False)
        frontier_v_values.append(v_values)
        frontier_expectations.append(expectations_f)
        frontier_variances.append(variances_f)

    plot_organic_farming_solutions(credibility_levels, frontier_v_values, frontier_expectations, frontier_variances, name="a_values", variable_name= "Credibility levels")


    




