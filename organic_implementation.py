import credibility_model as cred
import single_product as single
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

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

    mean_prices = [mean_price_improved, mean_price_hybrid, mean_price_local]
    mean_Q = [mean_yield_improved, mean_yield_hybrid, mean_yield_local]

    PQ_improved = yields_improved['price_quantity'].mean()
    PQ_hybrid = yields_hybrid['price_quantity'].mean()
    PQ_local = yields_local['price_quantity'].mean()

    PQ_values = [PQ_improved, PQ_hybrid, PQ_local]

    covariance_matrix = pd.read_csv(cov_csv, low_memory=False).drop(columns=['Unnamed: 0'])
    covariance_matrix = np.array(covariance_matrix)

    EXPECTATION_DIFFERENCE = 0.94
    VARIANCE = 0.0702**2
    
    mean_price_organic = mean_price_local
    mean_yield_organic= mean_yield_local * EXPECTATION_DIFFERENCE
    PQ_organic = yields_local['price_quantity'].mean() * EXPECTATION_DIFFERENCE

    covariance_PQ_P = covariance_matrix[2,5] * EXPECTATION_DIFFERENCE *-1 # Times -1 as the cov matrix measures Cov(PQ, -P)
    variance_price = covariance_matrix[5,5]
    cov_PQ = (PQ_local - mean_yield_local * mean_price_local) * EXPECTATION_DIFFERENCE 
    # Compute Var(Z P3 Q3) using the formula:
    # Var(Z P3 Q3) = Var(Z) * Var(P3 Q3) + Var(Z) * E[P3 Q3]^2 + Var(P3 Q3) * E[Z]^2
    var_Z = VARIANCE
    E_Z = EXPECTATION_DIFFERENCE
    var_P3Q3 = covariance_matrix[2,2]
    E_P3Q3 = PQ_local
    var_PQ = var_Z * var_P3Q3 + var_Z * (E_P3Q3 ** 2) + var_P3Q3 * (E_Z ** 2)

    if verbose :
        print("Mean Price Local:", mean_price_local)
        print("Mean Price Organic:", mean_price_organic)
        print("Mean PQ Local:", PQ_local)
        print("Mean PQ Organic:", PQ_organic)
        print("Mean Organic Yield" ,mean_yield_organic)
        print()
        print("Covariance PQ_P: ", covariance_PQ_P )
        print("Variance Price: ", variance_price)

    yields_organic = sample_q_organic(yields_local, mu_Z=EXPECTATION_DIFFERENCE, sigma_Z=VARIANCE**0.5, random_state=42)

    return {
        "covariance_matrix": covariance_matrix,
        "mean_prices": mean_prices,
        "mean_Q": mean_Q,
        "PQ_values": PQ_values,
        "yield_types": {
            'Improved': yields_improved['median_farm_yield_quintale'],
            'Hybrid': yields_hybrid['median_farm_yield_quintale'],
            'Local': yields_local['median_farm_yield_quintale'],
            'Organic': yields_organic
        },
        "organic_stats" : (mean_price_organic, mean_yield_organic, PQ_organic, covariance_PQ_P, cov_PQ, variance_price, var_PQ)
    }

def ensure_constraint_on_front(q_values, expectations, variances, r_value, forward_price, organic_stats):
    price_org, yield_org, PQ_org, cov_PQ_P_org, cov_PQ_org, var_price_org, var_PQ_org = organic_stats 
    variances_org = np.array(variances)
    expectations_org = np.array(expectations)
    variances_org[q_values > r_value] = single.compute_variance(r_value, var_PQ_org, var_price_org, cov_PQ_P_org)
    expectations_org[q_values > r_value] =  single.compute_expectation(r_value, yield_org, price_org, forward_price, cov_PQ_org)
    # Cap q_values_org at r
    q_values_org = np.minimum(q_values, r_value)
    # Ensure arrays are numpy arrays
    q_values_org = np.array(q_values_org)
    expectations_org = np.array(expectations_org)

    return q_values_org, expectations_org, variances_org


def create_base_case(stats, pm= 1, a =0.8):
    covariance_matrix = stats["covariance_matrix"]
    mean_prices = stats["mean_prices"]
    mean_Q = stats["mean_Q"]
    PQ_values = stats["PQ_values"]
    yield_types = stats["yield_types"]
    organic_stats = stats["organic_stats"]
    price_org, yield_org, PQ_org, cov_PQ_P_org, cov_PQ_org, var_price_org, var_PQ_org = organic_stats
    
    organic_yields = yield_types["Organic"]
    del yield_types["Organic"]

    alpha_organic = a
    f_organic = organic_stats[0] *pm
    r_value_organic = cred.compute_r_values({"organic": organic_yields}, [alpha_organic], verbose=True)
    cred.compute_fill_rate({"organic": organic_yields}, r_value_organic)

    q_values_org, variances_org, expectations_org = single.compute_frontier(2000,cov_PQ_org, cov_PQ_P_org, var_price_org, var_PQ_org, yield_org, price_org, f_organic)
    
    # Limit the pareto front to values where q_values <= r_value_organic
    r_value_organic = r_value_organic[0]
    q_values_org, expectations_org, variances_org = ensure_constraint_on_front(q_values_org,expectations_org ,variances_org, r_value_organic, f_organic, organic_stats)
    

    # Example usage for 1.1 E[P]
    price_multipliers = [pm for i in range(len(mean_prices))]
    E_Y = cred.compute_E_Y(mean_prices, PQ_values, price_multipliers)

    alphas = [a for i in range(len(yield_types))] # as defined above
    r_values = cred.compute_r_values(yield_types, alphas, verbose=True)
    
    cred.compute_fill_rate(yield_types, r_values)
    v_values_conventional, variances_conventional, expectations_conventional = cred.compute_frontier(2000, E_Y, covariance_matrix, r_values)

    expectations = [expectations_org, expectations_conventional]
    variances = [variances_org, variances_conventional]
    labels = ["Organic", "Conventional"]

    cred.plot_frontier(expectations, variances, labels=labels, folder="Figures/base_case_org")

def examine_forward_price_influence(stats, a=0.8, pm =1, price_range = [0.9, 1.15]):
    covariance_matrix = stats["covariance_matrix"]
    mean_prices = stats["mean_prices"]
    mean_Q = stats["mean_Q"]
    PQ_values = stats["PQ_values"]
    yield_types = stats["yield_types"]
    organic_stats = stats["organic_stats"]
    price_org, yield_org, PQ_org, cov_PQ_P_org, cov_PQ_org, var_price_org, var_PQ_org = organic_stats
    
    organic_yields = yield_types["Organic"]
    del yield_types["Organic"]

    price_multipliers = [pm for i in range(len(mean_prices))]
    E_Y = cred.compute_E_Y(mean_prices, PQ_values, price_multipliers)

    alphas = [a for i in range(len(yield_types))] # as defined above
    r_values = cred.compute_r_values(yield_types, alphas, verbose=True)

    v_values_conventional, variances_conventional, expectations_conventional = cred.compute_frontier(2000, E_Y, covariance_matrix, r_values)
    # Only select non-None points for conventional_points
    conventional_points = [(e, v) for e, v in zip(expectations_conventional, variances_conventional) if e is not None and v is not None]

    alpha_organic = a
    r_value_organic = cred.compute_r_values({"organic": organic_yields}, [alpha_organic], verbose=False)
    r_value_organic = r_value_organic[0]

    engage_organic_farming_arrays = []
    org_avg_array =[]
    org_low_risk = []
    org_high_risk = []

    for price_factor in np.arange(price_range[0], price_range[1], 0.005):
        f_organic = organic_stats[0] * price_factor
        q_values_org, variances_org, expectations_org = single.compute_frontier(2000,cov_PQ_org, cov_PQ_P_org, var_price_org, var_PQ_org, yield_org, price_org, f_organic)
        # Limit the pareto front to values where q_values <= r_value_organic
        
        q_values_org, expectations_org, variances_org = ensure_constraint_on_front(q_values_org,expectations_org ,variances_org, r_value_organic, f_organic, organic_stats)
        organic_points = [(q, e, v) for q,e, v in zip(q_values_org,expectations_org, variances_org) if e is not None and v is not None]
        # Remove None values before computing min/max
        valid_expectations_org = [x for x in expectations_org if x is not None]
        valid_expectations_conventional = [x for x in expectations_conventional if x is not None]
        
        start_front = min(valid_expectations_org + valid_expectations_conventional)
        end_front = max(valid_expectations_org + valid_expectations_conventional)
        
        engage_organic = []
        org_avg = []
        for required_expectation in range(math.floor(start_front), math.ceil(end_front)):
            filtered_organic = [pair for pair in organic_points if pair[1] >= required_expectation]
            filterd_conventional = [pair for pair in conventional_points if pair[0] >= required_expectation]
            if filtered_organic and filterd_conventional:
                org_q, organic_expectation, organic_variance = min(filtered_organic, key= lambda x: x[1])
                conventional_expectation, conventional_variance = min(filterd_conventional, key= lambda x: x[1])
                engage_organic.append(1 if organic_variance <= conventional_variance else 0)
                org_avg.append(org_q if organic_variance <= conventional_variance else 0)
            elif filtered_organic:
                engage_organic.append(1)
                org_avg.append(org_q)
            else:
                engage_organic.append(0)
                org_avg.append(0)

        #print("Average Allocation", np.mean(engage_organic))
        # Split org_avg into left and right halves and take their means
        mid = len(org_avg) // 2
        left_half_mean = np.mean(org_avg[:mid]) if mid > 0 else 0
        right_half_mean = np.mean(org_avg[mid:]) if mid > 0 else 0
        org_low_risk.append(left_half_mean)
        org_high_risk.append(right_half_mean)
        org_avg_array.append(np.mean(org_avg))
        engage_organic_farming_arrays.append(engage_organic)

    minimum_risk = [engage[0] for engage in engage_organic_farming_arrays]
    median_risk = [np.quantile(engage, 0.5) for engage in engage_organic_farming_arrays]
    maximum_risk = [engage[-1] for engage in engage_organic_farming_arrays]
    average_risk = [np.mean(engage) for engage in engage_organic_farming_arrays]

    # Compute lower and upper half means for each engage_organic array
    lower_half = []
    upper_half = []
    for engage in engage_organic_farming_arrays:
        mid = len(engage) // 2
        lower_half.append(np.mean(engage[:mid]) if mid > 0 else 0)
        upper_half.append(np.mean(engage[mid:]) if mid > 0 else 0)

    # Group results into a dictionary for clarity
    results = {
        "range": np.arange(price_range[0], price_range[1], 0.005),
        "minimum_risk": minimum_risk,
        "median_risk": median_risk,
        "maximum_risk": maximum_risk,
        "average_risk": average_risk,
        "org_avg_array": org_avg_array,
        "org_low_risk": org_low_risk,
        "org_high_risk": org_high_risk,
        "lower_half": lower_half,
        "upper_half": upper_half
    }
    return results

def examine_cred_level(stats, a=0.8, pm =1, cred_range =[0, 1]):
    covariance_matrix = stats["covariance_matrix"]
    mean_prices = stats["mean_prices"]
    mean_Q = stats["mean_Q"]
    PQ_values = stats["PQ_values"]
    yield_types = stats["yield_types"]
    organic_stats = stats["organic_stats"]
    price_org, yield_org, PQ_org, cov_PQ_P_org, cov_PQ_org, var_price_org, var_PQ_org = organic_stats
    
    organic_yields = yield_types["Organic"]
    del yield_types["Organic"]

    price_multipliers = [pm for _ in range(len(mean_prices))]
    E_Y = cred.compute_E_Y(mean_prices, PQ_values, price_multipliers)

    alphas = [a for i in range(len(yield_types))] # as defined above
    r_values = cred.compute_r_values(yield_types, alphas, verbose=False)

    v_values_conventional, variances_conventional, expectations_conventional = cred.compute_frontier(2000, E_Y, covariance_matrix, r_values)
    # Only select non-None points for conventional_points
    conventional_points = [(e, v) for e, v in zip(expectations_conventional, variances_conventional) if e is not None and v is not None]


    f_organic = organic_stats[0] * pm
    alpha_organic = a
    r_value_organic = cred.compute_r_values({"organic": organic_yields}, [alpha_organic], verbose=False)
    r_value_organic = r_value_organic[0]

    engage_organic_farming_arrays = []


    for cred_factor in np.arange(cred_range[0], cred_range[1], 0.01):
        alpha_organic = cred_factor
        r_value_organic = cred.compute_r_values({"organic": organic_yields}, [alpha_organic], verbose=False)
        r_value_organic = r_value_organic[0]

        q_values_org, variances_org, expectations_org = single.compute_frontier(2000,cov_PQ_org, cov_PQ_P_org, var_price_org, var_PQ_org, yield_org, price_org, f_organic)
        # Limit the pareto front to values where q_values <= r_value_organic
        
        q_values_org, expectations_org, variances_org = ensure_constraint_on_front(q_values_org,expectations_org ,variances_org, r_value_organic, f_organic, organic_stats)
        organic_points = [(e, v) for e, v in zip(expectations_org, variances_org) if e is not None and v is not None]
        # Remove None values before computing min/max
        valid_expectations_org = [x for x in expectations_org if x is not None]
        valid_expectations_conventional = [x for x in expectations_conventional if x is not None]
        
        start_front = min(valid_expectations_org + valid_expectations_conventional)
        end_front = max(valid_expectations_org + valid_expectations_conventional)
        
        engage_organic = []
        for required_expectation in range(math.floor(start_front), math.ceil(end_front)):
            filtered_organic = [pair for pair in organic_points if pair[0] >= required_expectation]
            filterd_conventional = [pair for pair in conventional_points if pair[0] >= required_expectation]
            if filtered_organic and filterd_conventional:
                organic_expectation, organic_variance = min(filtered_organic, key= lambda x: x[1])
                conventional_expectation, conventional_variance = min(filterd_conventional, key= lambda x: x[1])
                engage_organic.append(1 if organic_variance <= conventional_variance else 0)
            elif filtered_organic:
                engage_organic.append(1)
            else:
                engage_organic.append(0)

        #print("Average Allocation", np.mean(engage_organic))
        engage_organic_farming_arrays.append(engage_organic)

    minimum_risk = [engage[0] for engage in engage_organic_farming_arrays]
    median_risk = [np.quantile(engage, 0.5) for engage in engage_organic_farming_arrays]
    maximum_risk = [engage[-1] for engage in engage_organic_farming_arrays]
    average_risk = [np.mean(engage) for engage in engage_organic_farming_arrays]

    return np.arange(cred_range[0], cred_range[1], 0.01), minimum_risk, median_risk, maximum_risk, average_risk 

def examine_joint_influence(stats, a_range=[0, 1], pm_range=[0.9, 1.15], step_alpha=0.1, step_pm=0.05):
    """
    Computes a 2D matrix representing the average organic engagement as a function of
    both the credibility level (alpha) and the forward price multiplier.

    Returns:
        alpha_values: list of alpha values (credibility levels)
        price_multipliers: list of forward price multipliers
        average_matrix: 2D array where entry [i][j] is the average organic preference
                        for alpha_values[i] and price_multipliers[j]
    """
    covariance_matrix = stats["covariance_matrix"]
    mean_prices = stats["mean_prices"]
    mean_Q = stats["mean_Q"]
    PQ_values = stats["PQ_values"]
    yield_types = stats["yield_types"]
    organic_stats = stats["organic_stats"]
    price_org, yield_org, PQ_org, cov_PQ_P_org, cov_PQ_org, var_price_org, var_PQ_org = organic_stats

    organic_yields = yield_types["Organic"]
    del yield_types["Organic"]

    alpha_values = np.arange(a_range[0], a_range[1] + step_alpha, step_alpha)
    price_multipliers = np.arange(pm_range[0], pm_range[1] + step_pm, step_pm)

    average_matrix = []
    q_values_avg_matrix =[]
    org_low_risk_matrix = []
    org_high_risk_matrix = []
    lower_half_matrix = []
    upper_half_matrix = []


    for alpha in alpha_values:
        row = []
        row_q_org = []
        org_low_risk_row = []
        org_high_risk_row = []
        lower_half_row = []
        upper_half_row = []

        alphas = [alpha for _ in yield_types]
        r_values = cred.compute_r_values(yield_types, alphas, verbose=False)
        E_Y = cred.compute_E_Y(mean_prices, PQ_values, [1.0 for _ in mean_prices])  # PM handled later

        v_values_conventional, variances_conventional, expectations_conventional = cred.compute_frontier(
            2000, E_Y, covariance_matrix, r_values)
        conventional_points = [(e, v) for e, v in zip(expectations_conventional, variances_conventional) if e is not None and v is not None]
        valid_expectations_conventional = [x for x in expectations_conventional if x is not None]

        for pm in price_multipliers:
            f_organic = price_org * pm
            r_value_organic = cred.compute_r_values({"organic": organic_yields}, [alpha], verbose=False)[0]

            q_values_org, variances_org, expectations_org = single.compute_frontier(
                2000, cov_PQ_org, cov_PQ_P_org, var_price_org, var_PQ_org, yield_org, price_org, f_organic)

            q_values_org, expectations_org, variances_org = ensure_constraint_on_front(
                q_values_org, expectations_org, variances_org, r_value_organic, f_organic, organic_stats)

            organic_points = [(q, e, v) for q,e, v in zip(q_values_org, expectations_org, variances_org) if e is not None and v is not None]
            valid_expectations_org = [x for x in expectations_org if x is not None]

            if not valid_expectations_org:
                row.append(0.0)
                continue
            elif not valid_expectations_conventional:
                row.append(1)
                continue

            start_front = min(valid_expectations_org + valid_expectations_conventional)
            end_front = max(valid_expectations_org + valid_expectations_conventional)

            engage_organic = []
            org_avg = []
            
            for required_expectation in range(math.floor(start_front), math.ceil(end_front)):
                filtered_organic = [pair for pair in organic_points if pair[1] >= required_expectation]
                filtered_conventional = [pair for pair in conventional_points if pair[0] >= required_expectation]

                if filtered_organic and filtered_conventional:
                    o_q, o_e, o_v = min(filtered_organic, key=lambda x: x[1])
                    c_e, c_v = min(filtered_conventional, key=lambda x: x[1])
                    engage_organic.append(1 if o_v <= c_v else 0)
                    org_avg.append(o_q if o_v <= c_v else 0)
                elif filtered_organic:
                    engage_organic.append(1)
                    org_avg.append(o_q)
                else:
                    engage_organic.append(0)
                    org_avg.append(0)

            # Compute left/right half means for org_avg (q values)
            mid = len(org_avg) // 2
            left_half_mean = np.mean(org_avg[:mid]) if mid > 0 else 0
            right_half_mean = np.mean(org_avg[mid:]) if mid > 0 else 0
            row_q_org.append(np.mean(org_avg))
            # For engage_organic (0/1), also compute lower/upper half means
            mid_engage = len(engage_organic) // 2
            left_half_engage = np.mean(engage_organic[:mid_engage]) if mid_engage > 0 else 0
            right_half_engage = np.mean(engage_organic[mid_engage:]) if mid_engage > 0 else 0
            row.append(np.mean(engage_organic) if engage_organic else 0.0)
            org_low_risk_row.append(left_half_mean)
            org_high_risk_row.append(right_half_mean)
            lower_half_row.append(left_half_engage)
            upper_half_row.append(right_half_engage)


        org_low_risk_matrix.append(org_low_risk_row)
        org_high_risk_matrix.append(org_high_risk_row)
        lower_half_matrix.append(lower_half_row)
        upper_half_matrix.append(upper_half_row)
        q_values_avg_matrix.append(row_q_org)
        average_matrix.append(row)

    return (
    alpha_values,
    price_multipliers,
    np.array(average_matrix),
    np.array(q_values_avg_matrix),
    np.array(org_low_risk_matrix),
    np.array(org_high_risk_matrix),
    np.array(lower_half_matrix),
    np.array(upper_half_matrix)
    )


def plot_organic_farming_solutions(range_, values, labels = ["Risk Averse", "Risk Neutral", "Risk Loving", "Pareto front Average"], name = "undefined", variable_name = "Forward price premiums"):
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

    fig, ax1 = plt.subplots(figsize=(8, 5))

    for i, v in enumerate(values):
        ax1.plot(range_, v, label=labels[i])

    ax1.set_xlabel(variable_name)
    ax1.set_ylabel("Organic Farming land allocation (%)")
    ax1.set_title(f"Organic Farming Conversion: {variable_name}")
    ax1.legend()
    ax1.grid()

    plt.savefig(f"Figures/different_levels{name}.png")  # Or replace with your preferred filename
    plt.show()


def plot_heatmap(alpha_values, price_multipliers, average_matrix,
                 ylabel="Credibility Level", xlabel="Forward Price Multiplier",
                 title="Average Organic Preference Heatmap",
                 colorbar_label="Average Organic Preference",
                 filename="heatmap.png"):
    
    plt.figure(figsize=(8, 6))

    # Convert matrix to DataFrame with labeled rows and columns
    import pandas as pd
    df = pd.DataFrame(average_matrix, index=alpha_values, columns=price_multipliers)

    # Use seaborn heatmap
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap='viridis', cbar_kws={'label': colorbar_label},
                     annot_kws={"fontsize": 8}, linewidths=0.5, linecolor='gray')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_average_over_pareto_front(alpha_values, price_multipliers, average_matrix, 
                                    ylabel="Average Organic Preference", 
                                    xlabel="Forward Price Multiplier",
                                    title="Average Organic Preference vs Price Multiplier for Different Credibility Levels",
                                    filename="Figures/average_over_pareto_front_a_values.png"):
    """
    For each alpha (credibility level), plot the average organic preference over the Pareto front
    against the price multipliers.
    """
    plt.figure(figsize=(8, 5))
    for i, alpha in enumerate(alpha_values):
        plt.plot(price_multipliers, average_matrix[i, :], label=f"a={alpha:.2f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Credibility Level")
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_cost_per_hectare(percent_organic_farm_list, price_range_, org_avg_array_list, exp_price_org, 
                          labels=None, ylabel="Cost per Hectare ($)", xlabel="Percentage of land transitioned (%)", 
                          title="Cost per Hectare against percentage converted land", filename="Figures/cost_per_hectare.png"):
    
    plt.figure(figsize=(8, 5))
    for i, (percent_organic_farm, org_avg_array) in enumerate(zip(percent_organic_farm_list, org_avg_array_list)):
        label = labels[i] if labels is not None and i < len(labels) else f"Scenario {i+1}"
        cost = np.array(org_avg_array) *  np.array(price_range_-1) * exp_price_org
        plt.plot(percent_organic_farm, cost, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__": 
    stats = collect_crop_statistics(verbose=True)
    yield_types = stats["yield_types"]

    organic_yields = yield_types["Organic"]
    cred.compute_average_fill_rate_over_a({"organic": organic_yields}, label=["Organic"], name = "fill_percentage_organic" )
    
    create_base_case(stats)

    stats = collect_crop_statistics(verbose=False)
    results = examine_forward_price_influence(stats, price_range=[1, 1.5])
    range_ = results["range"]
    minimum_risk = results["minimum_risk"]
    median_risk = results["median_risk"]
    maximum_risk = results["maximum_risk"]
    average_risk = results["average_risk"]
    lower_half_risk = results["lower_half"]
    upper_half_risk = results["upper_half"]
    org_avg_array = results["org_avg_array"]
    avg_lower_half = results["org_low_risk"]
    avg_upper_half = results["org_high_risk"]

    values = [minimum_risk, median_risk, maximum_risk, average_risk]
    print()
    plot_organic_farming_solutions(range_, values, name = "price_premium")

    plot_cost_per_hectare([lower_half_risk, average_risk, upper_half_risk], range_, [avg_lower_half,org_avg_array,avg_upper_half ], 
                          stats["organic_stats"][0], labels=["Lower half", "Pareto front Average", "Upper half"])

    stats = collect_crop_statistics(verbose=False)
    range_, minimum_risk, median_risk, maximum_risk, average_risk = examine_cred_level(stats)
    values = [minimum_risk, median_risk, maximum_risk, average_risk]
    plot_organic_farming_solutions(range_, values, name = "cred_level", variable_name="Credibility Level")

    stats = collect_crop_statistics(verbose=False)
    alpha_values, price_multipliers, average_matrix, q_value_avg, org_low_risk_matrix, org_high_risk_matrix, lower_half_matrix, upper_half_matrix = examine_joint_influence(stats, a_range = [0.6,0.8],pm_range=[1, 1.5], step_pm=0.02)
    plot_average_over_pareto_front(alpha_values, price_multipliers, average_matrix)

    plot_cost_per_hectare(average_matrix, price_multipliers, q_value_avg, stats["organic_stats"][0], labels=[f"a={alpha:.2f}" for alpha in [0.6,0.7,0.8,0.9]], filename="Figures/cost_avg_a_levels.png")
    plot_cost_per_hectare(lower_half_matrix, price_multipliers, org_low_risk_matrix, stats["organic_stats"][0], labels=[f"a={alpha:.2f}" for alpha in [0.6,0.7,0.8,0.9]], filename="Figures/cost_low_a_levels.png")
    plot_cost_per_hectare(upper_half_matrix, price_multipliers, org_high_risk_matrix, stats["organic_stats"][0], labels=[f"a={alpha:.2f}" for alpha in [0.6,0.7,0.8,0.9]], filename="Figures/cost_high_a_levels.png")

    stats = collect_crop_statistics(verbose=False)
    alpha_values, price_multipliers, average_matrix, q_value_avg, org_low_risk_matrix, org_high_risk_matrix, lower_half_matrix, upper_half_matrix = examine_joint_influence(stats,pm_range=[1, 1.5])
    #plot_cost_per_hectare(average_matrix, price_multipliers, q_value_avg, stats["organic_stats"][0], labels=[f"a={alpha *0.1:.2f}" for alpha in range(11)],  filename="Figures/appendix_cost.png")
    
    
    # Compute baseline: average organic preference for a_Organic = 0.8 (fixed credibility)
    baseline_alpha_idx = np.argmin(np.abs(alpha_values - 0.8))
    baseline_row = average_matrix[baseline_alpha_idx, :]

    # Calculate the added benefit matrix: difference from baseline at each price premium
    added_benefit_matrix = average_matrix - baseline_row

    alpha_values = np.round(alpha_values,2)
    price_multipliers = np.round(price_multipliers,2)

    # Optionally, plot the added benefit heatmap as well
    plot_heatmap(
        alpha_values, price_multipliers, added_benefit_matrix,
        ylabel="Credibility Level",
        xlabel="Forward Price Multiplier",
        title="Added Benefit of changed Credibility Level",
        colorbar_label= "Added benefit (Percentage point)",
        filename="Figures/added_benefit_heatmap.png"
    )

    plot_heatmap(alpha_values, price_multipliers, average_matrix, filename="Figures/heatmap.png")











