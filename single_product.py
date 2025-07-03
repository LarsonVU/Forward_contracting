import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_variance(q, var_price_quantity, var_price, covariance_PQ_P):
    return (var_price_quantity - 2 * covariance_PQ_P * q + var_price * q ** 2)

def compute_expectation(q, mean_yield, mean_spot_price, sept_forward_price_quintale, covariance_price_yield):
    return mean_yield * mean_spot_price + covariance_price_yield + q * (sept_forward_price_quintale - mean_spot_price)

def compute_lambda(q_star, var_price, covariance_PQ_P, sept_forward_price_quintale, mean_spot_price):
    return (2* covariance_PQ_P - 2 *q_star *var_price) / (sept_forward_price_quintale - mean_spot_price)

def compute_solution(M, covariance_price_yield, covariance_PQ_P, var_price, var_price_quantity, mean_yield, mean_spot_price, sept_forward_price_quintale):
    """
    Compute the minimum variance for a given set of parameters.
    """
    # q* = sigma_PQ,P / var_price
    q_star_1 = max(0,covariance_PQ_P / var_price)
    # q* = (M- E[P]E[Q] +Cov(P,Q)) / (f-E[P])
    if not (sept_forward_price_quintale - mean_spot_price) == 0:
        q_star_2 =  (M - mean_spot_price * mean_yield - covariance_price_yield) / (sept_forward_price_quintale - mean_spot_price)
    else: 
        q_star_2 = np.nan
    
    if  compute_expectation(q_star_1, mean_yield, mean_spot_price, sept_forward_price_quintale, covariance_price_yield) >= M:
        return q_star_1, compute_variance(q_star_1, var_price_quantity, var_price, covariance_PQ_P), compute_expectation(q_star_1, mean_yield, mean_spot_price, sept_forward_price_quintale, covariance_price_yield), 0
    elif not np.isnan(q_star_2) and q_star_2 >= 0 and compute_expectation(q_star_2, mean_yield, mean_spot_price, sept_forward_price_quintale, covariance_price_yield) >= M :
        return q_star_2, compute_variance(q_star_2, var_price_quantity, var_price, covariance_PQ_P), compute_expectation(q_star_2, mean_yield, mean_spot_price, sept_forward_price_quintale, covariance_price_yield), compute_lambda(q_star_2, var_price, covariance_PQ_P, sept_forward_price_quintale, mean_spot_price)
    else:
        return np.nan, np.nan, np.nan, np.nan
    
        
def compute_frontier(Range, covariance_price_yield, covariance_PQ_P, var_price, var_price_quantity, mean_yield, mean_spot_price, sept_forward_price_quintale):
    """
    Compute the Pareto frontier for a given set of parameters.
    """
    q_values = []
    variances = []
    expectations = []

    for i in range(1, Range):
        M_value = i
        q_star, variance, expectation, lambda_k = compute_solution(M_value, covariance_price_yield, covariance_PQ_P, var_price, var_price_quantity, mean_yield, mean_spot_price, sept_forward_price_quintale)
        q_values.append(q_star)
        variances.append(variance)
        expectations.append(expectation)

    return q_values, variances, expectations

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
    plt.savefig(f"{folder}PF.png", dpi=300)
    plt.show()


def plot_q_values(q_values_list, e_Q, labels=None, folder="pareto_frontiers"):
    """
    Plot q values for one or more Pareto frontiers, starting from 100 before the first deviation from the initial value.
    Adds a dashed indicator on the x-axis where the plot starts.

    Args:
        q_values_list (list of lists): Each element is a list/array of q values for a frontier.
        labels (list of str, optional): Labels for each q_values array.
    """

    plt.figure(figsize=(10, 6))
    if not isinstance(q_values_list[0], (list, tuple, pd.Series, np.ndarray)):
        q_values_list = [q_values_list]

    plot_start = float('inf')
    for idx, q_values in enumerate(q_values_list):
        q_values = np.array(q_values)
        q_values = q_values
        initial_value = q_values[0]
        # Find the first index where q_values deviates from the initial value
        deviation_idx = np.argmax(q_values != initial_value)
        if np.all(q_values == initial_value):
            # No deviation, plot the whole array
            plot_start = 0
        else:
            if plot_start > deviation_idx-100:
                plot_start = max(0,deviation_idx-100)

    for idx, q_values in enumerate(q_values_list):
        label = labels[idx] if labels and idx < len(labels) else f"Frontier {idx+1}"
        plt.plot(range(plot_start, len(q_values)), q_values[plot_start:]/e_Q, label=label)

    plt.xlabel("Required expected profit ($)")
    plt.ylabel(r"$q^* / \mathbb{E}[Q]$ (%)")
    plt.title(r"Pareto Frontiers - $q^*$ Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder}q.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    yields = pd.read_csv("crop_yield.csv", low_memory=False)
    yields = yields[(yields['M-q706_cropSP'] < 10000) & (yields['M-q706_cropSP'] > 0)].copy()  # Remove outliers
    yields = yields[yields['C-q306_cropLarestAreaAcre'] > 0]
    crop_type = 'Improved'  # Change this to 'Improved' or 'Local' as needed
    yields = yields[yields['D-q409_varType']== crop_type]

    acres = yields['C-q306_cropLarestAreaAcre'].median()
    hectares =  acres / 2.47105  # Convert hectares to hectares

    # Convert yield per hectare to quintale per hectare
    yields['median_farm_yield_quintale'] = yields['L-tonPerHectare'] *10  #* hectares
    mean_yield = yields['median_farm_yield_quintale'].mean() # Mean yield of median farmer

    rupy_to_dollar = 0.012
    yields['dollar_spot_price'] = yields['M-q706_cropSP'] * rupy_to_dollar  # Convert to dollar
    mean_spot_price = yields['dollar_spot_price'].mean()

    yields['price_quantity'] = yields['dollar_spot_price'] * yields['median_farm_yield_quintale']  # Price times quantity

    # Five month (grow time) forward price in cwt
    sept_forward_price = 11
    conv_rate_cwt_to_quintale = 0.453592
    sept_forward_price_quintale = sept_forward_price / conv_rate_cwt_to_quintale

    # Load covariance matrices from CSV files
    cov_matrix = pd.read_csv("covariance_matrix_P_Q.csv", index_col=0)
    cov_matrix_PQ_P = pd.read_csv("covariance_matrix_PQ_P.csv", index_col=0)

    # Extract required quantities
    covariance_price_yield = cov_matrix.loc['Yield_' + crop_type, 'Price_' + crop_type]
    var_price = cov_matrix.loc['Price_' + crop_type,  'Price_' + crop_type]

    covariance_PQ_P = cov_matrix_PQ_P.loc['Price_Yield_' + crop_type,  'Price_' + crop_type]
    var_price_quantity = cov_matrix_PQ_P.loc['Price_Yield_' + crop_type, 'Price_Yield_' + crop_type]

    print(covariance_price_yield)
    print(covariance_PQ_P, var_price, mean_yield)


    q_star, variance, expectation, lambda_k = compute_solution(811.0719428989714, covariance_price_yield, covariance_PQ_P, var_price, var_price_quantity, mean_yield, mean_spot_price, mean_spot_price)
    print("q_star", q_star)
    print("Variance", variance)
    print("Expectation", expectation)


    q_values, variances, expectations = compute_frontier(2000, covariance_price_yield, covariance_PQ_P, var_price, var_price_quantity, mean_yield, mean_spot_price, sept_forward_price_quintale) 
    plot_frontier(expectations, variances)

    # print(q_values[0]/ mean_yield,  variances[0], expectations[0])
    # print(1, compute_variance(1, var_price_quantity, var_price, covariance_PQ_P), compute_expectation(1, mean_yield, mean_spot_price, sept_forward_price_quintale, covariance_price_yield))
    
    # Define a range of forward prices to analyze
    forward_prices = [(1+i/20)* mean_spot_price for i in range(-2, 3)]  # Forward prices from 1x to 5x the mean spot price
    frontiers_expectations = []
    frontiers_variances = []
    frontiers_q_values = []
    labels = []

    for i, fwd_price in enumerate(forward_prices):
        q_values, variances, expectations = compute_frontier( # Hybrid 870, Improved 1100
            1100, covariance_price_yield, covariance_PQ_P, var_price, var_price_quantity,
            mean_yield, mean_spot_price, fwd_price
        )
        frontiers_expectations.append(expectations)
        frontiers_variances.append(variances)
        frontiers_q_values.append(q_values)
        labels.append(f"Forward price = {(0.9 + i*0.05):.2f}E[P]")

    plot_frontier(frontiers_expectations, frontiers_variances, labels=labels, folder = "Figures/Improved")
    plot_q_values(frontiers_q_values, mean_yield, labels=labels, folder = "Figures/improved")