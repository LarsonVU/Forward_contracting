import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filesuffix = "_P_Q_neg_PQ" # suffices _P_Q and _PQ_P are also required
COVARIANCE_PRICE_YIELD =True
NEGATIVE = True

yields = pd.read_csv("crop_yield.csv", low_memory=False)
yields = yields[(yields['M-q706_cropSP'] < 10000) & (yields['M-q706_cropSP'] > 0)].copy()  # Remove outliers
yields = yields[yields['C-q306_cropLarestAreaAcre'] > 0]

acres = yields['C-q306_cropLarestAreaAcre'].median()
hectares =  acres / 2.47105  # Convert hectares to hectares

# Convert yield per hectare to quintale per hectare
yields['median_farm_yield_quintale'] = yields['L-tonPerHectare'] *10 

yields['time'] = pd.to_datetime(yields['L-q601_harvestDate'], errors='coerce', dayfirst=True)

# Filter out rows with invalid or missing time values
yields = yields.dropna(subset=['time'])

rupy_to_dollar = 0.012
yields['dollar_spot_price'] = yields['M-q706_cropSP'] * rupy_to_dollar  # Convert to dollar

yields['price_quantity'] = yields['dollar_spot_price'] * yields['median_farm_yield_quintale']  # Price times quantity

yields_improved = yields[yields['D-q409_varType']== 'Improved']
yields_hybrid = yields[yields['D-q409_varType']== 'Hybrid']
yields_local = yields[yields['D-q409_varType']== 'Local']


mean_price_improved =  yields_improved['dollar_spot_price'].mean()
mean_price_hybrid =  yields_hybrid['dollar_spot_price'].mean()
mean_price_local =  yields_local['dollar_spot_price'].mean()

print("Price improved", mean_price_improved)
print("Price hybrid", mean_price_hybrid)
print("Price local", mean_price_local)

mean_yield_improved = yields_improved['median_farm_yield_quintale'].mean()
mean_yield_hybrid = yields_hybrid['median_farm_yield_quintale'].mean()
mean_yield_local = yields_local['median_farm_yield_quintale'].mean()

print("Mean yield improved", mean_yield_improved)
print("Mean yield hybrid", mean_yield_hybrid)
print("Mean yield local", mean_yield_local)


sept_forward_price_improved = 11
sept_forward_price_hybrid = 10
sept_forward_price_local = 9

conv_rate_cwt_to_quintale = 0.453592

mean_prices = [mean_price_improved, mean_price_hybrid, mean_price_local]
mean_yields = [mean_yield_improved, mean_yield_hybrid, mean_yield_local]
forward_prices = [sept_forward_price_improved, sept_forward_price_hybrid, sept_forward_price_local]
forward_prices = [f/ conv_rate_cwt_to_quintale for f in forward_prices]



full_yields = []
yields['crop'] = yields['D-q409_varType']
yields_all = yields.copy()
crop_names = ['Improved', 'Hybrid','Local']

# Helper: get neighbors by hierarchical location
def get_neighbors_with_time(source_row, target_df, k=3):
    levels = ['A-q105_village', 'A-q104_subDistrict', 'A-q103_district', 'A-q102_state']
    for level in levels:
        # Filter by same time (same month) and location
        matches = target_df[
            (target_df[level] == source_row[level]) &
            (target_df['time'].dt.to_period('M') == source_row['time'].to_period('M'))
        ]
        if not matches.empty:
            return matches.nsmallest(k, 'median_farm_yield_quintale')
    
    # Fallback: Closest in time (ignores location)
    target_df = target_df.copy()
    target_df['timediff'] = (target_df['time'] - source_row['time']).abs()
    return target_df.nsmallest(k, 'timediff')

for idx, row in yields_all.iterrows():
    crop = row['crop']
    y_obs = row['median_farm_yield_quintale']
    
    # Get other two crops
    other_crops = [c for c in crop_names if c != crop]
    df1 = yields_all[yields_all['crop'] == other_crops[0]]
    df2 = yields_all[yields_all['crop'] == other_crops[1]]

    # Nearest neighbors by space+time
    nn1 = get_neighbors_with_time(row, df1)
    nn2 = get_neighbors_with_time(row, df2)

    if nn1.empty or nn2.empty:
        continue  # skip if can't estimate both

    y1 = nn1['median_farm_yield_quintale'].mean()
    y2 = nn2['median_farm_yield_quintale'].mean()

    p_obs = row['dollar_spot_price']
    p1 = nn1['dollar_spot_price'].mean()
    p2 = nn2['dollar_spot_price'].mean()

    # Construct vector for yields and prices
    price_yield_vector = {'Improved': None, 'Hybrid': None, 'Local': None}
    price_vector = {'Improved': None, 'Hybrid': None, 'Local': None}
    yield_vector = {'Improved': y_obs, 'Hybrid': y1, 'Local': y2}
    yield_vector[crop] = y_obs
    yield_vector[other_crops[0]] = y1
    yield_vector[other_crops[1]] = y2
    price_yield_vector[crop] = y_obs * p_obs
    price_yield_vector[other_crops[0]] = y1 * p1
    price_yield_vector[other_crops[1]] = y2 * p2
    price_vector[crop] = p_obs
    price_vector[other_crops[0]] = p1
    price_vector[other_crops[1]] = p2

    # Combine yields and prices into a single vector
    if COVARIANCE_PRICE_YIELD:
        if NEGATIVE:
            full_yields.append([price_yield_vector[c] for c in crop_names] + [-price_vector[c] if price_vector[c] is not None else None  for c in crop_names])
        else:
             full_yields.append([price_yield_vector[c] for c in crop_names] + [price_vector[c] if price_vector[c] is not None else None  for c in crop_names])
    else:
        full_yields.append([yield_vector[c] for c in crop_names] + [price_vector[c] for c in crop_names])

# Compute covariance matrix
yield_matrix = np.array(full_yields, dtype=float)
yield_matrix = yield_matrix[~np.isnan(yield_matrix).any(axis=1)]  # Remove rows with NaN
cov_matrix = np.cov(yield_matrix.T)
if COVARIANCE_PRICE_YIELD:
    # If we are using price-yield covariance, we have 6 columns
    column_names = ['Price_Yield_Improved', 'Price_Yield_Hybrid', 'Price_Yield_Local', 'Price_Improved', 'Price_Hybrid', 'Price_Local']
else:
    column_names = ['Yield_Improved', 'Yield_Hybrid', 'Yield_Local', 'Price_Improved', 'Price_Hybrid', 'Price_Local']
cov_df = pd.DataFrame(cov_matrix, index=column_names, columns=column_names)

print("Estimated covariance matrix of yields (using KNN by space and time):")
print(cov_df)

# Save covariance matrix to a CSV file for use in other scripts
cov_df.to_csv("covariance_matrix"+filesuffix +".csv")

std = np.sqrt(np.diag(cov_df))
cor_df = cov_df / np.outer(std, std)
cor_df = pd.DataFrame(cor_df, index=cov_df.index, columns=cov_df.columns)

print(cor_df)
cor_df.to_csv("corr_matrix"+filesuffix +".csv")