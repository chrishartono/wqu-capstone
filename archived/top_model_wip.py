#!/usr/bin/env python
# coding: utf-8

# # t-SNE, HMM for crypto 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sklearn
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import scipy.stats as stats
from sklearn.manifold import TSNE

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# In[2]:


import warnings
warnings.filterwarnings('ignore')


def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma

def replace_outliers(df):
        for column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)
            detected_outliers = df[outliers]

            clean_data_iqr = df[~outliers].fillna(0)

            mean_value = clean_data_iqr[column].mean()
            noise = np.random.normal(0, 0.1, len(detected_outliers))
            mean_value_with_noise = noise + mean_value

            df.loc[outliers, column] = mean_value_with_noise
            
        
            # Calculate the percentage of detected outliers
            percentage_detected = (len(detected_outliers) / len(df)) * 100
            print(f"Column '{column}': Detected outliers: {percentage_detected:.2f}%")

        return df

def delete_outliers(df):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)
        detected_outliers = df[outliers]

        clean_data_iqr = df[~outliers].fillna(0)

        # Calculate the percentage of detected outliers
        percentage_detected = (len(detected_outliers) / len(df)) * 100
        print(f"Column '{column}': Detected outliers: {percentage_detected:.2f}%")

        # If you want to delete outliers, simply update the DataFrame
        df = clean_data_iqr

    return df

# In[6]:

df = pd.read_parquet(r'C:\BOYAN LAB\wqu-capstone\dataset\binance_1h_ohlcv_2021-2025.parquet', engine='pyarrow')
df = df.set_index('date')

#%%
crypto_returns_norm = feature_normalize(df)
crypto_returns_norm = crypto_returns_norm.replace([np.inf, -np.inf], np.nan)

crypto_returns_norm =crypto_returns_norm.dropna(axis=1, how='any')



#%%
pca = PCA(n_components=0.95)
principalComponents = pca.fit_transform(crypto_returns_norm)


components = pca.components_
components
len(components)


#%%
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_ratio

# Get the feature names from your original DataFrame
feature_names = crypto_returns_norm.columns

# Create a DataFrame to store the feature names and weights for each principal component
components_df = pd.DataFrame(components, columns=feature_names)
components_df = components_df.fillna(0)

components_df_sorted = components_df.apply(lambda row: row.abs().nlargest(6), axis=1)

# Print the sorted features for each principal component
for i, component_features in enumerate(components_df_sorted.iterrows()):
    print(f"Principal Component {i+1}:")
    for feature, weight in component_features[1].items():
        print(f"Feature '{feature}': {weight:.4f}")
    print()


# In[31]:


components_df_sorted

# PCA (or any other linear method) is not a good choice. We need 11 components to achieve 95% explained variance.  

# %%

def compute_rolling_features(df, asset_col, window=24):
    df = df.copy()
    df['Returns'] = df[asset_col].pct_change()
    df['Rolling Returns'] = df['Returns'].rolling(window=window).mean()
    df['Rolling Volatility'] = df['Returns'].rolling(window=window).std()
    df.dropna(inplace=True)
    return df
# %%
volatility_threshold_low = 0.10
volatility_threshold_high = 0.20

def assign_regime(row):
    if row['Rolling Returns'] > 0 and row['Rolling Volatility'] > volatility_threshold_high:
        return 'Bullish/High Volatility'
    elif row['Rolling Returns'] > 0 and row['Rolling Volatility'] <= volatility_threshold_low:
        return 'Bullish/Low Volatility'
    elif row['Rolling Returns'] <= 0 and row['Rolling Volatility'] > volatility_threshold_high:
        return 'Bearish/High Volatility'
    elif row['Rolling Returns'] <= 0 and row['Rolling Volatility'] <= volatility_threshold_low:
        return 'Bearish/Low Volatility'
    else:
        return 'Neutral'

# %%
df['Regime'] = df.apply(assign_regime, axis=1)

# %%
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# Features to model
features = df[['Rolling Returns', 'Rolling Volatility']].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

model = GaussianHMM(n_components=4, covariance_type='full', n_iter=1000)
model.fit(features_scaled)

# Predict regimes
df['HMM Regime'] = model.predict(features_scaled)

#%%
btc_df = compute_rolling_features(df, 'close_btc-usdt')

print(btc_df.head())

# %%
import pandas as pd
import numpy as np

# Read parquet file
df = pd.read_parquet(r'C:\BOYAN LAB\wqu-capstone\dataset\binance_1h_ohlcv_2021-2025.parquet', engine='pyarrow')

# Check that the column exists and is numeric
col = 'close_btc-usdt'
assert col in df.columns, f"{col} not found in DataFrame columns."
df[col] = pd.to_numeric(df[col], errors='coerce')

def compute_rolling_features(df, asset_col, window=24):
    df = df.copy()
    df['Returns'] = df[asset_col].pct_change()
    df['Rolling Returns'] = df['Returns'].rolling(window=window).mean()
    df['Rolling Volatility'] = df['Returns'].rolling(window=window).std()
    df.dropna(inplace=True)
    return df

btc_df = compute_rolling_features(df, col)
print(btc_df.head())

# %%
import pandas as pd
import numpy as np

# Read your data (specify engine to avoid any issues)
df = pd.read_parquet(r'C:\BOYAN LAB\wqu-capstone\dataset\binance_1h_ohlcv_2021-2025.parquet', engine='pyarrow')

# Choose your asset column (e.g., for BTC-USDT)
asset_col = 'close_btc-usdt'
assert asset_col in df.columns

# Convert to float32 for safer math if needed
df[asset_col] = pd.to_numeric(df[asset_col], errors='coerce').astype('float32')

def compute_rolling_features(df, asset_col, window=24):
    df = df.copy()
    df['Returns'] = df[asset_col].pct_change()
    df['Rolling Returns'] = df['Returns'].rolling(window=window).mean()
    df['Rolling Volatility'] = df['Returns'].rolling(window=window).std()
    df.dropna(inplace=True)
    return df

btc_df = compute_rolling_features(df, asset_col, 4)

print(btc_df.head())


# %%
##################
#HMM
##################

#Other Features?  
features = btc_df[['Rolling Returns', 'Rolling Volatility']].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

model = GaussianHMM(n_components=5, covariance_type='full', n_iter=10000)
model.fit(features_scaled)

btc_df['HMM Regime'] = model.predict(features_scaled)

# %%
regime_names = {
    0: "Bullish/High Volatility",
    1: "Bullish/Low Volatility",
    2: "Bearish/High Volatility",
    3: "Bearish/Low Volatility",
    4: "Neutral"
}


plt.figure(figsize=(10, 12))

for regime in sorted(btc_df['HMM Regime'].unique()):
    plt.scatter(btc_df[btc_df['HMM Regime'] == regime].index, 
             btc_df[btc_df['HMM Regime'] == regime][asset_col],
        s=6,
        label=regime_names.get(regime, f"Regime {regime}")
    )

plt.legend(title="Market Regime")
plt.title("Market Regimes Detected by HMM")
plt.show()

# %%
# %%
# feature selection appraoch with Bayesian Network
# takes a lot of time

from pgmpy.estimators import HillClimbSearch, K2
from pgmpy.models import BayesianNetwork


df_sparse = btc_df[[asset_col,'Rolling Returns', 'Rolling Volatility']]

hc = HillClimbSearch(df_sparse)
best_model = hc.estimate(scoring_method=K2(df_sparse))

print(best_model.edges())


#%%
########################################################
# # t-SNE
########################################################


# In[ ]:

X_pre_tsne = btc_df[['Rolling Returns']]
X_tsne = feature_normalize(X_pre_tsne)
y_pre_tsne = btc_df[['HMM Regime']]


# check on lenghts
len(y_pre_tsne)
len(X_tsne)


# In[ ]:

def perform_tsne(X_data, y_data, perplexities, n_iter=1000, img_name_prefix='t-sne'):
    colors = sns.color_palette('tab20', n_colors=4)  # Choose a larger color palette

    for index, perplexity in enumerate(perplexities):
        # Perform t-SNE
        print('\nPerforming t-SNE with perplexity {} and {} iterations at max'.format(perplexity, n_iter))
        X_reduced = TSNE(verbose=2, perplexity=perplexity).fit_transform(X_data)
        print('Done..')

        # Prepare the data for seaborn
        print('Creating plot for this t-SNE visualization..')
        df = pd.DataFrame({'x': X_reduced[:, 0], 'y': X_reduced[:, 1], 'label': y_data})
        df['label'] = df['label'].astype(int)

        # Get the unique levels in the 'label' column
        unique_labels = df['label'].unique()

        # Create a dictionary mapping each level to a color
        color_dict = dict(zip(unique_labels, colors[:len(unique_labels)]))

        # Map the colors to the 'label' column
        df['color'] = df['label'].map(color_dict)

        # Draw the plot
        plt.figure(figsize=(14, 10))
        sns.scatterplot(data=df, x='x', y='y', hue='label', style='label', palette=color_dict, markers=True)
        plt.title("Perplexity: {} and Max_iter: {}".format(perplexity, n_iter))
        img_name = img_name_prefix + '_perp_{}_iter_{}.png'.format(perplexity, n_iter)
        print('Saving this plot as an image in the present working directory...')
        plt.savefig(img_name)
        plt.show()
        print('Done')


# In[35]:

perform_tsne(X_data = X_tsne,y_data=y_pre_tsne, perplexities =[2,5,10])

#X_reduced = TSNE(verbose=2, perplexity=20).fit_transform(X_tsne)
X_reduced = TSNE(n_components=1, verbose=2, perplexity=20).fit_transform(X_tsne)
perplexity, n_iter = 20,1000


#%%
colors = sns.color_palette('tab20', n_colors=4)  # Choose a larger color palette
# Prepare the data for seaborn
print('Creating plot for this t-SNE visualization..')
df = pd.DataFrame({'x': X_reduced[:, 0], 'y': X_reduced[:, 1], 'label': y_pre_tsne})
df['label'] = df['label'].astype(int)

# Get the unique levels in the 'label' column
unique_labels = df['label'].unique()

# Create a dictionary mapping each level to a color
color_dict = dict(zip(unique_labels, colors[:len(unique_labels)]))

# Map the colors to the 'label' column
df['color'] = df['label'].map(color_dict)


# Not surprisingly there is one clear class that can be separated nicely (if not perfectly) from the rest and that is label 4 (the majority of which are prices after 2019) 

#%%
def perform_tsne(X_data, y_data, perplexities, n_iter=1000, img_name_prefix='t-sne'):
    colors = sns.color_palette('tab20', n_colors=4)  # Choose a larger color palette

    # Ensure X_data and y_data have matching lengths
    assert len(X_data) == len(y_data), "X_data and y_data must have the same number of rows."

    for index, perplexity in enumerate(perplexities):
        # Perform t-SNE with 2 components
        print('\nPerforming t-SNE with perplexity {} and {} iterations at max'.format(perplexity, n_iter))
        X_reduced = TSNE(n_components=2, verbose=2, perplexity=perplexity, n_iter=n_iter).fit_transform(X_data)
        print('Done..')

        # Check the shape of X_reduced
        print(f"Shape of X_reduced: {X_reduced.shape}")

        # Prepare the data for seaborn
        print('Creating plot for this t-SNE visualization..')
        df = pd.DataFrame({'x': X_reduced[:, 0], 'y': X_reduced[:, 1], 'label': y_data})
        df['label'] = df['label'].astype(int)

        # Get the unique levels in the 'label' column
        unique_labels = df['label'].unique()

        # Create a dictionary mapping each level to a color
        color_dict = dict(zip(unique_labels, colors[:len(unique_labels)]))

        # Map the colors to the 'label' column
        df['color'] = df['label'].map(color_dict)

        # Draw the plot
        plt.figure(figsize=(14, 10))
        sns.scatterplot(data=df, x='x', y='y', hue='label', style='label', palette=color_dict, markers=True)
        plt.title("Perplexity: {} and Max_iter: {}".format(perplexity, n_iter))
        img_name = img_name_prefix + '_perp_{}_iter_{}.png'.format(perplexity, n_iter)
        print('Saving this plot as an image in the present working directory...')
        plt.savefig(img_name)
        plt.show()
        print('Done')

X_reduced = TSNE(n_components=2, verbose=2, perplexity=perplexity).fit_transform(X_data)
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 1: Normalize the data
scaler = StandardScaler()
crypto_returns_scaled = scaler.fit_transform(crypto_data)

# Step 2: Perform t-SNE
print("Performing t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=1)
crypto_tsne = tsne.fit_transform(crypto_returns_scaled)

# Step 3: Cluster the reduced data
print("Clustering the t-SNE results...")
kmeans = KMeans(n_clusters=4, random_state=42)  # Adjust the number of clusters as needed
clusters = kmeans.fit_predict(crypto_tsne)

# Step 4: Visualize the regimes
print("Visualizing the regimes...")
plt.figure(figsize=(12, 8))
sns.scatterplot(x=crypto_tsne[:, 0], y=crypto_tsne[:, 1], hue=clusters, palette="tab10", s=50)
plt.title("t-SNE Visualization with Regimes (Clusters)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Regimes")
plt.grid(True)
plt.show()