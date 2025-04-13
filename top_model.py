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

# In[2]:


import warnings
warnings.filterwarnings('ignore')


def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma


# In[4]:


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


# In[5]:


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


crypto_data = pd.read_csv('H:/personal/capstone/binance_1h_2021-2025.csv', delimiter = ',' )


# In[7]:


crypto_data = crypto_data.set_index('date')
#%%
crypto_data.head()

#%%

crypto_returns = crypto_data.pct_change()
crypto_returns = crypto_returns[1:]
crypto_returns.describe()


# In[12]:




np.random.seed(0)
data1 = np.random.normal(loc=0, scale=1, size=100)

# Creating subplots for QQ plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# QQ plot for data1
stats.probplot(data1, dist="norm", plot=axes[0])
axes[0].set_title('QQ Plot - Normal Distribution')
axes[0].set_xlabel('Theoretical Quantiles')
axes[0].set_ylabel('Sample Quantiles')

# QQ plot for data2
stats.probplot(crypto_returns["bch-usdt"], dist="norm", plot=axes[1])
axes[1].set_title('BCH-USDT QQ Plot - Distribution with Heavy Tails')
axes[1].set_xlabel('Theoretical Quantiles')
axes[1].set_ylabel('Sample Quantiles')

plt.tight_layout()
plt.show()


# In[ ]:


vols = pd.DataFrame(crypto_returns[["bch-usdt"]].rolling(120).std()).rename(columns={"bch-usdt": "bch-usdt STD"})

# set figure size
plt.figure(figsize=(12, 5))

# plot using rolling average
sns.lineplot(
    x=crypto_returns.index,
    y="bch-usdt STD",
    data=vols,
    label="Bitcoin Cash 5 day standard deviation rolling avg",
)
plt.show()



#%%
crypto_returns_norm = feature_normalize(crypto_returns)

crypto_returns_matrix = crypto_returns_norm.corr()
crypto_returns_matrix


#%%
#heat map
subset_data = crypto_returns.iloc[:20000, :20]
corr_matrix = subset_data.corr()

plt.figure(figsize=(12, 6)) 
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Pairwise Correlation Heatmap (Subset of Data)')
plt.show()


# In[ ]:
# Calculating spreads
#crypto_returns_norm['Spread'] = crypto_returns_norm['1inch-usdt'] - crypto_returns_norm['bch-usdt']


# # PCA

# In[23]:


crypto_returns_norm = crypto_returns_norm.dropna()


# In[24]:


pca = PCA(n_components=0.95)
principalComponents = pca.fit_transform(crypto_returns_norm)



components = pca.components_
components
len(components)



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


# PCA (or any other linear method) is not a good choice. We need 58 components to achieve 95% explained variance.  

########################################################
# # t-SNE
########################################################
Q1 = crypto_data["bch-usdt"].quantile(0.25)
Q2 = crypto_data["bch-usdt"].quantile(0.50)
Q3 = crypto_data["bch-usdt"].quantile(0.75)

# Define the quantile intervals and labels
quantile_intervals = [float('-inf'), Q1, Q2, Q3, float('inf')]
labels = [1, 2, 3, 4]

# Use pd.cut() to categorize the data based on the intervals and labels
crypto_data['Quantile Labels'] = pd.cut(crypto_data["bch-usdt"], bins=quantile_intervals, labels=labels, include_lowest=True)




plt.boxplot(crypto_data["bch-usdt"])
plt.title('Box Plot of Skewed Distribution')
plt.ylabel('Price')
plt.show()



# In[ ]:


# Define neutral colors for each quantile label
quantile_colors = ['green', 'blue', 'purple', 'silver']  # Add more colors if needed


# Create a dictionary to store quantile label names and their corresponding colors
quantile_legend_mapping = {}
for label, color in zip(crypto_data['Quantile Labels'].unique(), quantile_colors):
    quantile_legend_mapping[f'Quantile {label}'] = color

plt.figure(figsize=(12, 6))

for label, color in quantile_legend_mapping.items():
    label_data = crypto_data[crypto_data['Quantile Labels'] == int(label.split()[1])]
    plt.scatter(label_data.index, label_data["bch-usdt"], s=20, marker='o', label=label, color=color)

# Sort legend entries based on quantile label names
sorted_legend = [label for label, _ in sorted(quantile_legend_mapping.items(), key=lambda x: int(x[0].split()[1]))]
plt.legend(sorted_legend)

plt.title('Bitcoin Cash by Quantile')
plt.xlabel('Index')
plt.ylabel('Bitcoin cash')
plt.show()


# In[ ]:

crypto_returns = crypto_returns.dropna()
X_pre_tsne = crypto_returns[["bch-usdt"]]
X_tsne = feature_normalize(X_pre_tsne)
y_pre_tsne = crypto_data['Quantile Labels']


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
# ## Running regressions to determine relevant factors  
import statsmodels.api as sm


X = crypto_returns_norm.drop(columns=['bch-usdt'])

# Add a constant term to the predictor matrix (required by statsmodels)
X = sm.add_constant(X)

# Define the dependent variable
y = crypto_returns_norm['bch-usdt']

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the regression summary
print(model.summary())


# Linear regression is in fact not recommended as the correlations are obviously not linear. This can be seen in the next plots

sns.scatterplot(x=crypto_returns_norm['bch-usdt'],y=crypto_returns_norm['1inch-usdt']) 

#%%

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
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Step 1: Calculate rolling returns and volatility
rolling_window = 20  # Example: 20 periods
crypto_returns['Rolling Returns'] = crypto_returns['bch-usdt'].rolling(rolling_window).mean()
crypto_returns['Rolling Volatility'] = crypto_returns['bch-usdt'].rolling(rolling_window).std()

# Drop NaN values caused by rolling calculations
crypto_returns = crypto_returns.dropna()

# Step 2: Define thresholds for volatility
volatility_threshold_high = crypto_returns['Rolling Volatility'].quantile(0.75)
volatility_threshold_low = crypto_returns['Rolling Volatility'].quantile(0.25)

# Step 3: Assign regimes
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

crypto_returns['Regime'] = crypto_returns.apply(assign_regime, axis=1)

# Step 4: Normalize the data for t-SNE
scaler = StandardScaler()
crypto_returns_scaled = scaler.fit_transform(crypto_returns[['Rolling Returns', 'Rolling Volatility']])

# Step 5: Perform t-SNE
print("Performing t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=1)
crypto_tsne = tsne.fit_transform(crypto_returns_scaled)

# Step 6: Visualize the regimes
print("Visualizing the regimes...")
plt.figure(figsize=(12, 8))
sns.scatterplot(x=crypto_tsne[:, 0], y=crypto_tsne[:, 1], hue=crypto_returns['Regime'], palette="tab10", s=50)
plt.title("t-SNE Visualization with Regimes")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Regimes")
plt.grid(True)
plt.show()
# %%
