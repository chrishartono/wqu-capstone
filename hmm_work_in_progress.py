######################################################
# # Forecasting prices with HMM
######################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sklearn

class HMM(object):
    # Implements discrete 1-st order Hidden Markov Model 

    def __init__(self, tolerance = 1e-6, max_iterations=10000, scaling=True):
        self.tolerance=tolerance
        self.max_iter = max_iterations
        self.scaling = scaling

    def HMMfwd(self, a, b, o, pi):
        # Implements HMM Forward algorithm
        N = np.shape(b)[0]
        T = np.shape(o)[0]

        alpha = np.zeros((N, T))
        # Initialize first column with observation values
        alpha[:, 0] = pi * b[:, o[0]]
        c = np.ones((T))

        if self.scaling:
            c[0] = 1.0 / np.sum(alpha[:, 0])
            alpha[:, 0] = alpha[:, 0] * c[0]

            for t in range(1, T):
                c[t] = 0
                for i in range(N):
                    alpha[i, t] = b[i, o[t]] * np.sum(alpha[:, t - 1] * a[:, i])
                c[t] = 1.0 / np.sum(alpha[:, t])
                alpha[:, t] = alpha[:, t] * c[t]

        else:
            for t in range(1, T):
                for i in range(N):
                    alpha[i, t] = b[i, o[t]] * np.sum(alpha[:, t - 1] * a[:, i])

        return alpha, c

    def HMMbwd(self, a, b, o, c):
        # Implements HMM Backward algorithm
    
        N = np.shape(b)[0]
        T = np.shape(o)[0]
    
        beta = np.zeros((N,T))
        # initialise last row with scaling c
        beta[:,T-1] = c[T-1]
    
        for t in range(T-2,-1,-1):
            for i in range(N):
                beta[i,t] = np.sum(b[:,o[t+1]] * beta[:,t+1] * a[i,:])
            # scale beta by the same value as a
            beta[:,t]=beta[:,t]*c[t]

        return beta

    def HMMViterbi(self, a, b, o, pi):
        # Implements HMM Viterbi algorithm        
        
        N = np.shape(b)[0]
        T = np.shape(o)[0]
    
        path = np.zeros(T)
        delta = np.zeros((N,T))
        phi = np.zeros((N,T))
    
        delta[:,0] = pi * b[:,o[0]]
        phi[:,0] = 0
    
        for t in range(1,T):
            for i in range(N):
                delta[i,t] = np.max(delta[:,t-1]*a[:,i])*b[i,o[t]]
                phi[i,t] = np.argmax(delta[:,t-1]*a[:,i])
    
        path[T-1] = np.argmax(delta[:,T-1])
        for t in range(T-2,-1,-1):
            path[t] = phi[int(path[t+1]),t+1]
    
        return path,delta, phi

 
    def HMMBaumWelch(self, o, N, dirichlet=False, verbose=False, rand_seed=1):
        # Implements HMM Baum-Welch algorithm        
        T = np.shape(o)[0]
        M = int(max(o)) + 1  # now all hist time-series will contain all observation vals, but we have to provide for all

        digamma = np.zeros((1, N, T))
        

        # Initialization can be done either using the Dirichlet distribution (all randoms sum to one) 
        # or using approximates uniforms from matrix sizes
        if dirichlet:
            pi = np.random.dirichlet(np.ones(N), size=1)
            a = np.random.dirichlet(np.ones(N), size=N)
            b = np.random.dirichlet(np.ones(M), size=N)
        else:
            pi_randomizer = np.random.dirichlet(np.ones(N), size=1) / 100
            pi = 1.0 / N * np.ones(N) - pi_randomizer

            a_randomizer = np.random.dirichlet(np.ones(N), size=N) / 100
            a = 1.0 / N * np.ones([N, N]) - a_randomizer

            b_randomizer = np.random.dirichlet(np.ones(M), size=N) / 100
            b = 1.0 / M * np.ones([N, M]) - b_randomizer

        error = self.tolerance + 10
        itter = 0
        while error > self.tolerance and itter < self.max_iter:

            prev_a = a.copy()
            prev_b = b.copy()

            # Estimate model parameters
            alpha, c = self.HMMfwd(a, b, o, pi)
            beta = self.HMMbwd(a, b, o, c)

            for t in range(T - 1):
                for i in range(N):
                    for j in range(N):
                        digamma[i, j, t] = alpha[i, t] * a[i, j] * b[j, o[t + 1]] * beta[j, t + 1]
                digamma[:, :, t] /= np.sum(digamma[:, :, t])

            for i in range(N):
                for j in range(N):
                    digamma[i, j, T - 1] = alpha[i, T - 1] * a[i, j]
            digamma[:, :, T - 1] /= np.sum(digamma[:, :, T - 1])

            # Maximize parameter expectation
            for i in range(N):
                pi[i] = np.sum(digamma[i, :, 0])
                for j in range(N):
                    a[i, j] = np.sum(digamma[i, j, :T - 1]) / np.sum(digamma[i, :, :T - 1])
                for k in range(M):
                    filter_vals = (o == k).nonzero()
                    b[i, k] = np.sum(digamma[i, :, filter_vals]) / np.sum(digamma[i, :, :])

            error = (np.abs(a - prev_a)).max() + (np.abs(b - prev_b)).max()
            itter += 1

            if verbose:
                print("Iteration: ", itter, " error: ", error, "P(O|lambda): ", np.sum(alpha[:, T - 1]))

        return a, b, pi, alpha


# In[ ]:


np_hist_prices = crypto_data.values


# In[ ]:


crypto_data


# In[ ]:


def computeMoves(dataframe, holding_period=1):
    np_observations = dataframe.values
    moves = np_observations[holding_period:] - np_observations[:-holding_period]
    return moves


# In[ ]:


crypto_prices_moves = computeMoves(crypto_data)


# In[ ]:


crypto_prices_moves



hist_O = np.array(list(map(lambda x: 1 if x > 0 else (0 if x < 0 else 2), crypto_prices_moves)))


# In[ ]:


# Define a function to apply the logic element-wise to each column
def label_columns(arr):
    result = np.zeros_like(arr, dtype=int)
    result[arr > 0] = 1
    result[arr < 0] = 0
    result[arr == 0] = 2
    return result

# Apply the function to the entire array
moves_binarized = label_columns(crypto_prices_moves)


# In[ ]:


moves_binarized


# In[ ]:


moves_binarized_crypto = moves_binarized[:,0]


# In[ ]:


moves_binarized_rev = moves_binarized_crypto[::-1]


# In[ ]:


moves_binarized_rev.shape


# In[ ]:


hmm = HMM()
   
# Train the HMM using Baum-Welch algorithm
(a, b, pi_est, alpha_est) = hmm.HMMBaumWelch(moves_binarized_rev, 3, False, True)

# Use Viterbi to predict hidden states
(path, delta, phi) = hmm.HMMViterbi(a, b, moves_binarized_rev, pi_est)

path


# In[ ]:




import numpy as np

def computeMoves(dataarray, holding_period=1):
    moves = dataarray[holding_period:] - dataarray[:-holding_period]
    return moves

# Define a function to apply the logic element-wise to each column
def label_columns(arr):
    result = np.zeros_like(arr, dtype=int)
    result[arr > 0] = 1
    result[arr < 0] = 0
    result[arr == 0] = 2
    return result

# Apply the function to the entire array
moves_binarized = label_columns(crypto_prices_moves)

moves_binarized


# In[ ]:


# Assuming 'all_data' is a NumPy array with multiple columns
all_data = crypto_data[['bhc-usdt']].values  # Extracting the 'bhc-usdt' column as a NumPy array

hmm = HMM()

num_correct = 0.0
test_window = 6
N = len(all_data)
num_tests = N // test_window  # Use integer division (//) for the number of tests


# In[ ]:


all_data


# In[ ]:


n = 10


# In[ ]:


train_data = all_data[-n:-n-test_window:-1]
hist_moves = computeMoves(train_data, 1)  # Assuming holding_period=1 for daily moves
hist_O = label_columns(hist_moves)
hist_O = hist_O[::-1]
hist_O = hist_O.ravel()


# In[ ]:


hist_O


# In[ ]:


hist_O.ravel()


# In[ ]:


(a, b, pi_est, alpha_est) = hmm.HMMBaumWelch(hist_O, 2, False, False)


# In[ ]:


(a, b, pi_est, alpha_est)


# In[ ]:


(path, delta, phi) = hmm.HMMViterbi(a, b, hist_O, pi_est)


# In[ ]:


for n in range(1, N - test_window, test_window):
    train_data = all_data[-n:-n-test_window:-1]
    hist_moves = computeMoves(train_data, 1)  # Assuming holding_period=1 for daily moves
    hist_O = label_columns(hist_moves)
    hist_O = hist_O[::-1]
    hist_O = hist_O.ravel()
    # Assuming 2 hidden states and no Dirichlet randomization
    (a, b, pi_est, alpha_est) = hmm.HMMBaumWelch(hist_O, 2, False, False)
    (path, delta, phi) = hmm.HMMViterbi(a, b, hist_O, pi_est)
    path = path.astype(int)
    prediction_state = np.argmax(a[path[-1], :])
    prediction = np.argmax(b[prediction_state, :])
    
    # Assuming you want to predict based on the 'bhc-usdt' column
    price_difference = all_data[-n-test_window-1, 0] - all_data[-n-test_window, 0]
    
    if ((price_difference > 0 and prediction == 1) or 
        (price_difference < 0 and prediction == 0) or 
        (price_difference == 0 and prediction == 2)):
        num_correct += 1.0

print(num_correct / num_tests)


# In[ ]:


def HMMBaumWelch(self, o, N, dirichlet=False, verbose=False, rand_seed=1):
    # Implements HMM Baum-Welch algorithm        
    T = np.shape(o)[0]
    M = int(max(o)) + 1  # now all hist time-series will contain all observation vals, but we have to provide for all

    digamma = np.zeros((N, N, T))

    # Initialize A, B, and pi randomly, but so that they sum to one
    np.random.seed(rand_seed)

    # Initialization can be done either using the Dirichlet distribution (all randoms sum to one) 
    # or using approximates uniforms from matrix sizes
    if dirichlet:
        pi = np.random.dirichlet(np.ones(N), size=1)
        a = np.random.dirichlet(np.ones(N), size=N)
        b = np.random.dirichlet(np.ones(M), size=N)
    else:
        pi_randomizer = np.random.dirichlet(np.ones(N), size=1) / 100
        pi = 1.0 / N * np.ones(N) - pi_randomizer

        a_randomizer = np.random.dirichlet(np.ones(N), size=N) / 100
        a = 1.0 / N * np.ones([N, N]) - a_randomizer

        b_randomizer = np.random.dirichlet(np.ones(M), size=N) / 100
        b = 1.0 / M * np.ones([N, M]) - b_randomizer

    error = self.tolerance + 10
    itter = 0
    while error > self.tolerance and itter < self.max_iter:

        prev_a = a.copy()
        prev_b = b.copy()

        # Estimate model parameters
        alpha, c = self.HMMfwd(a, b, o, pi)
        beta = self.HMMbwd(a, b, o, c)

        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    digamma[i, j, t] = alpha[i, t] * a[i, j] * b[j, o[t + 1]] * beta[j, t + 1]
            digamma[:, :, t] /= np.sum(digamma[:, :, t])

        for i in range(N):
            for j in range(N):
                digamma[i, j, T - 1] = alpha[i, T - 1] * a[i, j]
        digamma[:, :, T - 1] /= np.sum(digamma[:, :, T - 1])

        # Maximize parameter expectation
        for i in range(N):
            pi[i] = np.sum(digamma[i, :, 0])
            for j in range(N):
                a[i, j] = np.sum(digamma[i, j, :T - 1]) / np.sum(digamma[i, :, :T - 1])
            for k in range(M):
                filter_vals = (o == k).nonzero()
                b[i, k] = np.sum(digamma[i, :, filter_vals]) / np.sum(digamma[i, :, :])

        error = (np.abs(a - prev_a)).max() + (np.abs(b - prev_b)).max()
        itter += 1

        if verbose:
            print("Iteration: ", itter, " error: ", error, "P(O|lambda): ", np.sum(alpha[:, T - 1]))

    return a, b, pi, alpha


# In[ ]:


all_data = crypto_data['bhc-usdt']


# In[ ]:


all_data[-n:-n-test_window:-1,:]

crypto_data['bhc-usdt']

print(path.dtype)


from sklearn.metrics import roc_curve, roc_auc_score



# Collect true labels (based on next-day market move)
true_labels = np.array([(1 if (all_data[-n-test_window-1,1]-all_data[-n-test_window,1]) > 0 else 0) for n in range(1, N-test_window, test_window)])

# Collect probabilities (scores) for the positive class (modify this part)
scores = []  # Replace with actual scores

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(true_labels, scores)

# Calculate AUC
auc = roc_auc_score(true_labels, scores)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


scores

#%%

class HMM(object):
    # Implements discrete 1-st order Hidden Markov Model 

    def __init__(self, tolerance = 1e-6, max_iterations=10000, scaling=True):
        self.tolerance=tolerance
        self.max_iter = max_iterations
        self.scaling = scaling

    def HMMfwd(self, a, b, o, pi):
        # Implements HMM Forward algorithm
    
        N = np.shape(b)[0]
        T = np.shape(o)[0]
    
        alpha = np.zeros((N,T))
        # initialise first column with observation values
        alpha[:,0] = pi*b[:,o[0]]
        c = np.ones((T))
        
        if self.scaling:
            
            c[0]=1.0/np.sum(alpha[:,0])
            alpha[:,0]=alpha[:,0]*c[0]
            
            for t in range(1,T):
                c[t]=0
                for i in range(N):
                    alpha[i,t] = b[i,o[t]] * np.sum(alpha[:,t-1] * a[:,i])
                c[t]=1.0/np.sum(alpha[:,t])
                alpha[:,t]=alpha[:,t]*c[t]

        else:
            for t in range(1,T):
                for i in range(N):
                    alpha[i,t] = b[i,o[t]] * np.sum(alpha[:,t-1] * a[:,i])
        
        return alpha, c

    def HMMbwd(self, a, b, o, c):
        # Implements HMM Backward algorithm
    
        N = np.shape(b)[0]
        T = np.shape(o)[0]
    
        beta = np.zeros((N,T))
        # initialise last row with scaling c
        beta[:,T-1] = c[T-1]
    
        for t in range(T-2,-1,-1):
            for i in range(N):
                beta[i,t] = np.sum(b[:,o[t+1]] * beta[:,t+1] * a[i,:])
            # scale beta by the same value as a
            beta[:,t]=beta[:,t]*c[t]

        return beta

    def HMMViterbi(self, a, b, o, pi):
        # Implements HMM Viterbi algorithm        
        
        N = np.shape(b)[0]
        T = np.shape(o)[0]
    
        path = np.zeros(T)
        delta = np.zeros((N,T))
        phi = np.zeros((N,T))
    
        delta[:,0] = pi * b[:,o[0]]
        phi[:,0] = 0
    
        for t in range(1,T):
            for i in range(N):
                delta[i,t] = np.max(delta[:,t-1]*a[:,i])*b[i,o[t]]
                phi[i,t] = np.argmax(delta[:,t-1]*a[:,i])
    
        path[T-1] = np.argmax(delta[:,T-1])
        for t in range(T-2,-1,-1):
            path[t] = phi[int(path[t+1]),t+1]
    
        return path,delta, phi

 
    def HMMBaumWelch(self, o, N, dirichlet=False, verbose=False, rand_seed=1):
        # Implements HMM Baum-Welch algorithm        
        
        T = np.shape(o)[0]

        M = int(max(o))+1 # now all hist time-series will contain all observation vals, but we have to provide for all

        digamma = np.zeros((N,N,T))

    
        # Initialise A, B and pi randomly, but so that they sum to one
        np.random.seed(rand_seed)
        
        # Initialisation can be done either using dirichlet distribution (all randoms sum to one) 
        # or using approximates uniforms from matrix sizes
        if dirichlet:
            pi = np.ndarray.flatten(np.random.dirichlet(np.ones(N),size=1))
            
            a = np.random.dirichlet(np.ones(N),size=N)
            
            b=np.random.dirichlet(np.ones(M),size=N)
        else:
            
            pi_randomizer = np.ndarray.flatten(np.random.dirichlet(np.ones(N),size=1))/100
            pi=1.0/N*np.ones(N)-pi_randomizer

            a_randomizer = np.random.dirichlet(np.ones(N),size=N)/100
            a=1.0/N*np.ones([N,N])-a_randomizer

            b_randomizer=np.random.dirichlet(np.ones(M),size=N)/100
            b = 1.0/M*np.ones([N,M])-b_randomizer

        
        error = self.tolerance+10
        itter = 0
        while ((error > self.tolerance) & (itter < self.max_iter)):   

            prev_a = a.copy()
            prev_b = b.copy()
    
            # Estimate model parameters
            alpha, c = self.HMMfwd(a, b, o, pi)
            beta = self.HMMbwd(a, b, o, c) 
    
            for t in range(T-1):
                for i in range(N):
                    for j in range(N):
                        digamma[i,j,t] = alpha[i,t]*a[i,j]*b[j,o[t+1]]*beta[j,t+1]
                digamma[:,:,t] /= np.sum(digamma[:,:,t])
    

            for i in range(N):
                for j in range(N):
                    digamma[i,j,T-1] = alpha[i,T-1]*a[i,j]
            digamma[:,:,T-1] /= np.sum(digamma[:,:,T-1])
    
            # Maximize parameter expectation
            for i in range(N):
                pi[i] = np.sum(digamma[i,:,0])
                for j in range(N):
                    a[i,j] = np.sum(digamma[i,j,:T-1])/np.sum(digamma[i,:,:T-1])
                for k in range(M):
                    filter_vals = (o==k).nonzero()
                    b[i,k] = np.sum(digamma[i,:,filter_vals])/np.sum(digamma[i,:,:])
    
            error = (np.abs(a-prev_a)).max() + (np.abs(b-prev_b)).max() 
            itter += 1            
            
            if verbose:            
                print ("Iteration: ", itter, " error: ", error, "P(O|lambda): ", np.sum(alpha[:,T-1]))
    
        return a, b, pi, alpha
        

def parseStockPrices(from_date, to_date, symbol):
    df = pdr.get_data_yahoo(symbol, start=from_date, end=to_date)

    np_hist_prices = np.empty(shape=[len(df), 7])
    for i, (_, row) in enumerate(df.iterrows()):
        np_hist_prices[i, 0] = row['Adj Close']
        np_hist_prices[i, 1] = row['Close']
        np_hist_prices[i, 2] = row.name.toordinal()
        np_hist_prices[i, 3] = row['High']
        np_hist_prices[i, 4] = row['Low']
        np_hist_prices[i, 5] = row['Open']
        np_hist_prices[i, 6] = row['Volume']

    return np_hist_prices
         
        
def calculateDailyMoves(hist_prices, holding_period):
    # calculate daily moves as absolute difference between close_(t+1) - close_(t)    

    assert holding_period > 0, "Holding period should be above zero"
    return (hist_prices[:-holding_period,1]-hist_prices[holding_period:,1])


# In[ ]:

hmm = HMM()

test_start_date = '2021-08-17'
test_end_date = '2023-08-15'
all_data = hist_prices = parseStockPrices(test_start_date, test_end_date, 'NVDA')
assert len(all_data)>0, "Houston, we've got a problem"


num_correct=0.0
test_window = 6
N=len(all_data)
num_tests=N/test_window
for n in range(1,N-test_window,test_window):
    train_data = all_data[-n:-n-test_window:-1,:]
    hist_moves = calculateDailyMoves(train_data,1)
    hist_O=np.array(list(map(lambda x: 1 if x>0 else (0 if x<0 else 2), hist_moves)))
    hist_O = hist_O[::-1]
    (a, b, pi_est, alpha_est) = hmm.HMMBaumWelch(hist_O, 2, False, False)
    (path, delta, phi)=hmm.HMMViterbi(a, b, hist_O, pi_est)
    path = path.astype(int)
    prediction_state=np.argmax(a[path[-1],:])
    prediction = np.argmax(b[prediction_state,:])
    if ((all_data[-n-test_window-1,1]-all_data[-n-test_window,1])>0 and prediction==1) or ((all_data[-n-test_window-1,1]-all_data[-n-test_window,1])<0 and prediction==0) or ((all_data[-n-test_window-1,1]-all_data[-n-test_window,1])==0 and prediction==2):
        num_correct+=1.0
print (num_correct/num_tests)

