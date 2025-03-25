import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
import json
import time

from isochrones import get_ichrone
from sbi import analysis
import corner
import pandas as pd
from sbi.utils import process_prior
from sbi import utils
import os

#from sbi.inference import SNPE as method # paper
from sbi.inference import SNLE as method
#from sbi.inference import SNRE as method
    

######### Define simulator

tracks = get_ichrone('mist', tracks=True)


def binary_color_mag_isochrones(m1, q, age, fe_h, log_dist):
    # isochrones.py needs log10(Age [yr]). 
    # Our age is in Gyr, so we take log10(age * 10^9) = log10(age) + 9
    dist = np.float64(10**log_dist)
    properties = tracks.generate_binary(m1, q * m1, np.log10(age) + 9, fe_h, distance=dist, bands=["G", "BP", "RP"])
    
    #g_mag = properties.G_mag.values[0]
    bp_mag = properties.BP_mag.values[0]
    rp_mag = properties.RP_mag.values[0]
    dist = np.array(dist)
    g_mag = properties.G_mag.values[0] - 5* np.log10(dist) + 5

    return np.array([g_mag, bp_mag, rp_mag, dist]).T

def simulator(theta):
    return torch.tensor(binary_color_mag_isochrones(*theta))


############ Set priors.

from torch.distributions import (Uniform, Exponential, Normal)
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from torch import tensor as tt

#from sbi.inference import simulate_for_sbi
from sbi.utils.user_input_checks import prepare_for_sbi


age_bounds = [0.01,11.0]            # Gyr
mass_bounds = [0.08, 10.0]           # M1
metallicity_bounds = [-2.0, 0.6]    # [Fe/H]
distance_bounds = [100.0, 1000.0]     # distance


num_simulations = 100_000
num_samples = 2_000

selec_prior = 1

if selec_prior == 1:

    prior = [
    TransformedDistribution(Exponential(rate=tt([2.3])), \
                            AffineTransform(loc=tt([mass_bounds[0]]), scale=tt([mass_bounds[1]]))),                   # M1
    Uniform(tt([0.]), tt([1.])),                                                                                      # q
    Uniform(tt([age_bounds[0]]), tt([age_bounds[1]])),                                                                # age
    TransformedDistribution(Normal(loc=tt([0.0]), scale=tt([0.5])), \
                            AffineTransform(loc=tt([metallicity_bounds[0]]), scale=tt([metallicity_bounds[1]]))),     # [Fe/H]
    Uniform(tt([np.log10(distance_bounds[0])]), tt([np.log10(distance_bounds[1])]))                                   # log10(dist)
    ]

elif selec_prior == 2:
    bounds = np.array([
    [mass_bounds[0],  mass_bounds[1]],                           # M1
    [0,      1],                                                 # q
    [age_bounds[0],   age_bounds[1]],                            # (Gyr)
    [metallicity_bounds[0], metallicity_bounds[1]],              # metallicity
    [np.log10(distance_bounds[0]), np.log10(distance_bounds[1])] # log10(distance)
    ])

    bounds = torch.tensor(bounds)
    prior = utils.BoxUniform(low=bounds.T[0], high=bounds.T[1])


sbi_simulator, sbi_prior = prepare_for_sbi(simulator, prior)
inference = method(sbi_prior)


####### Generate the simulations. 
### We do this ourselves (instead of using simulate_for_sbi) because if we don't then many will be NaNs and we end up with fewer simulations than we want.

def simulate_for_sbi_strict(simulator, proposal, num_simulations, max_trials=np.inf):
    num_trials, num_simulated, theta, x = (0, 0, [], [])

    with tqdm(total=num_simulations) as pbar:
    
        while num_simulated < num_simulations:
            time.sleep(0.1)

            N = num_simulations - num_simulated
            print(f"Running {N} simulations")
            _theta = proposal.sample((N, ))
            _x = simulator(_theta)
            _x = _x.squeeze(1)
            keep = np.all(np.isfinite(_x).numpy(), axis=1)
            
            theta.extend(np.array(_theta[keep]))
            x.extend(np.array(_x[keep]))

            num_trials += 1
            num_simulated += sum(keep)

            pbar.update(sum(keep))

            if num_trials > max_trials:
                print(f"Warning: exceeding max trials ({max_trials}) with {num_simulated} / {num_simulations} simulations")
                break
    
    theta = torch.tensor(np.vstack(theta))
    x = torch.tensor(np.vstack(x))
    return (theta, x)


# Generate the posterior file
posterior_path = "./data/train_posterior.pkl"
directory = os.path.dirname(posterior_path)

if not os.path.exists(directory):
    os.makedirs(directory)

if os.path.exists(posterior_path):
    print(f"Pre-loading posterior from {posterior_path}")
    with open(posterior_path, "rb") as fp:
        posterior, (theta, x) = pickle.load(fp)
else:
    theta, x = simulate_for_sbi_strict(sbi_simulator, sbi_prior, num_simulations)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator, sample_with='mcmc', mcmc_parameters={"num_chains":10,"thin":3})
    
    with open(posterior_path, "wb") as fp:
        pickle.dump((posterior, (theta, x)), fp)
        
    print(f"Saved posterior to {posterior_path}")



########### Estimations per star
    
not_now = False
if not_now:

    cluster_estimations = {}

    # data from Artem
    df = pd.read_csv('./data/binary_stars.csv')
    #df = df.sample(n=50)
    df = df[0:5000]
    n_sample = 1

    df['distance'] = df['Dist50'] * 1000 # pc
    print('len members=%s'%len(df))

    df['log_dist'] = np.log10(df['distance'])
    df['g_mag'] = df['Gmag'] - 5*df['log_dist'] + 5


    observation_per_cluster = df[['g_mag','BPmag','RPmag','distance']].to_numpy()
    observation_per_cluster = torch.from_numpy(observation_per_cluster)
    print(observation_per_cluster.shape)
    #observation_per_cluster = observation_per_cluster[:-7,:] # select some stars
    print(observation_per_cluster.shape)


    num_injections = len(observation_per_cluster)

    #_, L = sbi_prior.sample((1, )).shape
    L = 5 # number of parameters


    ##################
    # all_samples = np.empty((num_injections, num_samples, L))

    # for i, obs in enumerate(tqdm(observation_per_cluster)):
    #     all_samples[i] = posterior.sample((num_samples,), x=obs, show_progress_bars=False)
    ##################


    ############################# Parallel code
        
    from concurrent.futures import ThreadPoolExecutor

    # Function to process a block of observations
    def process_block(start_idx, end_idx, observation_per_cluster):
        block_samples = np.empty((end_idx - start_idx, num_samples, L))
        for i in range(start_idx, end_idx):
            block_samples[i - start_idx] = posterior.sample((num_samples,), x=observation_per_cluster[i], show_progress_bars=False)
        return block_samples

    # Split the observations into blocks of 200
    block_size = 200
    n_blocks = (num_injections + block_size - 1) // block_size  # Calculate the number of blocks

    all_samples = np.empty((num_injections, num_samples, L))


    # Using ThreadPoolExecutor to parallelize the process
    with ThreadPoolExecutor() as executor:
        futures = []
        # Initialize tqdm for the progress bar
        with tqdm(total=n_blocks, desc="Processing blocks", ncols=100) as pbar:
            # Submit each block of work to the executor
            for block_idx in range(n_blocks):
                start_idx = block_idx * block_size
                end_idx = min((block_idx + 1) * block_size, num_injections)
                futures.append(executor.submit(process_block, start_idx, end_idx, observation_per_cluster))
            
            # Process the results as they come in and update the progress bar
            for future in futures:
                block_samples = future.result()  # Get the result of the block
                block_idx = futures.index(future)
                start_idx = block_idx * block_size
                end_idx = min((block_idx + 1) * block_size, num_injections)
                all_samples[start_idx:end_idx] = block_samples
                pbar.update(1)  # Update the progress bar after each block is processed


    # Now all_samples is populated with the results

    ##############################

    estimated_params = np.zeros((len(all_samples), 5)) 

    for i in range(len(all_samples)):
        params = np.percentile(all_samples[i], 50, axis=0)
        for j in range(len(params)):
            estimated_params[i,j] = params[j]


    df['M1'] = 0
    df['q_ratio'] = -1
    df['age'] = 0
    df['fe_h'] = -10
    df['log_distance'] = 0
    df.reset_index(inplace=True)

    for i in range(len(observation_per_cluster)):
        df['M1'][i] = estimated_params[i,0]
        df['q_ratio'][i] = estimated_params[i,1]
        df['age'][i] = estimated_params[i,2]
        df['fe_h'][i] = estimated_params[i,3]
        df['log_distance'][i] = estimated_params[i,4]


    df_binary = df[(df['M1'] != 0) & (df['q_ratio'] != -1) & (df['age'] != 0) & (df['fe_h'] != -10) & (df['log_distance'] != 0)]
    df_binary = df_binary[['GaiaEDR3','Gmag','RPmag','BPmag','Plx','Dist50','M1','q_ratio','age','fe_h','log_distance']]
    df_binary.to_csv('./data_artem/binaries_{}.csv'.format(n_sample))
