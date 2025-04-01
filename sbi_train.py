import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
import json

from isochrones import get_ichrone
import pandas as pd
import os

from sbi import utils
#from sbi.inference import SNPE as method # paper
from sbi.inference import SNLE as method
#from sbi.inference import SNRE as method


tracks = get_ichrone('mist', tracks=True)

def binary_color_mag_isochrones(m1, q, age, fe_h):
    properties = tracks.generate_binary(m1, q * m1, np.log10(age) + 9, fe_h, bands=["G", "BP", "RP"])
    g_mag = properties.G_mag.values 
    bp_mag = properties.BP_mag.values
    rp_mag = properties.RP_mag.values
    return np.array([g_mag, bp_mag, rp_mag]).T

def simulator(theta):
    return torch.tensor(binary_color_mag_isochrones(*theta))


############ Set priors.

from torch.distributions import (Uniform, Exponential, Normal, Beta, Pareto)
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch import tensor as tt
from sbi.utils.user_input_checks import prepare_for_sbi

age_bounds = [0.2, 10.0]          # Gyr
mass_bounds = [0.3, 3.0]          # M1
metallicity_bounds = [-1.5, 0.5]  # [Fe/H] forced metallicity distribution

num_simulations = 50_000
num_samples = 10_000

# prior = [
#     TransformedDistribution(Exponential(rate=tt([2.3])), \
#                             AffineTransform(loc=tt([mass_bounds[0]]), scale=tt([mass_bounds[1]]))),                   # M1
#     Uniform(tt([0.]), tt([1.])),                                                                                      # q
#     Uniform(tt([age_bounds[0]]), tt([age_bounds[1]])),                                                                # age
#     TransformedDistribution(Normal(loc=tt([0.0]), scale=tt([0.5])), \
#                             AffineTransform(loc=tt([metallicity_bounds[0]]), scale=tt([metallicity_bounds[1]])))      # [Fe/H]
#     ]

# prior = [
#         TransformedDistribution(Pareto(scale=tt([1.]), alpha=tt([5.])), AffineTransform(loc=tt([0.8]), scale=tt([1.]))), # M1
#         Uniform(tt([0.]), tt([1.])),                                                                                     # q
#         Uniform(tt([1.]), tt([10.])),                                                                                    # age
#         TransformedDistribution(Beta(tt([10.]), tt([2.])), AffineTransform(loc=tt([-1.5]), scale=tt([0.5])))             # [Fe/H]
#         ]

bounds = np.array([
    [mass_bounds[0],  mass_bounds[1]],                           # M1
    [0,      1],                                                 # q
    [age_bounds[0],   age_bounds[1]],                            # (Gyr)
    [metallicity_bounds[0], metallicity_bounds[1]]               # metallicity
    ])
bounds = torch.tensor(bounds)
prior = utils.BoxUniform(low=bounds.T[0], high=bounds.T[1])

sbi_simulator, sbi_prior = prepare_for_sbi(simulator, prior)
inference = method(sbi_prior)


posterior_path = "./data/train_posterior_uniform.pkl"

def simulate_for_sbi_strict(simulator, proposal, num_simulations, max_trials=np.inf):
    num_trials, num_simulated, theta, x = (0, 0, [], [])
    while num_simulated < num_simulations:
        N = num_simulations - num_simulated
        print(f"Running {N} simulations")
        _theta = proposal.sample((N, ))
        _x = simulator(_theta)
        _x = _x.squeeze(1) # This will change it from [50000, 1, 3] to [50000, 3]
        keep = np.all(np.isfinite(_x).numpy(), axis=1)
        theta.extend(np.array(_theta[keep]))
        x.extend(np.array(_x[keep]))
        num_trials += 1
        num_simulated += sum(keep)
        if num_trials > max_trials:
            print(f"Warning: exceeding max trials ({max_trials}) with {num_simulated} / {num_simulations} simulations")
            break
    theta = torch.tensor(np.vstack(theta))
    x = torch.tensor(np.vstack(x))
    return (theta, x)

if os.path.exists(posterior_path):
    print(f"Pre-loading posterior from {posterior_path}")
    with open(posterior_path, "rb") as fp:
        posterior, (theta, x) = pickle.load(fp)

else:
    theta, x = simulate_for_sbi_strict(sbi_simulator, sbi_prior, num_simulations)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator, sample_with='mcmc')
    
    with open(posterior_path, "wb") as fp:
        pickle.dump((posterior, (theta, x)), fp)
        
    print(f"Saved posterior to {posterior_path}")



min_g_mag_simulated = x[:,0].numpy().min()
max_g_mag_simulated = x[:,0].numpy().max()
min_bp_mag_simulated = x[:,1].numpy().min()
max_bp_mag_simulated = x[:,1].numpy().max()
min_rp_mag_simulated = x[:,2].numpy().min()
max_rp_mag_simulated = x[:,2].numpy().max()

print(f'g_mag ranges from {min_g_mag_simulated} to {max_g_mag_simulated}')
print(f'bp_mag ranges from {min_bp_mag_simulated} to {max_bp_mag_simulated}')
print(f'rp_mag ranges from {min_rp_mag_simulated} to {max_rp_mag_simulated}')


########### Estimations per cluster
data = pd.read_csv('./data/members_dat.csv') # all data

list_clusters =  ['Alessi_5', 'Alessi_9', 'ASCC_101', 'BH_99', 'Blanco_1', 'IC_2602', 'NGC_2516',
  'NGC_2547', 'NGC_3532', 'NGC_7058', 'Pozzo_1', 'Melotte_22', 'NGC_2632', 'Trumpler_10']

for cluster in list_clusters:

    #cluster = 'NGC_6475'
    print(f'Estimations for cluster {cluster}')
    remove_stars = pd.read_csv('./data/SGs_and_RGBs_clusters_sourceid.csv')
    list_remove_stars = remove_stars['source_id'][(remove_stars['cluster'] == cluster)].to_list()

    df_cluster = data[data['cluster'] == cluster]
    df_cluster = df_cluster[~df_cluster['source_id'].isin(list_remove_stars)]

    df_cluster['g_mag'] = df_cluster['phot_g_mean_mag'] - 5* np.log10(1000/df_cluster['parallax']) + 5
    df_cluster['bp_mag'] = df_cluster['phot_bp_mean_mag'] - 5* np.log10(1000/df_cluster['parallax']) + 5
    df_cluster['rp_mag'] = df_cluster['phot_rp_mean_mag'] - 5* np.log10(1000/df_cluster['parallax']) + 5

    print('before removing stars outside boundaries', len(df_cluster))
    print('df g ranges: ', df_cluster['g_mag'].min(), df_cluster['g_mag'].max())
    print('df bp ranges: ', df_cluster['bp_mag'].min(), df_cluster['bp_mag'].max())
    print('df rp ranges: ', df_cluster['rp_mag'].min(), df_cluster['rp_mag'].max())


    cond_g = (df_cluster['g_mag'] >= min_g_mag_simulated) & (df_cluster['g_mag'] <= max_g_mag_simulated)
    cond_bp = (df_cluster['bp_mag'] >= min_bp_mag_simulated) & (df_cluster['bp_mag'] <= max_bp_mag_simulated)
    cond_rp = (df_cluster['rp_mag'] >= min_rp_mag_simulated) & (df_cluster['rp_mag'] <= max_rp_mag_simulated)
    bound_df_cluster = df_cluster[cond_g & cond_bp & cond_rp].reset_index(drop=True)
    print('after removing stars outside boundaries', len(bound_df_cluster))

    L = 4 # number of parameters
    num_injections = len(bound_df_cluster)
    parameters = np.empty((num_injections, num_samples, L))
    # g_mag, bp_mag, rp_mag

    for i in tqdm(range(num_injections)):
        observation_per_star = bound_df_cluster[['g_mag','bp_mag','rp_mag']].to_numpy()[i]
        parameters[i] = posterior.sample((num_samples,), x=torch.tensor(observation_per_star), show_progress_bars=False)
        

    final_parameters = {}
    for i in range(len(bound_df_cluster)):
        values = np.percentile(parameters[i], 50, axis=0)
        final_parameters[i] = {'mass': values[0],
                                'q': values[1],
                                'age': values[2],
                                'fe_h': values[3]
                                }
        
    with open(f'./data/estimations_{cluster}_paper.json', 'w') as f:
        json.dump(final_parameters, f, indent=4)
    
