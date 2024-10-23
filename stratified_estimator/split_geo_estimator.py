#/usr/bin/python3
# Script to compute local geometric and topological properties of an LLM embedding

import sys
import argparse
import numpy as np
import scipy.spatial
import scipy.stats
import torch

def geo_estimator(radii,volumes,npts,args):
    '''
    Estimate dimension and Ricci scalar curvature from a segment of log volume versus log radius

    Inputs: radii : np.array of radius values
            volumes : np.array of volume values
            npts    : total number of points in the dataset (used to normalize scaling coefficient)
            args.miller : Boolean, if True use Miller debiasing
            args.ricci : Boolean, if True estimate Ricci curvature, else set to zero
    Outputs: tuple with three float elements:
            scaling coefficient
            dimension
            Ricci scalar curvature
    '''

    # Construct linear problem
    rstack = np.column_stack((np.ones_like(radii),  # Column 0: Intercept
                              np.log(radii)))       # Column 1: Slope (dimension)

    # Solve linear problem
    pointwise_lfit_data = np.linalg.lstsq(rstack,np.log(volumes))

    # Deserialize and postprocess solution
    pointwise_lfit = pointwise_lfit_data[0]
    scaling_coeff = np.exp(pointwise_lfit[0])/npts
    dimension = pointwise_lfit[1]

    # Include Miller debiasing if requested
    if args.miller: 
        scaling_coeff = scaling_coeff * np.exp(0.5*pointwise_lfit_data[1][0]**2)

    # Ricci curvature computed off the residuals
    if args.ricci:
        residuals = pointwise_lfit_data[1]
        ricci = np.mean(-residuals*6*(dimension+2)/radii**2)
    else:
        ricci = 0.0

    return (scaling_coeff,dimension,ricci)

def stratification_test(radii, volumes, ws=10, alpha=1e-3):
    '''
    Locate the index of the smallest radius stratification using a sliding window Welch t-test

    Inputs: radii : np.array of radius values
            volumes : np.array of volume values
            ws : window size, defaults to 10 samples
            alpha : threshold for test statistic (probability, smaller is more stringent)
    
    Output: index into radii or None if no stratification found
    
    Test compares the dimension estimates in [-2*ws,-ws] versus [ws,2*ws]
    '''
    
    # Pointwise dimension estimates (noisy but fast)
    dimvec=np.gradient(np.log(volumes))/np.gradient(np.log(radii))

    # Run the test
    for w in range(2*ws,dimvec.shape[0]-2*ws):
        if scipy.stats.ttest_ind(dimvec[w-2*ws:w-ws],dimvec[w+ws:w+2*ws],equal_var=False).pvalue < alpha:
            return w

    return None # No stratifications found

parser = argparse.ArgumentParser(description='Estimate local geometric and topological properties of an LLM embedding')
parser.add_argument('filename',
                    help='File to parse.')
parser.add_argument('--path',
                    default='/media/TerraSAR-X/ER/data/20240119_embeddings',
                    help='Path to embedding file.  Output is to current directory.')
parser.add_argument('--no-miller',
                    dest='miller',
                    help='Disable Miller debiasing of scaling coefficient.',
                    action='store_false')
parser.add_argument('--ricci',
                    help='Compute Ricci curvature.',
                    action='store_true')
parser.add_argument('--vol-min',
                    help='Minimum volume to use for estimation.',
                    type=int,
                    default=10)
parser.add_argument('--vol-max',
                    help='Maximum volume to use for estimation.',
                    type=int,
                    default=50)
args=parser.parse_args()

print(args)

vol_min = args.vol_min # minimum number of points in ball for linear regression
vol_max = args.vol_max # maximum number of points in ball for linear regression

# Load data
coords = torch.load(args.path+'/embeddings_'+args.filename+'.pt',map_location='cpu').to(dtype=torch.float16).numpy()

# Compute distance matrix
dists = scipy.spatial.distance_matrix(coords,coords)
npts = dists.shape[0]

# Each column is a point; rows are distances to other points, in order
dists_sorted = np.sort(dists,axis=0)

np.savetxt('distsubset_'+args.filename+'.csv',
           dists_sorted[:,0:100],
           header='',
           delimiter=',',
           comments = '') # Remove the silly `#` on the header line

# Preallocate output data
pointwise_data = np.zeros((npts,8))

# Loop over each point (so that each point gets its own vector of radii)
for i in range(dists_sorted.shape[1]):

    # Distances to nearest points
    radii = dists_sorted[vol_min:vol_max,i]

    # Number of points within each radius (proportional to volume)
    volumes = np.arange(vol_min,vol_max)

    strat_idx = stratification_test(radii, volumes)

    if strat_idx is None:
        scaling_coeff_1,dimension_1,ricci_1 = geo_estimator(radii,volumes,npts,args)
        scaling_coeff_2,dimension_2,ricci_2 = 0., 0., 0.

        strat_idx = -1
        strat_radius = -1.

    else:
        scaling_coeff_1,dimension_1,ricci_1 = geo_estimator(radii[vol_min:strat_idx], volumes[vol_min:strat_idx], npts, args)
        scaling_coeff_2,dimension_2,ricci_2 = geo_estimator(radii[strat_idx:vol_max], volumes[strat_idx:vol_max], npts, args)

        strat_radius = radii[strat_idx]

    # Format output
    pointwise_data[i,:] = np.array((scaling_coeff_1,
                                    dimension_1,
                                    ricci_1,
                                    strat_idx, strat_radius,
                                    scaling_coeff_2,
                                    dimension_2,
                                    ricci_2))

   
# Save results
outfile = 'embeddings_'+args.filename+'_loglog_regressions'
if args.miller:
    outfile = outfile+'_miller'
if args.ricci:
    outfile = outfile+'_ricci'
outfile = outfile+ '.csv'

np.savetxt(outfile,
           pointwise_data,
           header='scaling_coeff_1,dimension_1,ricci_1,strat_idx,strat_radius,scaling_coeff_2,dimension_2,ricci_2',
           delimiter=',',
           comments = '') # Remove the silly `#` on the header line
