#/usr/bin/python3
# Script to compute local geometric and topological properties of an LLM embedding
#
#Copyright (c) 2024-2025 Michael Robinson
#
#Licensed under the Apache License, Version 2.0 (the "License"); you may not use 
# this file except in compliance with the License. You may obtain a copy of the 
#License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#License for the specific language governing permissions and limitations under
# the License.
#
# This material is based upon work supported by the Defense Advanced Research 
# Projects Agency (DARPA) SafeDocs program under contract HR001119C0072. 
# Any opinions, findings and conclusions or recommendations expressed in 
# this material are those of the authors and do not necessarily reflect the 
# views of DARPA.

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
    pointwise_lfit_data = np.linalg.lstsq(rstack,np.log(volumes),rcond=None)

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

    Inputs: radii : np.array of radius values (assumed to be sorted in ascending order)
            volumes : np.array of volume values
            ws : window size, defaults to 10 samples
            alpha : instantaneous false alarm rate for test (probability, smaller is more stringent)
    
    Output: (index,pvalue) into radii or None if no stratification found and p-value of test
    
    Test compares the dimension estimates in [-2*ws,-ws] versus [ws,2*ws]

    Note: Since this is a multiple testing scenario, one should apply a correction to alpha
          Because the tests run across sliding windows, they are highly correlated, which will
          make Bonferroni much too restrictive.  Therefore, any desired correction should be
          applied outside of this function!
    '''
    
    # Pointwise dimension estimates (noisy but fast)
    dimvec=np.gradient(np.log(volumes))/np.gradient(np.log(radii))

    # Run the test
    for w in range(2*ws,dimvec.shape[0]-2*ws):
        pvalue = scipy.stats.ttest_ind(dimvec[w-2*ws:w-ws],dimvec[w+ws:w+2*ws],equal_var=False).pvalue
        if pvalue < alpha:
            return w,pvalue

    return None,1.0 # No stratifications found

def estimate_stratifications(dists_sorted, vol_min, vol_max, npts, args, ws=10, alpha=1e-3):
    '''
    Detect and report stratification properties given radii to neighboring points

    Inputs: dists_sorted : np.array of radius values (assumed to be sorted in ascending order)
            vol_min,vol_max : minimum and maximum number of points to be used
            npts    : total number of points in the dataset (used to normalize scaling coefficient)
            args.nstrat : integer, maximum number of stratifications to detect
            args.miller : Boolean, if True use Miller debiasing
            args.ricci : Boolean, if True estimate Ricci curvature, else set to zero
            ws : window size, defaults to 10 samples
            alpha : false alarm rate for test (probability, smaller is more stringent)

    Output: a dictionary with keys:
            'scaling_coeffs' : list of scaling coefficient estimates
            'dimensions' : list of dimension estimates
            'riccis' : list of Ricci estimates
            'strat_radii' : list of radii at which stratifications were detected
            'strat_volumes' : list of volumes at which stratifications were detected
            'pvalues' : list of p-values for each stratification detected
    '''

    # Distances to nearest points
    radii = dists_sorted[vol_min:vol_max]

    # Number of points within each radius (proportional to volume)
    volumes = np.arange(vol_min,vol_max)

    # Prepare output lists
    output = dict()
    output['scaling_coeffs'] = []
    output['dimensions'] = []
    output['riccis'] = []
    output['strat_radii'] = []
    output['strat_volumes'] = []
    output['pvalues'] = []

    # Start of window for detecting stratifications, avoiding zero radii
    vol_min_current = np.argmax(radii>1e-10)

    for strat_number in range(args.nstrat):
        # End of window for detecting stratifications is end of data
        vol_max_current = radii.shape[0]

        # print('Looking for stratifications between: ' + str(vol_min_current) + ' to ' + str(vol_max_current))

        # Detect first stratification within window
        # Note the multiple test correction applied to the threshold based upon expected number of strata;
        # We assume that adjacent windows within a stratum are highly correlated.
        strat_idx,pvalue = stratification_test(radii[vol_min_current:vol_max_current],
                                               volumes[vol_min_current:vol_max_current],
                                               ws,
                                               alpha/args.nstrat)

        # If stratification is detected, update the end of the window.
        # Otherwise the window stays as is
        if strat_idx is not None:
            vol_max_current = strat_idx+vol_min_current
            # print('Found stratification at ' + str(vol_max_current))

        # print('Estimating: ' + str(vol_min_current) + ' to ' + str(vol_max_current))

        # Estimate geometry of this stratification
        scaling_coeff,dimension,ricci = geo_estimator(radii[vol_min_current:vol_max_current],
                                                      volumes[vol_min_current:vol_max_current],
                                                      npts,args)

        # Store output
        output['scaling_coeffs'].append(scaling_coeff)
        output['dimensions'].append(dimension)
        output['riccis'].append(ricci)
        
        output['strat_volumes'].append(vol_min+vol_min_current)
        output['strat_radii'].append(radii[vol_min_current])
        output['pvalues'].append(pvalue)

        # If no new stratifications were detected, exit the loop
        if strat_idx is None:
            break

        # Otherwise update the start of the window to the location of the most recently found stratification
        vol_min_current = strat_idx+vol_min_current

    return output


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Estimate local geometric and topological properties of an LLM embedding')
    parser.add_argument('filename',
                        help='File to parse.')
    parser.add_argument('--distfile',
                        help='The file contains sorted distances instead of coordinates.',
                        dest='distfile',
                        action='store_true')
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
    parser.add_argument('--nstrat',
                        help='Maximum number of stratifications to detect.',
                        type=int,
                        default=3)

    args=parser.parse_args()

    print(args)

    vol_min = args.vol_min # minimum number of points in ball for linear regression
    vol_max = args.vol_max # maximum number of points in ball for linear regression

    if args.distfile:
        dists_sorted = np.loadtxt(args.path+'/'+args.filename+'.csv',
                                  delimiter=',')
    else:
         # Load data
        coords = torch.load(args.path+'/embeddings_'+args.filename+'.pt',map_location='cpu').to(dtype=torch.float16).numpy()

        # Compute distance matrix
        dists = scipy.spatial.distance_matrix(coords,coords)

        # Each column is a point; rows are distances to other points, in order
        dists_sorted = np.sort(dists,axis=0)

        np.savetxt('distsubset_'+args.filename+'.csv',
                   dists_sorted[:,0:100],
                   header='',
                   delimiter=',',
                   comments = '') # Remove the silly `#` on the header line

    # Total number of points
    npts = dists_sorted.shape[0]

    # Build filename for output
    outfile = 'embeddings_'+args.filename+'_loglog_regressions'
    if args.miller:
        outfile = outfile+'_miller'
    if args.ricci:
        outfile = outfile+'_ricci'
    outfile = outfile+ '.csv'

    with open(outfile, 'wt') as fp:
        # Produce header
        fp.write('token_id,stratification_number,radius,volume,scaling_coeff,dimension,ricci,pvalue\n')
        
        # Loop over each point (so that each point gets its own vector of radii)
        for i in range(dists_sorted.shape[1]):

            # Do the estimation
            output = estimate_stratifications(dists_sorted[:,i], vol_min, vol_max, npts, args)

            # Format output
            for j in range(len(output['scaling_coeffs'])):
                fp.write(str(i)+
                         ','+
                         str(j)+
                         ','+
                         str(output['strat_radii'][j])+
                         ','+
                         str(output['strat_volumes'][j])+
                         ','+
                         str(output['scaling_coeffs'][j])+
                         ','+
                         str(output['dimensions'][j])+
                         ','+
                         str(output['riccis'][j])+
                         ','+
                         str(output['pvalues'][j]))
                fp.write('\n')
