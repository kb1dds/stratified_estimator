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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Select a ball of coordinates centered on a given point')
    parser.add_argument('filename',
                        help='File to parse.')
    parser.add_argument('center',
                        default=0,
                        type=int,
                        help='Center point index (0-based).')
    parser.add_argument('--path',
                        default='/media/TerraSAR-X/ER/data/20240119_embeddings',
                        help='Path to embedding file.  Output is to current directory.')
    parser.add_argument('--rad-min',
                        help='Minimum radius for selection.',
                        type=float,
                        default=0)
    parser.add_argument('--rad-max',
                        help='Maximum radius for selection.',
                        type=float,
                        default=5)

    args=parser.parse_args()

    print(args)

    # Load data
    coords = torch.load(args.path+'/embeddings_'+args.filename+'.pt',map_location='cpu').to(dtype=torch.float16).numpy()

    # Center point
    center=coords[args.center,:]

    # Build output file
    outfile = 'selection_'+args.filename+'_token_'+str(args.center)+'.csv'

    with open(outfile, 'wt') as fp:

        # Header row
        fp.write('token_id')
        for i in range(coords.shape[1]):
            fp.write(',X'+str(i+1))
        fp.write('\n')
        
        # Loop over each point (so that each point gets its own vector of radii)
        for i in range(coords.shape[0]):
            # Compute the distance between the center and this point
            distance = np.sqrt(np.sum(np.abs(coords[i,:]-center)**2));

            # Selection
            if(( distance >= args.rad_min ) and (distance < args.rad_max)):
                fp.write(str(i))
                for j in range(coords.shape[1]):
                    fp.write(','+str(coords[i,j]))
                fp.write('\n')
                
