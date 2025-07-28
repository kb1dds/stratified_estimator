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

import numpy as np
import numpy.random
import scipy.spatial
import scipy.stats
import plotly.graph_objects as go
import plotly.express as px

from stratified_estimator.split_geo_estimator import estimate_stratifications

### Parameter setup for the data
n = 2       # Dimensions 
spts = 2000 # number of points

### Parameter setup for the test
vol_min = 20
vol_max = 100

class Args:
    nstrat = 3
    miller = True
    ricci = True

args = Args()

#### Do not alter below this line

# Generate points on the sphere and a stick in n dimensions
coords = np.random.randn(spts,n+1)
coords = coords / np.sqrt(np.sum(coords**2,axis=1)[:,np.newaxis])
coords[:,0]=coords[:,0]-1
stick = np.zeros((int(spts/6),n+1))
stick[:,0]=np.linspace(0,5,int(spts/6))+(30/spts)*(np.random.random((int(spts/6),))-0.5)
coords=np.vstack((coords,stick))

# Compute distance matrix
dists = scipy.spatial.distance_matrix(coords,coords)

# Each column is a point; rows are distances to other points, in order
dists_sorted = np.sort(dists,axis=0)

# Total number of points
npts = dists_sorted.shape[0]

fig1=px.scatter(x=dists_sorted[vol_min:vol_max,spts+30],y=1+np.arange(vol_min,vol_max),log_x=True,log_y=True)
fig1.show()

#fig1=px.scatter(x=dists_sorted[vol_min:vol_max,-30],y=1+np.arange(vol_min,vol_max),log_x=True,log_y=True)
#fig1.show()

print(estimate_stratifications(dists_sorted[:,spts+30], vol_min, vol_max, npts, args, ws=10, alpha=1e-2))
print(estimate_stratifications(dists_sorted[:,-30], vol_min, vol_max, npts, args, ws=10, alpha=1e-2))

#quite

with open('lollipop_example.csv', 'wt') as fp:
    # Produce header
    fp.write('x,y,z,stratification_number,radius,volume,scaling_coeff,dimension,ricci,pvalue\n')
    
    # Loop over each point (so that each point gets its own vector of radii)
    pvalues = []
    
    for i in range(dists_sorted.shape[1]):
        
        # Do the estimation
        output = estimate_stratifications(dists_sorted[:,i], vol_min, vol_max, npts, args, ws=10,alpha=1e-3)
        pvalues.append(min(output['pvalues']))
        
        # Format output
        for j in range(len(output['scaling_coeffs'])):
            fp.write(str(coords[i,0])+
                     ','+
                     str(coords[i,1])+
                     ','+
                     str(coords[i,2])+
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

fig = go.Figure(data=[go.Scatter3d(x=coords[:,0],
                                   y=coords[:,1],
                                   z=coords[:,2],
                                   mode='markers',
                                   marker=dict(size=5,
                                               color=np.log10(pvalues),#[int(p<1e-3/npts) for p in pvalues], # Bonferroni correction
                                               colorscale='Viridis',
                                               opacity=0.5))])
fig.show()
