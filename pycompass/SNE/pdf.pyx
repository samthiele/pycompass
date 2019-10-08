#cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False,cdivision=True
#^ disables type checking etc - faster but causes seg-faults. on error

from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport acos
from libc.math cimport asin
from libc.math cimport atan2
from libc.math cimport log
from libc.math cimport exp
from libc.math cimport sqrt

#import numpy and scipy
import numpy as np
cimport numpy as np
import scipy.special as sp
import scipy.stats as stats

#cimport scipy as sp

#pull pi from numpy
cdef double pi = np.pi

"""
Prior probability function
"""
cpdef double pp(double phi, double theta, double nx, double ny, double nz):
    #check normal is pointing downwards
    if nz > 0:
        nx = -nx
        ny = -ny
        nz = -nz
    
    #calculate angle between normal vector and the normal estimate (phi,theta)
    cdef double _t = acos(nx*sin(phi)*cos(theta)+ny*cos(phi) * cos(theta)-nz*sin(theta))
    
    #return prior-probability
    return sin(_t) / (2*pi) #n.b. 2 pi is normalizing factor so that integrates to 1 over dphi,dtheta

"""
Evaluate prior probability over a grid.
"""
def prior(np.ndarray grid, np.ndarray normal):
    #grid is a list of phi,theta points such that grid[0] = [phi1,phi2,....]
    if grid.shape[0] != 2:
        raise ValueError("Grid must contain a list of phi and a list of theta.")
    if normal.shape[0] != 3:
        raise ValueError("Normal must be a list of 3 components.")
    
    #create location for output
    cdef np.ndarray out = np.zeros([grid.shape[1]],dtype=np.double)

    #loop through grid and evaluate prior
    cdef double trend, plunge
    cdef int i 
    for i in range(grid.shape[1]):
        #convert lat, lon to trend plunge
        plunge = -asin( cos(grid[0][i]) * cos(grid[1][i]) )
        trend = atan2( -sin(grid[0][i]),cos(grid[0][i]) * sin(grid[1][i]) ) - pi / 2
        
        out[i] = pp(trend,plunge,normal[0],normal[1],normal[2])
    return out

"""
3D gamma function
"""
cdef double logGamma3(double n):
    return (3./2.) * log(pi) + sp.gammaln(n) + sp.gammaln(n-0.5) + sp.gammaln(n-1.0)

"""
Calculate the log of the normalising factor used by the wishart distribution. This has been pulled out as it only needs
to be calculated once per observed covariance matrix (i.e. is independent of the P-matrix). Hence this is not speed critical.
"""
cpdef double wishLSF(double[:,:] X, int n):
    return (n-4)*0.5*log(np.linalg.det(X)) - (n*3./2.) * log(2.) - logGamma3(n/2.)
    
"""
Compute the natural logarithm of the probability density of the Wishart distribution. This has been optimised
specifically for this case, and hence expresses the scale matrix in terms of its eigens (ie. a P-matrix).

**Arguments**:
-X = the observed scale matrix. This should equal the covarariance (or inverse covariance) * the number of observations
-phi = the trend of the principal component of the scale matrix
-theta = the plunge of the principal component of the scale matrix
-alpha = angle of rotation of the 2nd eigenvector of the scale matrix
-e2 = 2nd eigenvalue of the scale matrix
-e3 = 3rd eigenvalue of the scale matrix
-nobserved = the number of data points used to constrain X
-lsf = the log-scale factor (as computed by wishLSF(...) associated with this covariance matrix.

**Returns**
-the natural logarith of the probability density
"""
cpdef double logWish(double[:,:] X, int n, double phi, double theta, double alpha, double e1, double e2, double e3, double lsf):
    ####################################
    ##Derive scale matrix
    ####################################
    #eigenvector 3
    cdef double e13 = sin(phi) * cos(theta)
    cdef double e23 = cos(phi) * cos(theta)
    cdef double e33 = -sin(theta)
    #eigenvector 2
    cdef double e12 = sin(phi) * sin(theta) * sin(alpha) - cos(phi) * cos(alpha)
    cdef double e22 = sin(phi) * cos(alpha) + sin(theta) * cos(phi) * sin(alpha)
    cdef double e32 = sin(alpha) * cos(theta)
    #eigenvector 1 (calculate using cross product to avoid using un-necessary trig functions)
    cdef double e11 = e23*e32 - e33*e22
    cdef double e21 = e33*e12 - e13*e32
    cdef double e31 = e13*e22 - e23*e12
    
    #calculate determinant of the scale matrix by multiplying it's eigens
    cdef double D = e1*e2*e3
    
    #calculate inverse scale matrix I = V^-1 = (basis * [[1,0,0],[0,e2,0],[0,0,e3]] * basis_T)^-1
    #(this will have eigenvectors corresponding to the above and eigenvalues of 1/e1,1/e2,1/e3)
    #(we compare this to the observed cov matrix
    e1 = 1.0/e1
    e2 = 1.0/e2 #N.B. We invert the eigenvalues so we compute the inverse of P (called I). The eigenvectors will not change
    e3 = 1.0/e3
    
    #calculate unique components of I
    cdef double i11 = (e1*e11**2 + e2*e12**2 + e3*e13**2)
    cdef double i22 = (e1*e21**2 + e2*e22**2 + e3*e23**2)
    cdef double i33 = (e1*e31**2 + e2*e32**2 + e3*e33**2)
    cdef double i12 = (e1*e11*e21 + e2*e12*e22 + e3*e13*e23)
    cdef double i13 = (e1*e11*e31 + e2*e12*e32 + e3*e13*e33)
    cdef double i23 = (e1*e21*e31 + e2*e22*e32 + e3*e23*e33)
    
    
    #compute the trace of I times X
    cdef double trIX = (i11*X[0,0]+i12*X[1,0]+i13*X[2,0])+(i12*X[0,1]+i22*X[1,1]+i23*X[2,1])+(i13*X[0,2]+i23*X[1,2]+i33*X[2,2])
    
    #compute the log wishart probability density
    #=scale factor - trace( I x icov /2) - (n/2) log (D), where D is the determinant of the scale matrix and I is its inverse
    #=scale factor - (1/2) * (trace(I x icov) - n * log(D))
    return lsf - 0.5 * (trIX + n*log(D))
     
     
##########Try #######################################################################
##1D Explicit Solutions (achieved by integrating over alpha). Reasonably fast...
#################################################################################
"""
Explicitely integrate the likelihood function.over alpha. e2 and e3 are treated as fixed (known) quantities. Note that this ignores
the scale factor (wishLSF) as it can be appied after doing the integration. 

**Arguments**:
 -X,n,phi,theta,e2,e3 = see description in documentation for logWish(...).
 -dA = the alpha step to use during the integration
 -lsf = the log scale factor to use in the wishart distribution.
"""
cdef double likExp1D(double[:,:] X, int n, double phi, double theta, double e1, double e2, double e3, int steps, double lsf):
    #evaluate integral over alpha = 0 to pi
    cdef double pd0 = exp(logWish(X,n,phi,theta,0,e1,e2,e3,lsf))
    cdef double pd1 = 0.0
    cdef double sum= 0.0
    cdef double dA = pi / steps
    cdef int i
    for i in range(steps):
       #calculate next integration point
       pd1 = exp(logWish(X,n,phi,theta,(i+1)*dA,e1,e2,e3,lsf))

       #compute area between current/next integration point (and sum it)
       sum += dA*pd0 + dA*(pd1-pd0)*0.5
       
       #move on to next point
       pd0=pd1
       
    #done :)
    return sum

"""
Evaluate likelihood function over a grid of (phi,theta)

**Arguments**:
 -grid = the grid of points on which to evaluate the likelihood function
 -cov = the observed covariance matrix
 -nobserved = the number of points used to compute the observed covariacne matrix 
 -steps = the number of integration steps to used. Default is 25.
"""
def likelihoodExp1D(np.ndarray grid, np.ndarray cov, int nobserved,int steps=25):
    #grid is a list of phi,theta points such that grid[0] = [phi1,phi2,....]
    if grid.shape[0] != 2:
        raise ValueError("Grid must contain a list of phi and a list of theta.")
    if cov.shape[0] != 3 and cov.shape[1] != 3:
        raise ValueError("Inverse covariance must be a 3x3 matrix.")
    
    
    #compute the scatter matrix
    cdef double[:,:] X = cov * nobserved
    
    #get eigenvalues (these are treated as fixed to avoid the disgusting triple-integral needed to reduce them out otherwise...)
    eval,evec = np.linalg.eig(cov)
    eval[::-1].sort()
    cdef double e1 = eval[0]
    cdef double e2 = eval[1]
    cdef double e3 = eval[2]
    
    #create location for output
    cdef double[:] out = np.zeros([grid.shape[1]])
    
    #loop through grid and evaluate likelihood
    cdef double trend, plunge
    cdef lsf = wishLSF(X,nobserved-1) #calculate the wishart scale factor
    cdef double[:,:] points = grid #convert grid to native c-type for speed
    cdef int i
    for i in range(grid.shape[1]):
        #convert lat, lon to trend plunge
        plunge = -asin( cos(points[0,i]) * cos(points[1,i]) )
        trend = atan2( -sin(points[0,i]),cos(points[0,i]) * sin(points[1,i]) ) - pi / 2.0
        
        out[i] = likExp1D(X,nobserved-1,trend,plunge,e1,e2,e3,steps,lsf)
    return out

"""
Posterior solver - calculates a prior grid and a likelihood grid and multiplies the two together.

**Arguments**:
 -grid = the grid of (phi,theta) pairs to solve the posterior over
 -cov = the observed covariance matrix.
 -nobserved = the number of observations used to calculate icov
 -normal = the outcrop normal vector used to calculate the prior.
"""
def posteriorExp1D(np.ndarray grid, np.ndarray cov, int nobserved, np.ndarray normal, int steps=25 ):
    #evaluate prior and likelihood
    pr = prior(grid,normal)
    lik = likelihoodExp1D(grid,cov,nobserved,steps=steps)
    
    #return the product of the two
    return pr * lik

    
"""
Compute the posterior probability at the specified point.
"""
def getPosteriorDensityAt(double phi, double theta, np.ndarray cov, int nobserved, np.ndarray n, int steps=500):
    #compute the scatter matrix
    cdef double[:,:] X = cov * nobserved
    
    #get eigenvalues (these are treated as fixed to avoid the disgusting triple-integral needed to reduce them out otherwise...)
    eval,evec = np.linalg.eig(cov)
    eval[::-1].sort()
    cdef double e1 = eval[0]
    cdef double e2 = eval[1]
    cdef double e3 = eval[2]
    
    cdef lsf = wishLSF(X,nobserved) #calculate the wishart scale factor
    
    #return posterior
    return likExp1D(X,nobserved,phi,theta,e1,e2,e3,steps,lsf) * pp(phi,theta,n[0],n[1],n[2])

##########################################################################
###MCMC SAMPLERS
##########################################################################

"""
Sample from the posterior distribution using MCMC metropolis sampler. 

**Arguments**:
 -cov = the observed covariance matrix
 -nobserved = the number of points used to compute the observed covariance matrix
 -normal = a 3D normal vector (numpy array) for the outcrop orientation. Used to compute the prior. If none, a uniform prior is used.
 -nsamples = the length of the MCMC chain (i.e. the number of samples to get). Default is 5000.
 -maxIter = the maximum number of iterations to try before giving up. Default is 1 million. This shouldn't need to change... 
 -proposalWidth = the proposal width for the Metropolis sampler (the standard deviation of the normal distribution used to sample each jump-vector from). 
                  Smaller numbers are faster (less extreme samples tried and rejected), but less likely to fully explore the data space (treat results with caution!). Default is 0.1.
 -verbose = True if information on sampler iterations and timing should be sent to the print buffer. Default is true.
 
 **Returns**:
 -a nsamples x 3 numpy array where each element contains each [phi,theta,alpha] in the markov chain
"""
def samplePosteriorMCMC(np.ndarray cov, int nobserved, double[:] normal=None, int nsamples=5000, int maxIter=100000,double proposalWidth=0.075,verbose=True):
    #declare chain
    cdef double[:,:] chain = np.zeros([nsamples,3],dtype=np.double) #phi,theta,alpha
    
    #compute eigenvalues and vectors of cov
    eval,evec = np.linalg.eig(cov)
    idx = eval.argsort()[::-1] #sort eigens in decending order...
    eval = eval[idx]
    evec = evec[:,idx] #n.b. columns of evec are vectors....
    cdef double e1 = eval[0]
    cdef double e2 = eval[1]
    cdef double e3 = eval[2]
    
    #compute the scatter matrix
    cdef double[:,:] X = cov * nobserved
    
    #set initial phi,theta to trend,plunge of third eigenvector of covariance
    chain[0,0],chain[0,1] = vec2TrendPlunge(evec[:,2]) 
    
    #calculate alpha and set to initial value
    chain[0,2] = asin(evec[2,1]/cos(chain[0,1])) #alpha = arcsin(e2.z / cos(theta)) #change 2 back to 1?
    
    #compute log probability (density) of initial state
    cdef double sf = wishLSF(X,nobserved-1)
    cdef double log_p_current = 0
    if normal is None:
        log_p_current = logWish(X,nobserved-1,chain[0,0],chain[0,1],chain[0,2],e1,e2,e3,sf) #uniform prior
    else:
        log_p_current = logWish(X,nobserved-1,chain[0,0],chain[0,1],chain[0,2],e1,e2,e3,sf) + log( pp(chain[0,0],chain[0,1],normal[0],normal[1],normal[2]) )
    
    #define variables for proposed states
    cdef double log_p_proposed = 0
    cdef double phi = 0
    cdef double theta = 0
    cdef double alpha = 0
    
    #create array of random numbers (to avoid calling np.random.normal(...) and np.rand(...) within the loop
    cdef double[:,:] nRand = np.random.normal(0,proposalWidth,size=(maxIter,3)) #3 arrays of maxIter (typically 1-million) samples from normal distribution.
    cdef double[:] logURand = np.log(np.random.rand(maxIter)) #array of maxIter (typically 1-million) numbers from standard uniform distribution.
    
    #define loop variables
    cdef int c = 0
    cdef int iter = 0
    cdef int i = 0
    cdef int randIdx = 0 #index of the current random number
    for i in range(nsamples-1):
        #find next state
        c = 0
        while c < maxIter:
            iter += 1 #track iterations
            randIdx += 1
            if randIdx >= maxIter: #wrap random index if need be (is ok if the random numbers repeat as the sampler will be in a different spot)
                randIdx -= maxIter
            
            #sample proposed state
            phi = chain[i,0] + nRand[randIdx,0] #np.random.normal(0,proposalWidth)
            theta = chain[i,1] + nRand[randIdx,1] #np.random.normal(0,proposalWidth)
            alpha = chain[i,2] + nRand[randIdx,2] #np.random.normal(0,proposalWidth)
                
            #ensure phi/theta/alpha map to correct domains
            if theta < 0: #plunge should be positive, if negative, flip
                theta -= theta 
                phi -= pi
            if theta > pi / 2: #plunge > 90, flip
                theta = pi - theta
                phi -= pi
            while phi < 0:
                phi += 2*pi
            while phi >= 2*pi:
                phi -= 2*pi
            while alpha < 0: #alpha should be in range 0-180
                alpha += pi
            while alpha > pi: #alpha should be in range 0-180
                alpha -= pi
            
            #evaluate log probability of proposed state
            if normal is None:
                log_p_proposed = logWish(X,nobserved-1,phi,theta,alpha,e1,e2,e3,sf) #uniform prior
            else:
                log_p_proposed = logWish(X,nobserved-1,phi,theta,alpha,e1,e2,e3,sf)+ log( pp(phi,theta,normal[0],normal[1],normal[2]) )
            
            #accept or reject?
            if logURand[randIdx] <= log_p_proposed - log_p_current:
                
                log_p_current = log_p_proposed
                
                #accept and store
                chain[i+1,0] = phi
                chain[i+1,1] = theta
                chain[i+1,2] = alpha
                
                break #move on to next one
            #keep on searching
            c += 1
            
            if c > maxIter:
                print ("Warning - MCMC sampler could not find valid state. Terminating chain. (values replaced with nans).")
                chain[i+1,0] = np.nan
                chain[i+1,1] = np.nan
                chain[i+1,2] = np.nan
                return chain
    if verbose:
        print ("Sampled %d points in %d iterations. On average %.1f iterations were needed per sample." % (nsamples,iter,iter/nsamples))

    return chain

"""
Grids a set of samples (as generated by samplePosteriorMCMC) for quick plotting (see mplstereonet.contour).

**Arguments**:
-grid = the grid of points (lat,lon) pairs to grid over. See grid(...) for details.
-trace = a N x 3 array containing the MCMC trace to grid, as generated by samplePosteriorMCMC(...).

**Returns**:
-a density grid for plotting with mplstereneot.contourf or similar. Density values will sum to 1 over the domain. 
"""
def gridSamples(np.ndarray grid, double[:,:] trace):      
    #create location for output
    cdef double[:] out = np.zeros([grid.shape[1]])
    
    #calculate bin dimensions
    cdef int N = int(sqrt(grid.shape[1]))
    cdef double dl = pi / sqrt(grid.shape[1]) #latitude and longitude range from -np.pi / 2 to pi/2
    
    #map phi/theta of each sample onto grid (this becomes a 2D histogram)
    cdef int idx
    cdef double df = 1.0 / len(trace) #how much is each sample worth? 
    cdef double hp = pi/2.0 #pi on two
    cdef double x,y
    cdef double lat,lon
    cdef int i = 0
    for i in range(len(trace)):
        #convert to cartesian coords (n.b. we don't need to worry about z as we know this is a unit vector!)
        x = sin(trace[i,0]) * cos(trace[i,1])
        y = cos(trace[i,0]) * cos(trace[i,1])
    
        #convert these to lat,lon
        lat = asin(y)
        lon = asin(x/cos(lat))
        
        #calculate the index of the relevant bin
        idx = <int>((lon+hp)/dl) * N + <int>((lat+hp)/dl)
        
        #increment frequency
        out[idx] += df
    
    #return histogram :)
    return out
    
########################################################################################
###UTILITY FUNCTIONS (N.B. These are not optimised, so are no faster than pure python
########################################################################################
"""
Utility function for creating covariance matrices from phi,theta, etc.
"""
def constructCOV(double phi, double theta, double alpha, double e1, double e2, double e3 ):
    #build basis matrix containing with columns = eigenvectors
    cdef double e11,e12,e13,e21,e22,e23,e31,e32,e33
    #eigenvector 3
    e13 = sin(phi) * cos(theta)
    e23 = cos(phi) * cos(theta)
    e33 = -sin(theta)
    #eigenvector 2
    e12 = sin(phi) * sin(theta) * sin(alpha) - cos(phi) * cos(alpha)
    e22 = sin(phi) * cos(alpha) + sin(theta) * cos(phi) * sin(alpha)
    e32 = sin(alpha) * cos(theta)
    #eigenvector 1 (calculate using cross product to avoid using un-necessary trig functions)
    e11 = e23 * e32 - e33 * e22
    e21 = e33*e12 - e13 * e32
    e31 = e13*e22 - e23 * e12
    
    #calculate I = P^-1 = (basis * [[e1,0,0],[0,e2,0],[0,0,e3]] * basis_T)^-1
    #(this will have eigenvectors corresponding to the above and eigenvalues of 1,e2,e3)
    #(we compare this to the observed icov matrix)
    cdef double p11,p12,p13,p22,p23,p33
    p11 = (e1*e11**2 + e2 * e12**2 + e3 * e13**2)
    p12 = (e1*e11*e21 + e2 * e12 * e22 + e3 * e13 * e23)
    p13 = (e1*e11*e31 + e2 * e12 * e32 + e3 * e13 * e33)
    p22 = (e1*e21**2 + e2 * e22**2 + e3 * e23**2)
    p23 = (e1*e21*e31 + e2 * e22 * e32 + e3 * e23 * e33)
    p33 = (e1*e31**2 + e2 * e32**2 + e3 * e33**2)

    return np.array([[p11,p12,p13],[p12,p22,p23],[p13,p23,p33]])

"""
Utility function for building grids: builds a discretisation of the southern-hemisphere of a unit circle containing N*N points.

**Arguments**:
 N = the number of points to use in each dimension (total number of points will be N**2)
**Returns**:
 - a grid of points such that each column represents a lat,long pair on the southern-hemisphere.
"""
def grid(N=50):
    bound = pi / 2.0
    grid = np.zeros((2,N**2),dtype=np.double)
    for ix,_lat in enumerate(np.linspace(-bound,bound,N)): #loop through x-values in grd
        for iy,_lon in enumerate(np.linspace(-bound,bound,N)): #loop through y-values in grd
            grid[0,iy*N+ix] = _lat
            grid[1,iy*N+ix] = _lon
    return grid
    
    
"""
Grids the great circle containing the most uncertainty regarding the orientation (i.e. fixes the axis of rotation to the 
long axis of the trace (smallest eigenvector of icov) and grids the great-circle perpendicular to this axis). Useful for quickly evaluating
the uncertainty around an orientation, as it tends to be contained entirely within this plane.

**Arguments**:
 -cov = the covariance matrix used to determine the plane of uncertainty (the normal to this plane is the smallest eigenvector of the icov matrix)
 -N = the number of points to produce (controls the grid spacing). Default is 50.
 
**Returns**:
 - A numpy array with N rows and columns representing the latitude and longitude of the points.
"""
def gridArc(double[:,:] cov,int N=50):
    #get eigens of icov
    eval,evec = np.linalg.eig(cov)
    idx = eval.argsort()[::-1] #sort eigens in decending order...
    eval = eval[idx]
    evec = evec[:,idx] #n.b. columns of evec are vectors....
    
    #get rotation axis (w)
    W = evec[:,0]
    
    #get start vector (A)
    A = np.cross(np.array([0,0,1]),W)
    A /= np.linalg.norm(A) #normalize A
    
    #create array to store output
    arc = np.zeros((2,N))
    
    #progressively rotate A by alpha around axis W
    WA = np.cross(W,A)
    for i,alpha in enumerate(np.linspace(0,np.pi,N)):
        _lat,_lon = vec2LL( cos(alpha)*A + sin(alpha)*WA ) #n.b. rotation is calculated using rodruigez's rotation formula
        arc[0][i] = _lat
        arc[1][i] = -_lon
    return arc #return!
    
    
"""
Convert a vector to a trend and plunge.

**Arguments**:
 - xyz = a numpy array of length 3 containing the vector to convert.
**Returns**:
 - trend = the trend of the vector (in radians)
 - plunge = the plunge of the vector (in radians)
"""
def vec2TrendPlunge(double[:] xyz):
    cdef np.ndarray out = np.zeros([2])
    out[0] =  atan2(xyz[0],xyz[1])
    out[1] = -asin(xyz[2])
    
    #map to correct domain
    if (out[1] < 0):
        out[0] = atan2(-xyz[0],-xyz[1])
        out[1] = -asin(-xyz[2])
    while (out[0] < 0):
        out[0] += 2 * np.pi
    
    return out
 
"""
Convert a vector to a lat,lon coordinate.

**Arguments**:
 - xyz = a numpy array of length 3 containing the vector to convert
**Returns**:
 - lat = the latitude of the point defined by the intersection of this vector and the unit sphere
 - lon = the longitude of the point defined by the intersection of this vector and the unit sphere 
"""
def vec2LL(double[:] xyz):
    cdef double lat = asin(-xyz[1])
    return np.array( [lat, asin(xyz[0]/cos(lat))] )
    
"""
Convert a trend and plunge measurement to a cartesian vector.
**Arguments**:
   -trend = the trend of the vector to transform. 0 < trend < 2* pi
   -plunge = the plunge of the vector to transform. 0 < plunge < pi / 2
**Returns:
-xyz = a numpy array represeting the vector.
"""
def trendPlunge2Vec(double trend, double plunge):
    return np.array( [sin(trend) * cos(plunge),
                      cos(trend) * cos(plunge),
                      -sin(plunge)])
    
"""
Convert a trend and plunge to lat/lon coordinates on a unit sphere
**Arguments**:
   -trend = the trend of the vector to transform. 0 < trend < 2* pi
   -plunge = the plunge of the vector to transform. 0 < plunge < pi / 2
**Returns:**
   -lat = the latitude of the position on the unit sphere intersected by this direction
   -lon = the longitude of the position on the unit sphere intersected by this direction
"""
def trendPlunge2LL(double trend, double plunge):
    return vec2LL( trendPlunge2Vec(trend,plunge) )
    
"""
Convert a lat,lon coordinate to a trend and plunge.
**Arguments**:
 -lat,lon = the lat,lon coords to transform. Note that: -np.pi < lat,lon < np.pi.
**Returns:**
 -trend = the trend of the vector
 -plunge = the plunge of the vector
"""
def llToTrendPlunge(double lat, double lon):
    return vec2TrendPlunge(llToVec(lat,lon))

"""
Converts a lat,lon coordinate to a cartestian vector. 
**Arguments**:
 -lat,lon = the lat,lon coords to transform. Note that: -np.pi < lat,lon < np.pi.
**Returns:
-xyz = a numpy array represeting the normal vector.
"""
def llToVec(lat,lon):
    return np.array( [cos(lat) * sin(lon),
                      sin(lat), 
                      -cos(lat) * cos(lon)])
                      

############################################
#KERNEL DENSITY ESTIMATION
############################################

#The following functions can be used to construct kernel density estimates from
#SNE measurements, as this is the most convenient way of analysing such large numbers
#of samples.

#Note that we only implement KDEs for the orientation aspects as scipy already has good libraries for
#doing KDEs of linear variables (e.g. thickness). 
"""
Fast part of the code for the sphericalKDE and circularKDE functions
"""
cdef double[:] fillKDE(int npoints, int ndata, double[:,:] grid, double[:,:] data, double bandwidth, int lookupRes, bint signed):
    
    #build lookup table for kernel and for acos calcs (speed hack)
    cdef double[:] kernel = stats.norm(0,bandwidth).pdf( np.linspace(0,pi,lookupRes+1) )
    
    #create arccosine lookup table
    cdef double[:] acos = np.arccos( np.linspace(-1,1,lookupRes+1) )
    
    #loop through grid points
    cdef double dot, alpha
    cdef double nf = 1.0 / ndata #normalising factor for each kernel
    cdef double[:] out = np.zeros(npoints)
    cdef int _g, _d
    
    for _g in range(npoints):
        gx = grid[_g,0]
        gy = grid[_g,1]
        gz = grid[_g,2]
        
        for _d in range(ndata):
            #compute dot product
            dot = grid[_g,0]*data[_d,0] + grid[_g,1]*data[_d,1] + grid[_g,2]*data[_d,2]
            
            #restrict angles to 0 - 90 [for non-directional vectors such as poles or strike vectors]
            if not signed:
                dot = abs(dot)
            
            #lookup corresponding alpha in lookup table
            alpha = acos[int((dot+1)* lookupRes / 2) ]
            
            #accumulate relevant kernel value
            out[_g] += nf*kernel[int(lookupRes*alpha/pi)]
    return out
                      
"""
Do a spherical KDE on directional data (e.g. poles to planes)

**Arguments**:
 -grid = a (2,n) list of (lats,lons) to evaluate the KDE on, as created by grid(...).
 -data = a (2,n) lsit of (trend,plunge) measurements to perform the KDE on
 -bandwidth = the standard deviation of the gaussian kernal used to build the KDE
 
**Keywords**:
 -lookupRes = the resolution of the lookup tables to use. Default is 1000. 
 -degrees = if True, data are treated as angles in degrees. Default is True (i.e. use degrees). 
"""
def sphericalKDE(np.ndarray grid, np.ndarray data, double bandwidth, **kwds):

    if kwds.get("degrees",True):
        bandwidth = np.deg2rad(bandwidth)
    
    #convert grid to xyz vectors
    gxyz = []
    for _lat,_lon in np.array(grid).T:
        gxyz.append( llToVec(_lat,_lon) )
    gxyz = np.array(gxyz)
    
    #convert data to xyz vectors
    dxyz = []
    for _t,_p in np.array(data).T:
        if kwds.get("degrees",True):
            dxyz.append( trendPlunge2Vec(np.deg2rad(_t),np.deg2rad(_p)) )
        else:
            dxyz.append( trendPlunge2Vec(_t,_p) )
    dxyz = np.array(dxyz)
    
    #evaluate and return
    return np.array(fillKDE(gxyz.shape[0],dxyz.shape[0],gxyz,dxyz,bandwidth, kwds.get("lookupRes",1000),False))
  
"""
Do a circular KDE on directional data. Note that this will not work  for signed data as it only evaluates over zero to 180. The code could easily be modified to allow signed data (e.g. wind directions, slip vectors), 
but I don't need too (yet) and am a lazy shit... [to force signed directions, simply take the signed dot product rather than the absolute dot product when calculating the angle in fillKDE(...). 

**Arguments**:
 -data = a list of measurements to perform the KDE on
 -bandwidth = the standard deviation of the gaussian kernal used to build the KDE
 -signed = True if the circular data are signed vectors (e.g. wind directions or dip-directions) as opposed to directions (such as strike vectors). Default is False. 
**Keywords**:
 -outputRes = the resolution of the output KDE array. Default is 1000.
 -lookupRes = the resolution of the lookup tables to use. Default is 1000. 
 -degrees = if True, data are treated as angles in degrees. Default is True (i.e. use degrees). 
""" 
def circularKDE(np.ndarray data, double bandwidth, bint signed = False, **kwds):

    if kwds.get("degrees",True):
        bandwidth = np.deg2rad(bandwidth)
        
    #build grid vectors
    gxyz = []
    outputRes = kwds.get("outputRes",1000)
    for _t in np.linspace(0,2*np.pi,outputRes):
        gxyz.append( [np.sin(_t),np.cos(_t),0] )   
    gxyz = np.array(gxyz)
    
    #convert data to data vectors
    dxyz = []
    for _t in np.array(data):
        if kwds.get("degrees",True):
            dxyz.append( trendPlunge2Vec(np.deg2rad(_t), 0 ) )
        else:
            dxyz.append( trendPlunge2Vec(_t, 0 ) )
    dxyz = np.array(dxyz) 
    
    #evaluate
    kde = np.array(fillKDE(gxyz.shape[0],dxyz.shape[0],gxyz,dxyz,bandwidth, kwds.get("lookupRes",1000), signed))
    
    #normalise
    if kwds.get("degrees",True):
        kde = kde / np.trapz(kde,np.linspace(0,360,outputRes))
    else:
        kde = kde / np.trapz(kde,np.linspace(0,2*np.pi,outputRes))
    
    return kde
