import sys, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.stats
import math
import mplstereonet

#load SNE
from pycompass.IOTools import ccXML
from pycompass.SNE import pdf


"""
Simple class for storing, visualising and performing common tests on SNE data. 

Sam Thiele 2019
"""
class SNEList:
    
    """
    Utility function to quickly load SNE from a data dictionary as loaded by ccXML. 
    
    **Arguments**:
     - dataset = the ccXML object containing the data.
     - o = the entry point (typically a GeoObject) to enter the data tree at. 
    
    **Returns**:
     - a SNEList object. 
    """
    def loadFromData( data, o ):
        
        #search for SNEs
        SNEs = data.filterByName("SNE_Samples",data=o)
    
        #extract data
        trend = []
        plunge = []
        thick = []
        x = []
        y = []
        z = []
        for s in SNEs:
            #gather data from SNE points
            points = s["POINTS"]
            x += list(np.fromstring( points["x"], dtype=np.float, sep=','))
            y += list(np.fromstring( points["y"], dtype=np.float, sep=','))
            z += list(np.fromstring( points["z"], dtype=np.float, sep=','))
            trend += list(np.fromstring( points["trend"], dtype=np.float, sep=','))
            plunge += list(np.fromstring( points["plunge"], dtype=np.float, sep=','))
            thick += list(np.fromstring( points["thickness"], dtype=np.float, sep=','))
        
        return SNEList( x, y, z, trend, plunge, thick )
    
    """
    Construct SNE object.
    """
    def __init__( self, x=[], y=[], z=[], trend=[], plunge=[], thickness=[]):
    
        #cast input to numpy arrays and store
        self.trend = np.array(trend)
        self.plunge = np.array(plunge)
        self.thickness = np.array(thickness)
        self.pos = np.vstack([x,y,z])
        
        #init KDEs to none
        self.trendKDE = None
        self.plungeKDE = None
        self.thickKDE = None
        self.oriKDE = None
     
    """
    Return a new (copy) SNEList that combines SNEs from this object and another one. 
    """
    def merge( self, SNEList2 ):
        out = SNEList()
        out.append(SNEList2)
        return out
    
    """
    Add data from a SNEList2 to this one. 
    """    
    def append( self, SNEList2 ):
        self.trend = np.append(self.trend,SNEList2.trend)
        self.plunge = np.append(self.plunge,SNEList2.plunge)
        self.thickness = np.append(self.thickness,SNEList2.thickness)
        self.pos = np.append(self.pos,SNEList2.pos,axis=1)
        
        #reset any KDEs
        self.trendKDE = None
        self.plungeKDE = None
        self.thickKDE = None
        self.oriKDE = None
    
    """
    Bins data in this SNEList 

    **Arguments**:
     - vals = a list of the same length as this SNEList containg the value to decide which bin each measurement belongs
     - binwidth = the size of each bin
     - minv = the value to start binning at
     - maxv = the value to end binning at

    **Returns**:
     - bins = the lower boundary of each bin
     - values = a SNEList object containing the data in this bin, or None if no SNEs fall within it. 
    """
    def bin(self,vals,binwidth,minv,maxv):
        vals = np.array(vals)
        bins = np.arange(minv,maxv,binwidth)
        values = [None for n in range(len(bins))]
        
        #loop through bins
        for i in range(len(bins)):
            _min = bins[i]
            _max = bins[i] + binwidth
            mask = np.logical_and(vals > _min, vals < _max )
            if sum(mask) > 0:
                values[i] = SNEList(self.pos[0,mask],
                                    self.pos[1,mask],
                                    self.pos[2,mask],
                                    self.trend[mask],
                                    self.plunge[mask],
                                    self.thickness[mask])
        
        #return
        return bins,values
    
    ############################
    ##STATISTICAL INTERROGATION
    ############################
    
    """
    Returns the specified KDE, computing it if necessary (the first time it's requested). 
    
    **Arguments**:
     -variable = the KDE to compute/retrieve. Should be "strike", "dip", "trend", "plunge" or "thickness" (1D KDEs) or "orientation" (2D).
    **Keywords**:
     -bw = the bandwidth of the kde to calculate. Default is 2.5 degrees for angular data and 5.0 cm for thickness data. 
     -recalc = if True, this forces the KDE to be recalculated. 
    **Returns**:
     -grid = the grid points associated with each KDE. Will be a 1D list of x coords for 1D KDEs and a 2D grid for "orientation".
     -KDE = the kernel density estimates associated with each grid point. 
    
    """
    def getKDE(self,variable,**kwds):
        
        #get bandwidth
        bw = kwds.get("bw",5.0)
        recalc = kwds.get("recalc",False)
        
        if "trend" in variable.lower():
            #compute KDE?
            if self.trendKDE is None or recalc:
                #evaluate KDE
                self.trendKDE = pdf.circularKDE( np.array(self.trend), bw, outputRes=360, signed = True )
                
                #build corresponding grid
                self.trendGrid = np.linspace(0,360,len(self.trendKDE))
                
            #return
            return np.array(self.trendGrid), np.array(self.trendKDE)
        
        if "plunge" in variable.lower():
            if self.plungeKDE is None or recalc:
                
                #evaluate on signed plunge
                self.plungeKDE = pdf.circularKDE( self.plunge, bw, outputRes=360, signed = False )
                
                #mirror and extract domain 0-90 degrees
                self.plungeKDE[0:180] += np.flip(self.plungeKDE[180:360],axis=0) 
                self.plungeKDE[0:90] += np.flip(self.plungeKDE[90:180],axis=0)
                self.plungeKDE = self.plungeKDE[0:90]
                
                #mirror
                #self.plungeKDE[0:90] += self.plungeKDE[180:90]
                #self.plungeKDE = self.plungeKDE[0:90]
                
                #build grid
                self.plungeGrid = np.linspace(0,90,len(self.plungeKDE))
                
            return np.array(self.plungeGrid), np.array(self.plungeKDE)
        
        if "strike" in variable.lower():
            grid,kde = self.getKDE("trend")
            kde[0:180] += kde[180:360] #mirror
            kde[180:360] = kde[0:180]
            
            #phase shift by 90 degrees
            strikeKDE = np.array( [ kde[ (i+90) % 360] for i in range(360) ] )
            
            return np.array(grid),strikeKDE
        
        if "dip" in variable.lower():
            grid,kde = self.getKDE("plunge")
            return np.array(grid), np.flip(kde,axis=0)
        
        if "thick" in variable.lower():
            if self.thickKDE is None or recalc:
                #get range data for thickness KDE
                gmin = kwds.get("gmin",0)
                gmax = kwds.get("gmax",np.max(self.thickness))
                res = kwds.get("res",1000)
                bw_method = kwds.get("bw",0.025)
                
                #calculate bw for the gaussian kernal
                #(scipy's definition the kernel standard deviation = passed bandwidth * standard deviation of the sample
                if not bw is None:
                    bw_method = bw / np.std(self.thickness)
                
                #build grid
                self.thickGrid = np.linspace(gmin,gmax,res)

                #evaluate KDE
                self.thickKDE = scipy.stats.gaussian_kde(self.thickness,bw_method=bw_method)(self.thickGrid)
                
            return np.array(self.thickGrid), np.array(self.thickKDE)
        
        if "ori" in variable.lower():
            res = kwds.get("res",100)
            
            #check already computed
            if self.oriKDE is None or recalc:
                #make grid
                self.oriGrid = pdf.grid(100)

                #calculate KDE
                self.oriKDE = pdf.sphericalKDE(self.oriGrid,np.array([self.trend,self.plunge]),bw,degrees=True)

            return np.array(self.oriGrid), np.array(self.oriKDE)
        
        #error
        assert False, "Variable '%s' is not recognised." % variable
    
    """
    Evaluate probabilities by integrating the strike, dip or thickness KDEs. Note that this doesn't wrap angular data...
    
    **Arguments**:
     -variable = the variable to evaluate. Should be a string that matches "strike", "dip" or "thickness". 
     -minv = the minima of the section of pdf to integrate. Must range between 0 and 360 for strike/trend and 0 to 90 for dip. 
     -maxv = the maxima of the section of pdf to integrate. Must range between 0 and 360 for strike/trend and 0 to 90 for dip. 
    
    **Keywords**:
     -keywords are passed to the getKDE(....=) function.
    """
    def evalP1D( self, variable, minv, maxv, **kwds ):
        
        assert minv < maxv, "Don't be stupid (or, if evaluating angles such as the sector 315 to 045, split into two sections)."
        
        #get grid, kde
        grid,kde = self.getKDE(variable)
        
        #normalise kde
        kde /= np.trapz(kde,grid)
        
        #special case: strike ranges 0 to 180 (not 0 to 360), so we need to double kde vals
        if "strike" in variable.lower():
            kde *= 2
            
        #mask values to integrate
        mask = np.logical_and(grid >= minv,grid <= maxv)
        
        #integrate and return
        return np.trapz(kde[mask],grid[mask])        
        
    """
    Calculates the mean SNE orientation and returns it as an xyz unit vector. 
    """
    def getMeanVector(self):
        res = np.zeros(3)
        init = pdf.trendPlunge2Vec( np.deg2rad(self.trend[0]),np.deg2rad(self.plunge[0]))
        for i in range(1,len(self.trend)):
            vec = pdf.trendPlunge2Vec( np.deg2rad(self.trend[i]),np.deg2rad(self.plunge[i]))
            if np.dot(init,vec) > 0:
                res += vec
            else:
                res -= vec
        return res/np.linalg.norm(res)
    
    """
    Calculates the mean SNE orientation and returns it as a trend and plunge (in degrees). 
    """
    def getMeanTrendPlunge(self):
        return np.rad2deg( pdf.vec2TrendPlunge( self.getMeanVector() ) )
    
    """
    Calculates the mean SNE thickness and returns it. 
    """
    def getMeanThickness(self):
        return np.mean(self.thickness)
    
    ##############################
    ##PLOTTING
    ##############################
    """
    Plot a 1D KDE. 
    
    **Arguments**:
     -variable = the variable to plot. Should be a string that matches "strike", "dip" or "thickness". 
    """
    def plotKDE1D( self, variable, **kwds ):
        fig = plt.figure()
        grid,kde = self.getKDE(variable,**kwds)
        plt.plot(grid,kde)
        return fig, plt.gca()
        
    """
    Plot a rose diagram of the "strike" KDE.
    
    **Keywords**:
     - plotArgs = a dict of keywords to pass to the plot(...) function to modify the rose diagram outline.
     - fillArgs = a dict of keywords to pass to the fill(...) function to modify rose diagram fill. 
     
    **Returns**:
     - fig, ax = the figure and axes containing the rose diagram. 
    """
    def plotRose( self, **kwds ):
        _theta,kde = self.getKDE("strike",**kwds)
        
        #make figure?
        if "ax" in kwds:
            ax = kwds.get("ax")
            fig = ax.figure
        else:
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111, projection='polar')

        #get args
        plotArgs = kwds.get("plotArgs",{"color":'k'})
        fillArgs = kwds.get("fillArgs",{"color":'gray',"alpha":0.25})
        
        #plot
        ax.plot( np.deg2rad(_theta),kde, **plotArgs)
        ax.fill( np.deg2rad(_theta),kde, **fillArgs)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.get_yaxis().set_visible(False)
        return fig, ax
        
    """
    Plot a 2D orientation KDE on a stereonet.
    
    **Keywords**:
     - kwds are passed directly to plt.contourf(...).  
     
    **Returns**:
     - fig, ax = the figure and axes containing the rose diagram. 
     
    """
    def plotStereonet( self, **kwds ):
        #get kde
        grid,kde = self.getKDE("orientation",degrees=False,**kwds)

        #reshape kde into a grid
        res = int(np.sqrt(grid.shape[1]))
        assert res == np.sqrt(grid.shape[1]), "Error plotting KDE - supplied grid must be square."
        kde2D = np.reshape(kde,(res,res)).T

        #build meshgrid of plotting coordinates (in lat,lon)
        bound = np.pi / 2
        X,Y = np.meshgrid(np.linspace(-bound,bound,res), np.linspace(-bound,bound,res)) #rectangular plot of polar data

        #plot
        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(111, projection='stereonet')
        
        if not "cmap" in kwds:
            kwds["cmap"] = "coolwarm"
        cs = ax1.contourf(X,Y,kde2D,**kwds)

        return fig, ax1      
        
        
        
        
"""
Class for combining multiple SNELists (each representing different structures) with equal weighting (rather than just merging the SNEs, which weights based on the number of SNEs for each structure).
"""
class CombinedSNEList:
    """
    Init a combinedSNEList object.
    
    **Arguments**:
     - name = a name for this set of structures
     - SNEs = a list of SNEList objects which will be combined with equal weighting. 
    """
    def __init__(self, name, SNEs):
        self.name = name
        self.SNEs = SNEs
        assert len(SNEs) > 0, "Error -CombinedSNEList must contain at least one SNEList...."
    """
    Calculates a combined KDE by summing KDEs of each structure in this combined SNE List.
    
    **Arguments**:
     -variable = the KDE to compute/retrieve. Should be "strike", "dip", "trend", "plunge" or "thickness" (1D KDEs) or "orientation" (2D).
    **Keywords**:
     -bw = the bandwidth of the kde to calculate. Default is 2.5 degrees for angular data and 5.0 cm for thickness data. 
     -recalc = if True, this forces the KDE to be recalculated. 
     -gmin,gmax = required for thickness kdes to specify range of values to evaluate kde over. 
    **Returns**:
     -grid = the grid points associated with each KDE. Will be a 1D list of x coords for 1D KDEs and a 2D grid for "orientation".
     -KDE = the kernel density estimates associated with each grid point. 
    """
    def getKDE(self,variable,**kwds):
        
        #set default gmin/gmax for thickness kde
        if "thick" in variable:
            kwds["gmin"] = kwds.get("gmin",0.0)
            kwds["gmax"] = kwds.get("gmax",10)
        
        #get grid and KDE from first SNE
        grid,kde = self.SNEs[0].getKDE(variable,**kwds)
        
        #sum all others onto original (thus forcing equal weighting)
        for i in range(1,len(self.SNEs)):
            _,_k = self.SNEs[i].getKDE(variable,**kwds)
            kde += _k
            
        #normalise    
        kde /= len(self.SNEs)
        
        return grid,kde
    
    
    """
    Evaluate probabilities by integrating the strike, dip or thickness KDEs. Note that this doesn't wrap angular data...
    
    **Arguments**:
     -variable = the variable to evaluate. Should be a string that matches "strike", "dip" or "thickness". 
     -minv = the minima of the section of pdf to integrate. Must range between 0 and 360 for strike/trend and 0 to 90 for dip. 
     -maxv = the maxima of the section of pdf to integrate. Must range between 0 and 360 for strike/trend and 0 to 90 for dip. 
    
    **Keywords**:
     -keywords are passed to the getKDE(....=) function.
    """
    def evalP1D( self, variable, minv, maxv, **kwds ):
        
        assert minv < maxv, "Don't be stupid (or, if evaluating angles such as the sector 315 to 045, split into two sections)."
        
        #get grid, kde
        grid,kde = self.getKDE(variable)
        
        #special case: strike ranges 0 to 180 (not 0 to 360), so we need to double kde vals
        if "strike" in variable.lower():
            kde *= 2
            
        #mask values to integrate
        mask = np.logical_and(grid >= minv,grid <= maxv)
        
        #integrate and return
        return np.trapz(kde[mask],grid[mask])     
        
    """
    Plot a 1D KDE. 
    
    **Arguments**:
     -variable = the variable to plot. Should be a string that matches "strike", "dip" or "thickness". 
    """
    def plotKDE1D( self, variable, **kwds ):
        fig = plt.figure()
        grid,kde = self.getKDE(variable,**kwds)
        plt.plot(grid,kde)
        return fig, plt.gca() 
        
    """
    Plot a rose diagram of the "strike" KDE.
    
    **Keywords**:
     - plotArgs = a dict of keywords to pass to the plot(...) function to modify the rose diagram outline.
     - fillArgs = a dict of keywords to pass to the fill(...) function to modify rose diagram fill. 
     
    **Returns**:
     - fig, ax = the figure and axes containing the rose diagram. 
    """
    def plotRose( self, **kwds ):
        _theta,kde = self.getKDE("strike",**kwds)
        
        #make figure?
        if "ax" in kwds:
            ax = kwds.get("ax")
            fig = ax.figure
        else:
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111, projection='polar')

        #get args
        plotArgs = kwds.get("plotArgs",{"color":'k'})
        fillArgs = kwds.get("fillArgs",{"color":'gray',"alpha":0.25})
        
        #plot
        ax.plot( np.deg2rad(_theta),kde, **plotArgs)
        ax.fill( np.deg2rad(_theta),kde, **fillArgs)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.get_yaxis().set_visible(False)
        return fig, ax
    
    
    """
    Plot a 2D orientation KDE on a stereonet.
    
    **Keywords**:
     - kwds are passed directly to plt.contourf(...).  
     
    **Returns**:
     - fig, ax = the figure and axes containing the rose diagram. 
     
    """
    def plotStereonet( self, **kwds ):
        #get kde
        grid,kde = self.getKDE("orientation",degrees=False,**kwds)

        #reshape kde into a grid
        res = int(np.sqrt(grid.shape[1]))
        assert res == np.sqrt(grid.shape[1]), "Error plotting KDE - supplied grid must be square."
        kde2D = np.reshape(kde,(res,res)).T

        #build meshgrid of plotting coordinates (in lat,lon)
        bound = np.pi / 2
        X,Y = np.meshgrid(np.linspace(-bound,bound,res), np.linspace(-bound,bound,res)) #rectangular plot of polar data

        #plot
        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(111, projection='stereonet')
        
        if not "cmap" in kwds:
            kwds["cmap"] = "coolwarm"
        cs = ax1.contourf(X,Y,kde2D,**kwds)

        return fig, ax1
