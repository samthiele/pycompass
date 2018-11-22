import xmltodict
import mplstereonet
import matplotlib.pyplot as plt
import re
import numpy as np

'''
Class for loading/reading/sorting xml data files exported from CloudCompare
'''
class ccXML:
    '''
    Creates a dataset encapsulating the output files of a CloudCompare "Compass" interpretation.
    '''
    def __init__( self, rootpath ):
        #store root path
        self.rootpath = rootpath
        
        #load xml document using untangle
        #self.xmlRoot = untangle.parse(rootpath)
        with open(rootpath) as fd:
            self.dataDict = xmltodict.parse(fd.read())
            
    '''
    Return a list of (xml) dicts of the specified type, or an empty list. Add the "data = ..." keyword
    to search custom data dictionaries (otherwise self.dataDict is used)
    
    **Arguments**:
     - key = the xml key to match
    **Optional Keywords**:
     - data = a custom xml tree dictionary (as loaded by xmltodict)
    '''
    def filterByKey( self, key, **kwds):
        out = []
        
        dictRoot = kwds.get("data",self.dataDict)
        assert isinstance(dictRoot, dict), "Cannot filter by key: data argument is not a dict" #fail if data isn't a dict
        
        for k, v in dictRoot.items():
            #does this key match?
            if (k == key):
                if isinstance(v, dict):
                    out += [v] #append dictionary
                else: #all other values, including lists
                    out += v #yes, store [NOTE - v is a list when multiple elements have the same name - these are appended in this instance)]
                
            #recurse if value is also a dict
            if isinstance(v,dict):
                out += self.filterByKey(key, data = v)
                
            #or if data is a list (this happens when multiple nodes have the same name)
            elif isinstance(v, list): 
                for d in v: #search list for dicts
                    if isinstance(d, dict):
                        out += self.filterByKey(key, data=d)
        return out

    '''
    Returns list of the xml objects with arguments matching a given regex string. By default searches self.dataDict, though a custom object can
    also be passed
    
    **Arguments**:
     - arg = the name of the argument to search for
     - match = a regex string used to match against the argument value
    **Optional Keywords**:
     - data = a custom xml tree dictionary (as loaded by xmltodict)
    '''
    def filterByArgument( self, arg, match, **kwds):
        #get dict
        dictRoot = kwds.get("data",self.dataDict)
        
        #fail if data isn't a dict
        assert isinstance(dictRoot, dict), "Cannot filter by argument: data argument is not a dict" #fail if data isn't a dict
        
        out = []
        
        #do we match?
        if (re.search(match,dictRoot.get(arg,""))):
            out += [dictRoot]
        #recurse through children
        for k, v in dictRoot.items():
            if isinstance(v,dict): #child is a dict
                out += self.filterByArgument(arg, match, data = v)
            elif isinstance(v, list): #child is a list - expand list
                for e in v:
                    if isinstance(e, dict):
                        out += self.filterByArgument(arg, match, data = e)
                    else:
                        print (e)
        
        #return whatever we found
        return out
    
    '''
    Returns list of the xml objects with the specified name. By default searches self.dataDict, though a custom object can
    also be passed
    
    **Arguments**:
     - name = the name of the object(s) to search for
    **Optional Keywords**:
     - data = a custom xml tree dictionary (as loaded by xmltodict)
    '''
    def filterByName( self, name, **kwds):
        #get dict
        dictRoot = kwds.get("data",self.dataDict)
        assert isinstance(dictRoot, dict), "Cannot filter by name: data argument is not a dict" #fail if data isn't a dict
        return self.filterByArgument("@name", name, data = dictRoot)
        
    
    '''
    Returns the xml object with the specified id (as a dict). By default searches self.dataDict, though a custom object can
    also be passed
    
    **Optional Keywords**:
     - data = a custom xml tree dictionary (as loaded by xmltodict)
    '''
    def findByID( self, uid, **kwds):
        #get dict
        dictRoot = kwds.get("data",self.dataDict)
        assert isinstance(dictRoot, dict), "Cannot filter by id: data argument is not a dict" #fail if data isn't a dict
        
        #do the search. n.b. this could be done using filterByArgument, but it would be slower.
        
        #are we the one?
        if (dictRoot.get("@id","-1") == str(uid)): #we have a match! :)
                    return dictRoot
        
        #no - recurse through dict children
        for k, v in dictRoot.items():
            if isinstance(v,dict): #child is a dict
                out = self.findByID(uid, data = v)
                if out != None:
                    return out
            elif isinstance(v, list): #child is a list - expand list
                for e in v:
                    if isinstance(e, dict):
                        out = self.findByID(uid, data = e)
                        if out != None:
                            return out
        
        #not found!
        return None
        
    '''
    Extracts trace data as numpy arrays.
    
    **Keywords**:
     - data = the dict or subset to extract traces from. Default is the entire dataset
     **Returns**:
     - a list of (6 x n) numpy arrays who's columns contain the x,y,z and nx,ny,nz coordinates of each point in each trace
    '''
    def extractTraces( self, **kwds):
        #get dict
        dictRoot = kwds.get("data",self.dataDict)
        strict = kwds.get("strict", False)
        
        #extract traces
        traces = self.filterByKey( "TRACE", data = dictRoot)
        out = []
        for t in traces:
            if not 'POINTS' in t:
                print ("Warning - could not find point data for trace %s" % t['@id'])
                continue
            #load trace data
            x = list(map(float,t['POINTS']['x'].split(",")))
            y = list(map(float,t['POINTS']['y'].split(",")))
            z = list(map(float,t['POINTS']['z'].split(",")))

            #ensure lists are all the same length
            #assert (len(x) == len(y)) and (len(y) == len(z))

            #get trace normals
            if not 'nx' in t['POINTS']:
                if strict:
                    print ("Warning: Could not load normals for %s. This trace has been ignored." % t['@id'])
                    continue
                else:
                    nx = [0] * len(x)
                    ny = [0] * len(x)
                    nz = [1] * len(x)
            else:
                nx  = list(map(float,t['POINTS']['nx'].split(",")))
                ny  = list(map(float,t['POINTS']['ny'].split(",")))
                nz  = list(map(float,t['POINTS']['nz'].split(",")))

            #assert (len(nx) == len(ny) and len(ny) == len(nz) and len(nz) == len(z))
            out.append(np.vstack([np.array(x),np.array(y),np.array(z),np.array(nx),np.array(ny),np.array(nz)]))
        return out
    
    '''
    Extracts corresponding upper, interior and lower regions from all GeoObjects.
    
    **Keywords**:
     - data = the dict or subset to extract surfaces from. Default is the entire dataset
     **Returns**:
     - name = the name of the associated GeoObject
     - upper = the "upper" part of the GeoObject (as an xml dict)
     - interior = the "interior" part of the GeoObject (as an xml dict)
     - lower = the "lower" part of the GeoObject (as an xml dict)
     -
     - two lists of corresponding "upper" and "lower" objects (as xml dictionaries)
    '''
    def extractRegions( self, **kwds):
        #get dict
        dictRoot = kwds.get("data",self.dataDict)
        
        #get GeoObjects
        geoObjects = self.filterByKey( "GEO_OBJECT", data = dictRoot)
        
        if len(geoObjects) == 0: #no geoobjects - therefore check if dataDict is a geoObject itself
            if 'ccCompassType' in dictRoot and dictRoot['ccCompassType'] == "GeoObject":
                geoObjects = [dictRoot]
        
        names = []
        upper = []
        interior = []
        lower = []
        for g in geoObjects:
            names.append( g['@name'])
            upper += self.filterByName("Upper Boundary",data=g)
            interior += self.filterByName('Interior',data=g)
            lower += self.filterByName('Lower Boundary',data=g)
        
        return names, upper, interior, lower
        
    '''
    Plots a contoured stereonet of poles-to-planes contained in this dataset, or a custom list
    of planes.
    
    **Optional Keywords**:
     - data = a custom list of plane measurements (dictionaries with "Strike" and "Dip" keys)
     - show_outcrop = If True, the normal vectors to any TRACE objects will be contoured over the stereonet. This is 
                      useful when using orientations estimated from the trace data, as the outcrop orientation may skew the
                      results.
    **Returns**
     - fig, ax = the figure and the axis object containing the plot
    '''
    def plotPlanes(self, **kwds):
        #get the dataset
        dataset = kwds.get("data",[self.dataDict])
        
        #init plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='stereonet')

        #load orientations lists
        strikes = []
        dips = []
        dip_dirs = []
        
        #loop through data dicts
        for o in dataset:
            planes = self.filterByKey('PLANE', data = o)
            for p in planes:
                strikes += [float(p['DipDir'])-90] #n.b. we use this to avoid ambiguity in strike direction (though Compass uses british RHR)
                dips += [float (p['Dip'])]
                #dip_dirs += [float (p['DipDir'])]
        #plot poles
        ax.pole(strikes, dips, 'x', markersize=6, color='k')       
        
        #plot contours
        cax = ax.density_contourf(strikes, dips, measurement='poles', cmap = "Blues")
        fig.colorbar(cax)

        
        #plot outcrop orientation to give indication of uncertainty/outcrop bias
        if (kwds.get("show_outcrop",False)):
            #list to store normals in
            normals = [[],[],[]] #[x-coords, y-coords, z-coords]
            
            #loop through trace datasets
            for o in dataset:
                traces = self.filterByKey('TRACE', data = o)
                for t in traces:
                    
                    #does this trace have normals?
                    if t['POINTS']['@normals'] == "True":
                        #extract normal coords
                        normals[0] += list(map(float,t['POINTS']['nx'].split(",")))
                        normals[1] += list(map(float,t['POINTS']['ny'].split(",")))
                        normals[2] += list(map(float,t['POINTS']['nz'].split(",")))
                        
             
            #plot normals as contours
            plunge, bearing = mplstereonet.vector2plunge_bearing(normals[0], normals[1], normals[2])
            cax = ax.density_contour(plunge, bearing, measurement='lines', cmap = "winter")
            fig.colorbar(cax)
            #ax.line(plunge, bearing, marker='o', color='grey', markersize=1)
            
        #draw grid
        ax.set_xticks( [x / (2*np.pi) for x in range(-180,180,45)] )  
        ax.set_yticks( [x / (2*np.pi) for x in range(-180,180,45)] ) 
        ax.grid()
        
        return fig, ax

    '''
    Plots a rose diagram of planes contained in this dataset, or a custom list
    of planes.
    
    **Optional Keywords**:
     - data = a custom list of plane measurements (dictionaries with "Strike" and "Dip" keys)
     - width = the width of each bin
     - symmetric = True. if False, bins are separated by 180 degrees are not summed.
     - offset = a rotation to apply to the whole rose diagram (e.g. use +90 to plot dip-dir rather than strike)
    **Returns**
     - fig, ax = the figure and the axis object containing the plot
    '''
    def plotRose(self, **kwds):
        #get the dataset
        dataset = kwds.get("data",[self.dataDict])
        width = kwds.get("width",10)
        symmetric = kwds.get("symmetric",True)
        offset = kwds.get("offset", 0)
        
        #init plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')

        #load orientations lists
        strikes = []
        dips = []
        dip_dirs = []
        
        #loop through data dicts
        for o in dataset:
            planes = self.filterByKey('PLANE', data = o)
            for p in planes:
                strikes += [float(p['DipDir'])-90+offset] #n.b. we use this to avoid ambiguity in strike direction (though Compass uses british RHR)
                dips += [float (p['Dip'])]
                #dip_dirs += [float (p['DipDir'])]
        
        #build histogram of strike
        bin_edges = np.arange(-5,366,width)
        count, bin_edges = np.histogram(strikes, bin_edges) #build histogram counts
        count[0] += count[-1] #sum first and last values (due to cyclicity)
        
        if symmetric:
            half = np.sum(np.split(count[:-1], 2), 0)
            count = np.concatenate([half, half])
            ax.bar(np.deg2rad(np.arange(0, 360, width)), count, width=np.deg2rad(width), bottom=0.0, color='.8', edgecolor='k')
        else:
            ax.bar(np.deg2rad(np.arange(0, 360, width)), count[:-1], width=np.deg2rad(width), bottom=0.0, color='.8', edgecolor='k')
        ax.set_theta_zero_location('N') #north is zero not x-axis
        ax.set_theta_direction(-1) #geographic not mathematical rotations
        ax.set_rlabel_position(180)
        return fig, ax
    
    '''
    Plots a contoured stereonet of poles-to-planes representing two structure types (as specified by the names)
    
    **Arguments**:
     - structures = a list of regex strings used to define different structure types (by name)
    **Optional Keywords**:
     - data = a custom list of plane measurements (dictionaries with "Strike" and "Dip" keys)
     - show_outcrop = If True, the normal vectors to any TRACE objects will be contoured over the stereonet. This is 
                      useful when using orientations estimated from the trace data, as the outcrop orientation may skew the
                      results.
     - symbols = a list of symbols corresponding to each structure type in the structure list. Default is ".".
     - colours = a list of colours corresponding to each structure type in the structures list. Default is 'k'.
     - contours = a list defining contour types for each structure type. Can be "Line", "Filled" or "None".
     - mean = a list containing True if a mean-plane should be drawn for each structure type. Default is False
     - values = a list of lists defining the values for each contour for each structure type. Default is None (automatically generate values)     
     - cmap = a list of cmaps to use for each structure. Default is "Blues".
     - marker_size = a list of marker sizes for each structure. Default is 6.
     - alphas = a list of alpha values for each structure. Default is 0.75.
     - showCMaps = if True colour ramps are drawn for each contour plot
     - models = A list of models (one for each structure type) if inliers and outliers are to be plotted separately. (cf. RadialStats.RadialModel)
     - outlier_thresh = A list of thresholds used in combination with the models to define outliers (cf. RadialStats.RadialModel) 
    **Returns**
     - fig, ax, cax = the figure and the axis object containing the plot and a list of contour objects
    '''
    def plotMultiPlanes(self, structures, **kwds):
        #get the dataset
        dataset = kwds.get("data",[self.dataDict])
        
        #split into structure types
        structure_data = [ self.filterByName(regx) for regx in structures ]
        
        #get drawing kwds
        symbols = kwds.get("symbols",['.'] * len(structure_data))
        cols = kwds.get("colours", ['k'] * len(structure_data))
        contours = kwds.get("contours",["None"] * len(structure_data))
        cmaps = kwds.get("cmap",["Blues"] * len(structure_data))
        size = kwds.get("marker_size", [6] * len(structure_data))
        alphas = kwds.get("alphas", [0.75] * len(structure_data))
        values = kwds.get("values", None)
        showCMaps = kwds.get("showCMaps",True)
        mean = kwds.get("mean",[False] * len(structure_data))
        models = kwds.get("models",[None] * len(structure_data))
        outlier_thresh = kwds.get("outlier_thresh",[None] * len(structure_data))
        
        #length of all arrays should match
        assert len(structure_data) == len(symbols), "Error - symbols array is the incorrect length."
        assert len(structure_data) == len(cols), "Error - symbols cols is the incorrect length."
        assert len(structure_data) == len(contours), "Error - contours array is the incorrect length."
        assert len(structure_data) == len(cmaps), "Error - cmaps array is the incorrect length."
        assert len(structure_data) == len(size), "Error - size array is the incorrect length."
        assert len(structure_data) == len(alphas), "Error - alphas array is the incorrect length."
        assert len(structure_data) == len(mean), "Error - mean array is the incorrect length."
        assert len(structure_data) == len(models), "Error - models array is the incorrect length."
        assert len(structure_data) == len(outlier_thresh), "Error - outlier_thresh array is the incorrect length."
        if values != None:
            assert len(structure_data) == len(values)

        #init plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='stereonet')
        cax = []
        #plot each structure type
        for i,data in enumerate(structure_data):
            #init orientations lists
            strikes = []
            dips = []
            x = []
            y = []
            #loop through data finding plane objects
            for o in data:
                planes = self.filterByKey('PLANE', data = o)
                for p in planes:
                    strikes += [float(p['DipDir'])-90] #n.b. we use this to avoid ambiguity in strike direction (though Compass uses british RHR)
                    dips += [float (p['Dip'])]
                    x += [float (p['Cx'])]
                    y += [float (p['Cy'])]
            
            strikes = np.array(strikes)
            dips = np.array(dips)
            x = np.array(x)
            y = np.array(y)
            
            #outlier detection?
            out_mask = np.array([True] * len(strikes)) #default is all outliers
            if models[i] != None and outlier_thresh[i] != None: #do we have enough info to determine outliers?
                print ("\tComputing outliers for threshold %d" % outlier_thresh[i])
                strike_pred=models[i].predict(np.stack([x,y]).T) #generate predicted values given the model
                diff = models[i].loss(strikes,strike_pred) #calculate the difference between predicted and observed
                out_mask = diff > outlier_thresh[i] #define outliers (boolean array)
                
            in_mask = np.logical_not(out_mask) #calculate inliers

            #plot poles
            #ax.pole(strikes[out_mask], dips[out_mask], symbols[i], markersize=size[i], color=cols[i], alpha=alphas[i])       
            #ax.pole(strikes[in_mask], dips[in_mask], '^', markersize=size[i], color=cols[i], alpha=alphas[i])
            ax.pole(strikes,dips,symbols[i], markersize=size[i], color=cols[i], alpha=alphas[i])
            
            #plot model?
            if models[i] != None and outlier_thresh[i] != None:
                #calculate centroid of inlier points 
                cx = np.mean(x[in_mask])
                cy = np.mean(y[in_mask])
                
                #calculate expected strike given this point
                strike_pred = models[i].predict(np.stack([[cx],[cy]]).T)[0]
                
                #plot domain
                ax.plane(strike_pred-outlier_thresh[i]+90,90,'--',c='gray') #n.b. + 90 as we are mapping the domain of poles not the actual planes!
                ax.plane(strike_pred+outlier_thresh[i]+90,90,'--',c='gray')
                
                #plot radial line
                ax.plane(strike_pred,90,'-',c='gray')
                
            #plot contours?
            if contours[i] == "Line":
                if (values == None):
                    cax.append( ax.density_contour(strikes, dips, measurement='poles', cmap = cmaps[i], sigma=2) )
                else:
                    cax.append( ax.density_contour(strikes, dips, levels=values[i], measurement='poles', cmap = cmaps[i], sigma=2) )
                if showCMaps:
                    fig.colorbar(cax[-1])
            if contours[i] == "Filled":
                if (values == None):
                    cax.append( ax.density_contourf(strikes, dips, measurement='poles', cmap = cmaps[i], sigma=2) )
                else:
                    cax.append( ax.density_contourf(strikes, dips, levels=values[i], measurement='poles', cmap = cmaps[i], sigma=2) )
                if showCMaps:
                    fig.colorbar(cax[-1])

            #plot n-observations
            xloc = 0.95
            yloc = -0.04 + 0.032 * i
            ax.text(xloc,yloc, "n=%d" % len(strikes),color=cols[i],transform = ax.transAxes, ha='left', va='center')
            
            #calculate mean?
            if mean[i] == True:
                if len(strikes) > 0:
                    strike, dip = mplstereonet.fit_pole(strikes,dips,measurement='poles')           
                    ax.plane(strike,dip,cols[i])
                
            
            
        #plot outcrop orientation to give indication of uncertainty/outcrop bias
        if (kwds.get("show_outcrop",False)):
            #list to store normals in
            normals = [[],[],[]] #[x-coords, y-coords, z-coords]
            
            #loop through trace datasets
            for o in dataset:
                traces = self.filterByKey('TRACE', data = o)
                for t in traces:
                    
                    #does this trace have normals?
                    if t['POINTS']['@normals'] == "True":
                        #extract normal coords
                        normals[0] += list(map(float,t['POINTS']['nx'].split(",")))
                        normals[1] += list(map(float,t['POINTS']['ny'].split(",")))
                        normals[2] += list(map(float,t['POINTS']['nz'].split(",")))
                        
             
            #plot normals as contours
            plunge, bearing = mplstereonet.vector2plunge_bearing(normals[0], normals[1], normals[2])
            cax = ax.density_contour(plunge, bearing, measurement='lines', cmap = "winter")
            fig.colorbar(cax)
            
        #draw grid
        ax.set_xticks( [x / (2*np.pi) for x in range(-180,180,45)] )  
        ax.set_yticks( [x / (2*np.pi) for x in range(-180,180,45)] ) 
        ax.grid()
        
        return fig, ax, cax     
"""
Returns the goeref shift values I use for La Palma.... I should really store this in the xml files sometime?
"""

"""
Object that can hold data assocated with a GeoObject, as defined by the Compass plugin 
in CloudCompare.
"""
import xmltodict

class GeoObject:

	"""
	Create a GeoObject object from an xml node loaded from xml using IOTools
	"""
	def __init__( self, data, xmlNode ):
		name = xmlNode['@name']
		id = xmlNode['@id']
		
		#get pinch-nodes
		interior = data.filterByName("Interior")
		for node in interior:
			nodes = data.filterByKey("PINCH_NODE",data=interior)

		#get upper and lower traces
		upper = data.filterByName("Upper Boundary",data=data)
		lower = data.filterByName("Lower Boundary",data=data)

		for node in upper:
			tUpper = data.extractTraces(data=upper)
		
		for node in lower:
			tLower = data.extractTraces(data=lower)
		
		upperTraces = []
		lowerTraces = []
		
		
		
def getGeorefShift():
    return -218000,-3182000,-1000 #todo - store georef shift in the xml files?
    
"""
Converts local coords of the format [[x1,x2,x3,...],[y1,y2,y3,...],[z1,z2,z3,...] to global UTM coords by subtracting the georef shift
"""
def localToUTM( xyz ):
    xyz = np.array(xyz)
    if xyz.shape[0] != 3:
        xyz = xyz.T
    assert xyz.shape[0] == 3, "Error - xy is the wrong shape. Make sure it contains xyz coords in separate arrays."
    return xyz[0] - getGeorefShift()[0], xyz[1] - getGeorefShift()[1], xyz[2] - getGeorefShift()[2]

"""
Converts global coords of the format [[x1,x2,x3,...],[y1,y2,y3,...]] to global UTM coords by adding the georef shift
"""
def globalToUTM( xyz ):
    xyz = np.array(xyz)
    if xyz.shape[0] != 3:
        xyz = xyz.T
    assert xyz.shape[0] == 3, "Error - xy is the wrong shape. Make sure it contains xyz coords in separate arrays."
    assert xy.shape[0] == 3, "Error - xy is the wrong shape. Try transposing it."
    return xyz[0] + getGeorefShift()[0], xyz[1] + getGeorefShift()[1], xyz[2] + getGeorefShift()[2]
