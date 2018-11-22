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
		
		
		
		