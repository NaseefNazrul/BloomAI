Bloom Probability checker 

1) user drops pin on the map gets lat/lon values 
2) then we get data from gee for the closest NDVI/NDWI data 
3) then we feed that data to the ML
4) it should return a json file to the frontend
5) EG: Based on current satellite data, this location has a 75% chance of having wildflower blooms right now 
probabilty will be the confidence level