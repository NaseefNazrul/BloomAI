// vars to be declared there are more in the beginning but not shown here
var YEARS = [2020, 2021, 2022, 2023, 2024, 2025];
var START_MONTH = 1;
var END_MONTH = 12;
var MAX_CLOUD_PERCENT = 80;

var numPolygons = non_bloom_sites.coordinates().length();

var indices = ee.List.sequence(0, numPolygons.subtract(1));

var nonBloomSites = ee.FeatureCollection(
  indices.map(function(index) {
    var polygonCoords = ee.List(non_bloom_sites.coordinates().get(index));

    var geometry = ee.Geometry.Polygon(polygonCoords);
    var centroid = geometry.centroid();
    var coordinates = centroid.coordinates();
    
    return ee.Feature(geometry, {
      'id': ee.Number(1000).add(index),
      'Site': ee.String('Non_Bloom_').cat(ee.Number(index).add(1).format('%d')),
      'Type': 'Non_Bloom',
      'Season': 'N/A',
      'Area': geometry.area(),
      'location_type': 'non_bloom',
      'longitude': coordinates.get(0),
      'latitude': coordinates.get(1)
    });
  })
);

print('Loaded non-bloom sites:'+ nonBloomSites.size());
print('First non-bloom site:'+ nonBloomSites.first());

// masking over maps
function maskLandsatClouds(image) {
  var qa = image.select('QA_PIXEL');
  var cloudMask = qa.bitwiseAnd(1 << 3).eq(0)
                  .and(qa.bitwiseAnd(1 << 4).eq(0))
                  .and(qa.bitwiseAnd(1 << 5).eq(0));
  return image.updateMask(cloudMask);
}

// iterating through each yaer in list 
function processYear(year) {
  var startDate = ee.Date.fromYMD(year, START_MONTH, 1);
  var endDate = ee.Date.fromYMD(year, END_MONTH, 30);
  
  var landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(nonBloomSites)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUD_COVER', MAX_CLOUD_PERCENT))
    .map(maskLandsatClouds);

  var landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterBounds(nonBloomSites)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUD_COVER', MAX_CLOUD_PERCENT))
    .map(maskLandsatClouds);

  var landsat = landsat8.merge(landsat9);

  var withIndices = landsat.map(function(image) {
    var ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
    var ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
    
    return image.addBands([ndvi, ndwi])
      .set('system_date', image.date().format('YYYY-MM-dd'))
      .set('year', year);
  });

  return withIndices;
}

// ===== TIME SERIES EXTRACTION =====
var allYearsCollection = ee.ImageCollection(ee.FeatureCollection(YEARS.map(processYear)).flatten());

print('Total images across all years:' + allYearsCollection.size());

var timeSeries = allYearsCollection.map(function(image) {
  return image.select(['NDVI', 'NDWI']).reduceRegions({
    collection: nonBloomSites,
    reducer: ee.Reducer.mean(),
    scale: 30,
    tileScale: 2
  }).map(function(feature) {
    return feature.set({
      'date': image.get('system_date'),
      'year': image.get('year'),
      'cloud_cover': image.get('CLOUD_COVER')
    });
  });
}).flatten();

// Filter out any failed reductions
timeSeries = timeSeries.filter(ee.Filter.notNull(['NDVI']));

// exporting
Export.table.toDrive({
  collection: timeSeries,
  description: 'non_bloom_sites_timeseries_2020_2025',
  folder: 'EarthEngineExports',
  fileFormat: 'CSV',
  selectors: ['id', 'Site', 'Type', 'location_type', 'date', 'year', 
             'NDVI', 'NDWI', 'cloud_cover', 'longitude', 'latitude']
});

// barely works tho
// Use a simple image for visualization instead of processYear
var landsatImage = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(nonBloomSites)
  .filterDate('2023-06-01', '2023-06-30')
  .first();

var ndvi = landsatImage.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
Map.addLayer(ndvi, {min: -0.2, max: 0.8, palette: ['white', 'yellow', 'green']}, 'Sample NDVI 2023');
Map.addLayer(nonBloomSites, {color: 'blue'}, 'Non-Bloom Sites');

print('===== Alhamdulillah =====');