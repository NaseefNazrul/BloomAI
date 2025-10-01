// vars to be declared
var YEARS = [2020, 2021, 2022, 2023, 2024, 2025];
var SPRING_START = '03-01';
var SPRING_END = '05-31'; // varies
var MAX_CLOUD_PERCENT = 80;

var bloomSites = ee.FeatureCollection(table.map(function(feature) {
  return ee.Feature(feature.geometry(), {
    'id': feature.get('id'),
    'Site': feature.get('Site'),
    'Type': feature.get('Type'),
    'Season': feature.get('Season'),
    'Area': feature.get('Area')
  });
}));

print('Loaded sites:' + bloomSites.size());


function maskLandsatClouds(image) {
  var qa = image.select('QA_PIXEL');
  var cloudMask = qa.bitwiseAnd(1 << 3).eq(0)  // Cloud shadow
                  .and(qa.bitwiseAnd(1 << 4).eq(0))  // Snow
                  .and(qa.bitwiseAnd(1 << 5).eq(0)); // Cloud
  return image.updateMask(cloudMask);
}

// process each year to make the datapipeline auto
function processYear(year) {
  var startDate = ee.Date.fromYMD(year, 3, 1);  // March 1
  var endDate = ee.Date.fromYMD(year, 5, 31);   // May 31
  
  // landsat 8
  var landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(bloomSites)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUD_COVER', MAX_CLOUD_PERCENT))
    .map(maskLandsatClouds);

  // landsat 9
  var landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterBounds(bloomSites)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUD_COVER', MAX_CLOUD_PERCENT))
    .map(maskLandsatClouds);

  var landsat = landsat8.merge(landsat9);

  // NDVI and NDWI bands
  var withIndices = landsat.map(function(image) {
    var ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
    var ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
    
    return image.addBands([ndvi, ndwi])
      .set('system_date', image.date().format('YYYY-MM-dd'))
      .set('year', year);
  });

  return withIndices;
}

// time series
var allYearsCollection = ee.ImageCollection(ee.FeatureCollection(YEARS.map(processYear)).flatten());

print('Total images across all years:' + allYearsCollection.size());

// extract time series
// This approach is much more efficient than your previous reduceRegions method
var timeSeries = allYearsCollection.map(function(image) {
  // Reduce the image by each site - this happens in parallel
  return image.select(['NDVI', 'NDWI']).reduceRegions({
    collection: bloomSites,
    reducer: ee.Reducer.mean(),
    scale: 30,
    tileScale: 2  // Reduced to prevent computation errors
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

print('Time series feature count:' + timeSeries.size());
print('Sample time series feature:' + timeSeries.first());

// calculating summary - didnt use due to limited computational power
var siteSummary = bloomSites.map(function(site) {
  var siteId = site.get('id');
  var siteName = site.get('Site');
  
  var siteData = timeSeries.filter(ee.Filter.eq('id', siteId));
  
  // Calculate statistics
  var ndviStats = siteData.aggregate_stats('NDVI');
  var ndwiStats = siteData.aggregate_stats('NDWI');
  
  // Find peak NDVI date
  var maxNdvi = siteData.aggregate_max('NDVI');
  var peakData = siteData.filter(ee.Filter.eq('NDVI', maxNdvi)).first();
  
  return ee.Feature(site.geometry(), {
    'Site': siteName,
    'Type': site.get('Type'),
    'Area': site.get('Area'),
    'years_analyzed': YEARS.join(','),
    'total_observations': siteData.size(),
    'mean_NDVI': ndviStats.get('mean'),
    'max_NDVI': maxNdvi,
    'std_NDVI': ndviStats.get('stdDev'),
    'mean_NDWI': ndwiStats.get('mean'),
    'max_NDWI': siteData.aggregate_max('NDWI'),
    'std_NDWI': ndwiStats.get('stdDev'),
    'peak_date': ee.Feature(peakData).get('date'),
    'peak_year': ee.Feature(peakData).get('year')
  });
});

// exporting all
// Export 1: Full time series data
Export.table.toDrive({
  collection: timeSeries,
  description: 'bloomwatch_full_timeseries_2020_2025',
  folder: 'EarthEngineExports',
  fileFormat: 'CSV',
  selectors: ['id', 'Site', 'Type', 'date', 'year', 'NDVI', 'NDWI', 'cloud_cover']
});

// Export 2: Site summaries
Export.table.toDrive({
  collection: siteSummary,
  description: 'bloomwatch_site_summaries_2020_2025',
  folder: 'EarthEngineExports', 
  fileFormat: 'CSV'
});

// visualisation but didnt work
// Add a sample year to the map for verification
var sampleYear = processYear(2023).median();
Map.addLayer(sampleYear.select('NDVI'), {min: -0.2, max: 0.8, palette: ['white', 'yellow', 'green']}, 'Sample NDVI 2023');
Map.addLayer(bloomSites, {color: 'red'}, 'Study Sites');

print('===== ALHAMDULILLAH IT WORKS =====');