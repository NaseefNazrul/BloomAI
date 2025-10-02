import ee
import time
from datetime import datetime, timedelta

class EarthEngineDataTester:
    """
    Complete tester for Earth Engine data retrieval
    Based on the working code from our earlier debug session
    """
    
    def __init__(self):
        self.initialize_earth_engine()
    
    def initialize_earth_engine(self):
        """Initialize Earth Engine"""
        try:
            ee.Initialize(project='stable-ring-473811-a3')
            print("✅ Earth Engine initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Earth Engine initialization failed: {e}")
            return False
    
    def get_complete_satellite_data(self, lat, lon, date, satellite='Landsat-9', buffer_meters=200):
        """
        Get complete satellite data including cloud cover and image metadata
        Based on the working debug_landsat_data function
        """
        print(f"\n{'='*60}")
        print(f"🛰️  TESTING {satellite} at ({lat}, {lon}) on {date}")
        print(f"{'='*60}")
        
        start_time = time.time()
        point = ee.Geometry.Point([lon, lat])
        area = point.buffer(buffer_meters)
        
        # Choose collection (from working code)
        if satellite == 'Landsat-9':
            collection_id = 'LANDSAT/LC09/C02/T1_L2'
        else:  # Landsat-8
            collection_id = 'LANDSAT/LC08/C02/T1_L2'
        
        try:
            # Filter collection (optimized from working code)
            filtered = (ee.ImageCollection(collection_id)
                       .filterBounds(area)
                       .filterDate(date, f'{date}T23:59:59')
                       .sort('CLOUD_COVER')
                       .limit(1))
            
            collection_size = filtered.size().getInfo()
            print(f"📊 Images found: {collection_size}")
            
            if collection_size == 0:
                print("❌ No images available for this date")
                return None
            
            # Get the image
            image = filtered.first()
            
            # Get complete image info (from working debug code)
            image_info = image.getInfo()['properties']
            image_id = image_info.get('LANDSAT_PRODUCT_ID', 'N/A')
            cloud_cover = image_info.get('CLOUD_COVER', 'N/A')
            date_acquired = image_info.get('DATE_ACQUIRED', 'N/A')
            
            print(f"🔍 Image Details:")
            print(f"   Product ID: {image_id}")
            print(f"   Date: {date_acquired}")
            print(f"   Cloud Cover: {cloud_cover}%")
            
            # Calculate vegetation indices (from working code)
            ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
            ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
            composite = ndvi.addBands(ndwi)
            
            # Get area statistics
            area_stats = composite.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=area,
                scale=30,
                maxPixels=1e6,
                bestEffort=True
            ).getInfo()
            
            ndvi_val = area_stats.get('NDVI')
            ndwi_val = area_stats.get('NDWI')
            
            # Get QA_PIXEL data for cloud analysis (from debug code)
            qa_stats = image.select('QA_PIXEL').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=area,
                scale=100,
                maxPixels=1e6
            ).getInfo()
            qa_pixel = qa_stats.get('QA_PIXEL')
            
            processing_time = time.time() - start_time
            
            # Complete result set
            result = {
                'success': True,
                'satellite': satellite,
                'date': date_acquired,
                'vegetation_data': {
                    'ndvi': ndvi_val,
                    'ndwi': ndwi_val
                },
                'cloud_data': {
                    'cloud_cover_percentage': cloud_cover,
                    'qa_pixel_value': qa_pixel,
                    'quality': 'clear' if cloud_cover < 10 else 'partly_cloudy' if cloud_cover < 50 else 'cloudy'
                },
                'image_metadata': {
                    'product_id': image_id,
                    'collection': collection_id,
                    'buffer_radius_m': buffer_meters
                },
                'coordinates': {
                    'latitude': lat,
                    'longitude': lon
                },
                'processing_time': round(processing_time, 2)
            }
            
            print(f"✅ DATA RETRIEVAL SUCCESS:")
            print(f"   NDVI: {ndvi_val:.4f}")
            print(f"   NDWI: {ndwi_val:.4f}")
            print(f"   Cloud Cover: {cloud_cover}%")
            print(f"   QA Pixel: {qa_pixel}")
            print(f"   Processing Time: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def test_satellite_fallback(self, lat, lon, date):
        """
        Test Landsat-9 → Landsat-8 fallback with complete data
        """
        print(f"\n🛰️  TESTING SATELLITE FALLBACK for {date}")
        
        # Try Landsat-9 first
        result = self.get_complete_satellite_data(lat, lon, date, 'Landsat-9', 200)
        
        if result and result['success'] and result['vegetation_data']['ndvi'] is not None:
            print("🎯 Landsat-9 data acquisition successful")
            return result
        
        # Fall back to Landsat-8
        print("🔄 Falling back to Landsat-8...")
        result = self.get_complete_satellite_data(lat, lon, date, 'Landsat-8', 200)
        
        if result and result['success'] and result['vegetation_data']['ndvi'] is not None:
            print("🎯 Landsat-8 data acquisition successful")
            return result
        
        print("💥 Both satellites failed to retrieve data")
        return None
    
    def test_date_fallback_complete(self, lat, lon, target_date, max_days_back=5):
        """
        Test date fallback with complete data retrieval
        """
        print(f"\n📅 TESTING DATE FALLBACK from {target_date}")
        
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        
        for days_back in range(0, max_days_back + 1):
            current_date = (target_dt - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            if days_back > 0:
                print(f"🔄 Trying {days_back} day(s) back: {current_date}")
            
            result = self.get_complete_satellite_data(lat, lon, current_date, 'Landsat-9', 200)
            
            if result and result['success'] and result['vegetation_data']['ndvi'] is not None:
                if days_back > 0:
                    result['fallback_info'] = {
                        'original_date': target_date,
                        'days_offset': days_back,
                        'fallback_used': True
                    }
                    print(f"✅ Fallback successful: Found data from {current_date}")
                return result
        
        print("💥 Date fallback failed: No data found in search period")
        return None
    
    def run_comprehensive_data_test(self, test_cases):
        """
        Run comprehensive test with complete data retrieval
        """
        print("🚀 COMPREHENSIVE EARTH ENGINE DATA TEST")
        print("=" * 60)
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 TEST CASE {i}/{len(test_cases)}")
            print(f"Location: ({test_case['lat']:.4f}, {test_case['lon']:.4f})")
            print(f"Target Date: {test_case['date']}")
            
            # Test with satellite fallback
            result = self.test_satellite_fallback(
                test_case['lat'], 
                test_case['lon'], 
                test_case['date']
            )
            
            if result:
                results.append(result)
            else:
                # If primary fails, test date fallback
                print("🔄 Primary date failed, testing date fallback...")
                fallback_result = self.test_date_fallback_complete(
                    test_case['lat'], 
                    test_case['lon'], 
                    test_case['date']
                )
                if fallback_result:
                    results.append(fallback_result)
                else:
                    print("💥 Test case completely failed")
        
        # Print comprehensive summary
        self.print_detailed_summary(results)
        return results
    
    def print_detailed_summary(self, results):
        """Print detailed test summary with all data"""
        print(f"\n{'='*60}")
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*60}")
        
        successful_tests = [r for r in results if r.get('success') and r['vegetation_data']['ndvi'] is not None]
        
        print(f"Total tests: {len(results)}")
        print(f"Successful retrievals: {len(successful_tests)}")
        print(f"Success rate: {len(successful_tests)/len(results)*100:.1f}%" if results else "0%")
        
        if successful_tests:
            print(f"\n📈 SUCCESSFUL DATA RETRIEVALS:")
            for result in successful_tests:
                veg = result['vegetation_data']
                cloud = result['cloud_data']
                meta = result['image_metadata']
                
                print(f"  • {result['satellite']} | {result['date']}")
                print(f"    NDVI: {veg['ndvi']:.4f} | NDWI: {veg['ndwi']:.4f}")
                print(f"    Clouds: {cloud['cloud_cover_percentage']}% | Quality: {cloud['quality']}")
                print(f"    Product: {meta['product_id']}")
                print(f"    Time: {result['processing_time']}s")
                if result.get('fallback_info'):
                    print(f"    ⚠️  Used fallback: {result['fallback_info']['days_offset']} day(s) offset")
                print()

# Test configurations
TEST_CASES = [
    {'lat': 33.2767, 'lon': -110.3062, 'date': '2023-04-28'},  # Your working location
    {'lat': 33.2767, 'lon': -110.3062, 'date': '2023-04-20'},  # Alternative date
    {'lat': 34.0522, 'lon': -118.2437, 'date': '2023-04-15'},  # Los Angeles
    {'lat': 40.7128, 'lon': -74.0060, 'date': '2023-04-10'},   # New York
]

# Run the complete test
if __name__ == "__main__":
    print("🌍 COMPLETE EARTH ENGINE DATA RETRIEVAL TEST")
    print("Testing: NDVI, NDWI, Cloud Cover, Image Metadata")
    
    tester = EarthEngineDataTester()
    
    # Run comprehensive test
    all_results = tester.run_comprehensive_data_test(TEST_CASES)
    
    # Performance and data quality analysis
    print(f"\n⚡ DATA QUALITY ANALYSIS:")
    if all_results:
        successful = [r for r in all_results if r.get('success') and r['vegetation_data']['ndvi'] is not None]
        
        if successful:
            avg_ndvi = sum(r['vegetation_data']['ndvi'] for r in successful) / len(successful)
            avg_cloud = sum(r['cloud_data']['cloud_cover_percentage'] for r in successful) / len(successful)
            avg_time = sum(r['processing_time'] for r in successful) / len(successful)
            
            print(f"Average NDVI: {avg_ndvi:.4f}")
            print(f"Average Cloud Cover: {avg_cloud:.1f}%")
            print(f"Average Processing Time: {avg_time:.2f}s")
            print(f"Data Reliability: {len(successful)}/{len(all_results)} successful retrievals")
            
            # Satellite usage
            landsat9 = [r for r in successful if r['satellite'] == 'Landsat-9']
            landsat8 = [r for r in successful if r['satellite'] == 'Landsat-8']
            print(f"Landsat-9 usage: {len(landsat9)} | Landsat-8 usage: {len(landsat8)}")