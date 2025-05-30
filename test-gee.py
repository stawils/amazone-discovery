# Test GEE connection

from src.gee_provider import GoogleEarthEngineProvider
try:
    gee = GoogleEarthEngineProvider()
    print('✅ Google Earth Engine ready!')
except Exception as e:
    print(f'❌ GEE Error: {e}')
