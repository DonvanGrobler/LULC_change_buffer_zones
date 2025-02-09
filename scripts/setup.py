# Earth Engine & Mapping
import ee
import geemap

# Importing the utils module
import utils

# Automatically authenticate and initialize Earth Engine when creating the map
try:
    ee.Initialize()
except ee.EEException:
    print("🌍 Earth Engine authentication required. Please follow the prompt.")
    ee.Authenticate()
    ee.Initialize()

print("✅ Packages successfully loaded and setup complete.")