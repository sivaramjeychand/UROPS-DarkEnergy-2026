
from astroquery.vizier import Vizier

# J/ApJ/874/32
catalog_id = "J/ApJ/874/32"
v = Vizier(columns=['**'])
catalogs = v.get_catalogs(catalog_id)

print(f"Found {len(catalogs)} tables in {catalog_id}:")
for name in catalogs.keys():
    print(f"- {name}")
    print(f"  Columns: {catalogs[name].colnames}")
