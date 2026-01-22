
from astroquery.vizier import Vizier
import os

# J/ApJ/874/32/table1  Environment and hosts of Type Ia supernovae (Rose+, 2019)
catalog_id = "J/ApJ/874/32"

print(f"Downloading {catalog_id} from VizieR...")
v = Vizier(columns=['**'])
v.ROW_LIMIT = -1
catalogs = v.get_catalogs(catalog_id)

output_dir = os.path.join("data", "external", "Rose19")
os.makedirs(output_dir, exist_ok=True)

if len(catalogs) > 0:
    for table_name in catalogs.keys():
        table = catalogs[table_name]
        filename = os.path.join(output_dir, f"{table_name.replace('/','_')}.csv")
        # Convert to pandas to handle masked arrays/formatting issues
        df = table.to_pandas()
        df.to_csv(filename, index=False)
        print(f"Saved {table_name} to {filename}")
else:
    print("No catalogs found.")
