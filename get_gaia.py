"""Looks up parallax distances in Gaia, and writes these to a file.

These data can later be merged with the rest in get_data.get_merged_table."""

import get_data
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import units as u
from pathlib import Path
from astropy.table import Table, vstack

# get names of stars in format that simbad understands
data = get_data.get_merged_table()
names = [n.replace("d", " ") for n in data["Name"]]

# get info from simbad
Simbad.add_votable_fields('parallax')
s_path = Path("data/stars_simbad.dat")
if s_path.exists():
    s = Table.read(s_path, format="ascii.commented_header")
else:
    s = Simbad.query_objects(names)
    s.write(s_path, format="ascii.commented_header")

# make coordinate object using simbad RA and DEC
c = SkyCoord(ra=s["RA"], dec=s["DEC"], unit=(u.hourangle, u.deg))

# query gaia
Gaia.ROW_LIMIT = 8
w = 0.1 * u.arcmin
h = 0.1 * u.arcmin
g_tables = []
for name, ci in zip(data["Name"], c):
    g_path = Path(f"data/gaia/{name}.dat")
    if g_path.exists():
        g = Table.read(g_path, format="ascii.commented_header")
    else:
        g = Gaia.query_object(ci, width=w, height=h)
        g_path.parent.mkdir(parents=True, exist_ok=True)
        g.write(g_path, format="ascii.commented_header")
    print(g)
    g_tables.append(g[:1])

g_all = vstack(g_tables)
g_all.add_column(data["Name"], index=0)
g_all.write("data/gaia/merged.dat", format="ascii.commented_header", overwrite=True)
