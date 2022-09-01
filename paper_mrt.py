"""Machine readable tables"""

from get_data import get_merged_table, get_fuse_h2_details
from astropy.table import Table

data = get_merged_table()
keep_columns = ["Name"]
q_with_unc = ["lognhtot", "lognhi", "lognh2", "d"]
for q in q_with_unc:
    keep_columns += [q, q + "_unc"]
q_other = ["fh2", "denhtot", "hiref"]
for q in q_other:
    keep_columns += [q]
data[keep_columns].write("paper-tables/mrt_test.dat", format="mrt", overwrite=True)

h2data = get_fuse_h2_details(components=True, stars=data["Name"])
h2mrt = Table()
h2mrt["Name"] = h2data["Name"]
for j in range(8):
    # the densities from J = 0 to J = 7
    old_colname = f"lognj{j}"
    old_uncname = old_colname + "_unc"
    new_colname = f"N{j}"
    new_uncname = "e_" + new_colname

    h2mrt[new_colname] = h2data[old_colname]
    h2mrt[new_colname].description = f"Log10 of column density H2(J = {j})"
    h2mrt[new_uncname] = h2data[old_uncname]
    h2mrt[new_uncname].description = f"sigma(Log10) column density H2(J = {j})"


h2data[keep_columns_h2].write("paper-tables/mrt_h2.dat", format="mrt", overwrite=True)
