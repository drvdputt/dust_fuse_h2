"""Machine readable tables"""

from get_data import get_merged_table, get_fuse_h2_details
from astropy.table import Table

data = get_merged_table()

main_stars = data["Name"]
data_comp = get_merged_table(comp=True)
comp_stars = data_comp["Name"]
other_stars = [
    "HD003827, HD022586, HD177989, HD210121, HD210839, HD220057, HD239683, HD303308, HD315021"
]


def convert_gastable_to_mrt(data, fn):
    q_with_unc = ["lognhtot", "lognhi", "lognh2", "d"]
    q_descriptions = [
        f"Log10 of column density N({name})" for name in ["H", "HI", "H2"]
    ] + ["Distance"]

    gasmrt = Table()
    gasmrt["Name"] = data["Name"]
    gasmrt["Name"].description = "Star name"
    for q, description in zip(q_with_unc, q_descriptions):
        q_unc = q + "_unc"
        new_colname = q
        new_colname_unc = "e_" + new_colname

        gasmrt[new_colname] = data[q]
        gasmrt[new_colname].description = description
        gasmrt[new_colname_unc] = data[q_unc]
        gasmrt[new_colname_unc].description = "sigma " + description

    q_other = ["fh2", "denhtot", "hiref"]
    q_other_descriptions = ["Molecular fraction", "Number density", "HI data reference"]
    for q, description in zip(q_other, q_other_descriptions):
        new_colname = q

        gasmrt[new_colname] = data[q]
        gasmrt[new_colname].description = description

    gasmrt.write(fn, format="mrt", overwrite=True)


convert_gastable_to_mrt(data, "paper-tables/mrt_gas.dat")
convert_gastable_to_mrt(data_comp, "paper-tables/mrt_gas_comp.dat")

h2data = get_fuse_h2_details(components=True, stars=main_stars)
h2data_comp = get_fuse_h2_details(components=True, stars=comp_stars)
h2data_other = get_fuse_h2_details(components=True, stars=other_stars)


def convert_h2table_to_mrt(h2data, fn):
    h2mrt = Table()
    h2mrt["Name"] = h2data["Name"]
    h2mrt["Name"].description = "Star name"
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
        h2mrt.write(fn, format="mrt", overwrite=True)


convert_h2table_to_mrt(h2data, "paper-tables/mrt_h2.dat")
convert_h2table_to_mrt(h2data_comp, "paper-tables/mrt_h2_comp.dat")
convert_h2table_to_mrt(h2data_other, "paper-tables/mrt_h2_other.dat")
