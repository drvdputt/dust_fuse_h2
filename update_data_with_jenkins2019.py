from astropy.table import Table
import numpy as np

old_data = Table.read(
    "data/fuse_h1_h2.dat", format="ascii.commented_header", header_start=-1
)

jenkins_data = Table.read("data/jenkins2019_hi_h2.fits")

# modify the name column of jenkins data
jenkins_names = []
for s in jenkins_data["Name"].data:
    s = s.decode("utf-8").strip(" ")
    if "BD" in s:
        s = s.replace(" ", "d")
    if "HD" in s and "HDE" not in s:
        s = "HD{:0>6}".format(int(s[2:]))
    jenkins_names.append(s)

# go over the data, and change the values as needed
for ji, s in enumerate(jenkins_names):
    if s in old_data["name"].data:

        jrow = jenkins_data[ji]
        new_hi = jrow["logNHI"]

        # their uncertainties are an upper and lower bound
        new_hi_unc_lower = jrow["e_logNHI"]
        new_hi_unc_upper = jrow["E_logNHI"]
        new_hi_unc = max(new_hi_unc_lower, new_hi_unc_upper)

        # They say that their estimted uncertainties represent 1.5sigma
        # somehow. So divide by 1.5 to get sigma. (Should probably ask
        # Karl for advice).
        new_hi_unc /= 1.5

        old_index = np.where(old_data["name"].data == s)[0][0]
        old_data[old_index]["lognhi"] = new_hi
        old_data[old_index]["lognhi_unc"] = new_hi_unc
        old_data[old_index]["hiref"] = 15
        # update the data and change the reference number to 15 (meaning
        # Jenkins et al. 2019, should also put this in paper)_

old_data.write(
    "data/fuse_h1_h2_update.dat",
    format="ascii.commented_header",
    header_start=-1,
    overwrite=True,
    formats={"lognhi_unc": "{:.2f}"},
)
# the formats option is used to round to 0.01 for the uncertainy
# (dividing by 1.5 sigma results in 0.066666666 etc.)
