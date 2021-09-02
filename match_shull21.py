"""Find stars that are both in our sample and in Shull+21"""

import numpy as np
import get_data
from matplotlib import pyplot as plt

data = get_data.get_merged_table()
shull = get_data.get_shull2021()
matches = [name for name in data["Name"] if name in shull["Name"]]
print(len(matches), " matches found")
print(matches)

data_comp = data[np.isin(data["Name"], matches)]
refs = data_comp['hiref']
shull_comp = shull[np.isin(shull["Name"], matches)]


def compare_shull(param):
    plt.figure()
    x = shull_comp[param]
    y = data_comp[param]
    plt.plot(x, x, color="k")
    plt.scatter(x, y, c=refs)
    plt.colorbar()
    plt.ylabel("ours")
    plt.xlabel("shull")
    plt.title(param)


# compare_shull("nhtot")
compare_shull("EBV")
compare_shull("fh2")
compare_shull("nhi")
compare_shull("nh2")

plt.show()
