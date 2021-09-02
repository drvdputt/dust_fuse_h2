"""Find stars that are both in our sample and in Shull+21"""

import get_data

data = get_data.get_merged_table()
shull = get_data.get_shull2021()
matches = [name for name in data["Name"] if name in shull["Name"]]
print(len(matches), " matches found")
print(matches)
