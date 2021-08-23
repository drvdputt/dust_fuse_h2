"""this script downloads the IUE SWP data for our lyman alpha fitting,
if available. It uses the CSV table from a MAST search as input, and
uses the URLs in this table to download it."""

import pandas as pd
from astropy.utils.data import download_file
import shutil
from pathlib import Path

targets = [
    "BD+56d524",
    "HD023060",
    "HD046202",
    "HD047129",
    "HD062542",
    "HD093827",
    "HD096675",
    "HD099872",
    "HD152248",
    "HD179406",
    "HD190603",
    "HD197770",
    "HD209339",
    "HD216898",
    "BD+52d3210",
    "HD037332",
    "HD037525",
    "HD093028",
    "HD094493",
    "HD097471",
    "HD235874",
    "HD051013",
    "HD326329",
    # for comparison
    "HD045314",
]

# load the search results table
def main():
    search_results = pd.read_csv("MAST_IUE_SWP.csv", header=4)

    for t in targets:
        if "HD" in t:
            number = int(t[2:])
            rex = "HD[ 0]+{}".format(number)

        elif "BD" in t:
            number1 = int(t[3:5])
            number2 = int(t[6:])
            rex = "BD[ \+]*{}[ d]*{}".format(number1, number2)
            # print(rex)

        where_match = search_results["target_name"].str.match(rex, na=False)
        if where_match.sum() > 0:
            matches = search_results[where_match]
            print(matches)
            download_matches(t, matches)


def download_matches(target, matches):
    target_dir = Path.cwd() / "data" / target
    target_dir.mkdir(exist_ok=True)

    print("downloading files for " + target)
    for n, u in zip(matches["target_name"], matches["dataURL"]):
        dl_file = Path(download_file(u, cache=True))

        target_file = target_dir / Path(u).name
        shutil.copy(dl_file, target_file)

if __name__ == "__main__":
    main()
