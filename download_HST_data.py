from astroquery import mast
from pathlib import Path


def main():
    # list of stars for which to download data
    with open("download_data_list.csv", "r") as f:
        star_names = [s.strip(" \n") for s in f.readlines()]

    data_found = {}
    for target in star_names:
        # results = mast.Observations.query_object(
        #     name, radius="0.1 arcmin", dataproduct_type=["spectrum"]
        # )
        results = mast.Observations.query_criteria(
            objectname=target, radius="0.1 arcmin", dataproduct_type=["spectrum"],
        )
        count = process_mast_results(target, results)
        data_found[target] = count

    print("Data search/download overview (number of spectra found)")
    for key, value in data_found.items():
        print(key, " ", value)


def process_mast_results(target, results):
    print(target)
    count = {}

    # save results to disk for later debugging
    target_dir = Path.cwd() / "data" / target
    target_dir.mkdir(exist_ok=True)
    results.write(
        target_dir / "mast_search_results.txt",
        format="ascii.fixed_width",
        include_names=[
            "obs_collection",
            "obs_id",
            "target_name",
            "t_exptime",
            "filters",
            "proposal_id",
            "obsid",
        ],
    )

    for row in results:
        o = row["obs_collection"]
        i = row["instrument_name"]
        f = row["filters"]
        print("\t", o, i, f)

        # HST STIS
        if "STIS" in i and (f == "E140H" or f == "E140M"):
            products = mast.Observations.get_product_list(row)
            manifest = mast.Observations.download_products(
                products, productType="SCIENCE", download_dir=str(target_dir)
            )
            count["STIS"] = count.get("STIS", 0) + 1

        # IUE
        if "SWP" in i:
            products = mast.Observations.get_product_list(row)
            manifest = mast.Observations.download_products(
                products, productType="SCIENCE", download_dir=str(target_dir)
            )

            if "HIGH" in f:
                count["IUE H"] = count.get("IUE H", 0) + 1
            if "LOW" in f:
                count["IUE L"] = count.get("IUE L", 0) + 1

    return count


main()
