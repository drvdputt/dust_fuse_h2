from astroquery import mast
from pathlib import Path


def main():
    with open("list_for_data_lookup.csv", "r") as f:
        star_names = [s.strip(" \n") for s in f.readlines()]

    data_found = {}
    for target in star_names:
        # results = mast.Observations.query_object(
        #     name, radius="0.1 arcmin", dataproduct_type=["spectrum"]
        # )
        results = mast.Observations.query_criteria(
            objectname=target,
            radius="0.1 arcmin",
            dataproduct_type=["spectrum"],
        )
        count = process_mast_results(target, results)
        data_found[target] = count

    print("Data search/download overview (number of relevant STIS spectra found)")
    for key, value in data_found.items():
        print(key, ' ', value)

def process_mast_results(target, results):
    print(target)
    count = 0
    for row in results:
        i = row["instrument_name"]
        f = row["filters"]
        if "STIS" in i and f == "E140H":
            target_dir = Path.cwd() / "data" / target
            target_dir.mkdir(exist_ok=True)

            # for c, cc in zip(results.colnames, results[1]):
            #     print(c, cc)

            # obsid = row['obs_id']
            products = mast.Observations.get_product_list(row)
            print("Found the following products for {}:".format(target))
            print(products)
            manifest = mast.Observations.download_products(
                products, productType="SCIENCE", download_dir=str(target_dir)
            )
            count += 1
    return count


main()
