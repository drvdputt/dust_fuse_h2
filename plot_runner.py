"""
Automated version of interesting plots notebook. Will later be used to
make consistent plots for paper.
"""

from plot_fuse_results import plot_results2
from get_data import get_merged_table, get_bohlin78
from astropy.table import Column
from pathlib import Path
from multiprocessing.pool import Pool
import numpy as np


def main():
    out_dir = Path("./scatter-plots")
    out_dir.mkdir(exist_ok=True)

    # This work + gordon 09
    data = get_merged_table()
    # gordon 09 comparison stars
    data_comp = get_merged_table(comp=True)
    # bohlin 78 copernicus sightlines
    data_bohlin78 = get_bohlin78()

    # Write out data as used, for later reference. Might be useful for
    # thinking about specific points using e.g. highlighting subsets
    # using TopCat.
    data.write("data_main.fits")
    data_comp.write("data_comp.fits")
    data_bohlin78.write("data_bohlin.fits")

    # add comments for certain stars here
    data.add_column(Column(["no"] * len(data), dtype="<U16", name="comment"))

    def set_comment(name, s):
        data["comment"][data["Name"] == name] = s

    # stars far below the general NH/AV trend
    for name in ["HD200775", "HD164906", "HD045314", "HD206773"]:
        set_comment(name, "lo_h_av")
    # stars far above the general NH/AV trend
    set_comment("HD096675", "hi_h_av")

    # stars to avoid in the fitting, indicated by comment
    ignore = ["lo_h_av", "hi_h_av"]

    # correlations to plot
    columns_to_correlate = (
        ["AV", "nhtot"],
        ["EBV", "nhtot"],
        ["AV", "fh2"],
        ["EBV", "fh2"],
        ["RV", "fh2"],
        ["nhtot", "NH_AV"],
        ["AV", "NH_AV",],
        ["EBV", "NH_EBV"],
        ["nhtot", "NH_EBV"],
        ["RV", "NH_AV"],
        ["fh2", "NH_AV"],
        ["fh2", "NH_EBV"],
        ["nhtot", "fh2"],
    )

    job_args = []
    for (xparam, yparam) in columns_to_correlate:
        if xparam in data_bohlin78.colnames and yparam in data_bohlin78.colnames:
            use_bohlin = data_bohlin78
        else:
            use_bohlin = None
        job_args.append((data, xparam, yparam, data_comp, use_bohlin, ignore, out_dir))

    with Pool(16) as p:
        p.map(wrapper, job_args)


def wrapper(args):
    data, xparam, yparam, data_comp, use_bohlin, ignore, out_dir = args

    xdata_main = data[xparam]
    ydata_main = data[yparam]
    xdata_all = np.concatenate([data[xparam], data_comp[xparam]])
    ydata_all = np.concatenate([data[yparam], data_comp[yparam]])
    sx = np.std(xdata_all)
    sy = np.std(ydata_all)

    def suitable_range(param):
        data_main = data[param]
        all_data = np.concatenate([data[param], data_comp[param]])
        s = np.std(all_data)
        # use smallest range using suggested boundaries
        pmin = max(min(data_main) - s, min(all_data) - 0.1 * s)
        pmax = min(max(data_main) + s, max(all_data) + 0.1 * s)
        return [pmin, pmax]

    fig = plot_results2(
        data,
        xparam,
        yparam,
        suitable_range(xparam),
        suitable_range(yparam),
        data_comp=data_comp,
        data_bohlin=use_bohlin,
        ignore_comments=ignore,
    )
    fig.tight_layout()
    fn = f"{yparam}_vs_{xparam}.pdf"
    fig.savefig(out_dir / fn, bbox_inches="tight")


if __name__ == "__main__":
    main()
