"""Investigate different spectra by the same instrument for a certain target."""
import get_spectrum
import argparse
from pathlib import Path
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("data_type", choices=["STIS", "IUEH", "IUEL"])
    args = parser.parse_args()
    target = args.target
    data_type = args.data_type

    data_dir = Path("./data") / target
    if data_type == "STIS":
        filenames = data_dir.glob("**/*_x1d.fits")
    elif data_type == "IUEH":
        filenames = data_dir.glob("*mxhi.gz")
    elif data_type == "IUEL":
        filenames = data_dir.glob("*mxlo_vo.fits")
    # convert all to string to make sure the rest works
    filenames = [str(f) for f in filenames]

    fig, ax = plt.subplots()
    for fn in filenames:
        plot_spectrum(ax, fn)

    plt.show()


def plot_spectrum(ax, filename):
    wavs, flux, errs, _ = get_spectrum.auto_wavs_flux_errs(str(filename))
    ax.plot(wavs, flux, alpha=0.5)


if __name__ == "__main__":
    main()
