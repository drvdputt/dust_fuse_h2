from pathlib import Path

# can be manually tweaked
target_use_which_spectrum = {
    "HD097471": "data/HD097471/swp19375mxlo_vo.fits",
    "HD094493": "data/HD094493/mastDownload/HST/o54306010/o54306010_x1d.fits",
    "HD037525": "data/HD037525/swp27579.mxhi.gz",
    "HD093827": "data/HD093827/swp50536.mxhi.gz",
    "HD051013": "data/HD051013/swp22860.mxhi.gz",
    "HD096675": "data/HD096675/swp41717.mxhi.gz",
    "HD023060": "data/HD023060/swp11151mxlo_vo.fits",
    "HD099872": "data/HD099872/mastDownload/HST/o6lj0i020/o6lj0i020_x1d.fits",
    "HD152248": "data/HD152248/swp54576.mxhi.gz",
    "HD209339": "data/HD209339/mastDownload/HST/o5lh0b010/o5lh0b010_x1d.fits",
    "HD197770": "data/HD197770/mastDownload/HST/oedl04010/oedl04010_x1d.fits",
    "HD037332": "data/HD037332/swp32289.mxhi.gz",
    "HD093028": "data/HD093028/swp05521.mxhi.gz",
    "HD062542": "data/HD062542/mastDownload/HST/obik01010/obik01010_x1d.fits",
    "HD190603": "data/HD190603/swp01822.mxhi.gz",
    "HD046202": "data/HD046202/swp08845.mxhi.gz",
    "HD047129": "data/HD047129/swp07077.mxhi.gz",
    "HD235874": "data/HD235874/swp34158mxlo_vo.fits",
    "HD216898": "data/HD216898/swp43934.mxhi.gz",
    "HD326329": "data/HD326329/swp48698.mxhi.gz",
    "HD179406": "data/HD179406/swp36939.mxhi.gz",
    "BD+52d3210": "data/BD+52d3210/swp34153mxlo_vo.fits",
    "BD+56d524": "data/BD+56d524/swp20330mxlo_vo.fits",
}

def get_spectrum(target):
    """
    Get spectrum for the given target. Tweak the variable
    get_spectrum.target_use_which_spectrum to choose the right data.
    Depending on whether a IUE or STIS spectrum was chosen, different
    steps will be taken. The end result is the spectral data in a common
    format

    Returns
    -------
    wav, flux: ndarray of wavelengths (angstrom) and fluxes (erg s-1 cm-2 angstrom-1)

    """
    pass


# Some code to generate the above dict from scratch. Manual tweaking can
# occur after.
if __name__ == "__main__":
    gen_dict = {}
    here = Path(".")
    for d in list(here.glob("./data/HD*")) + list(here.glob("./data/BD*")):
        has_iue_h = False
        has_iue_l = False
        has_hst_stis = False
        # has_hst_cos = False

        # lower in this list of ifs is higher priority
        target = Path(d).name

        # def set_if_exists(glob_pattern):
        #     files = d.glob(glob_pattern)
        #     if len(files) > 0:
        #         spectrum_file = files[0]

        iue_l_files = list(d.glob("*mxlo_vo.fits"))
        if len(iue_l_files) > 0:
            spectrum_file = str(iue_l_files[0])

        iue_h_files = list(d.glob("*mxhi.gz"))
        if len(iue_h_files) > 0:
            spectrum_file = str(iue_h_files[0])

        hst_stis_files = list(d.glob("**/*x1d.fits"))
        if len(hst_stis_files) > 0:
            spectrum_file = str(hst_stis_files[0])

        gen_dict[target] = spectrum_file

    print(gen_dict)
