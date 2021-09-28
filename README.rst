Analysis for FUSE H2/Extinction paper
=====================================

This paper is Van De Putte, Clayton, Gordon, et al. (in prep).

The H2 absorption line modeling to the FUSE data is done.
The results are used to determine NH2

More recently, Lyman alpha absorption fitting was done to determine NHI.
For this, IUE and HST STIS data were used.

The rest of the analysis compares the extinction results from Gordon+2009 with the newly obtained H2 and HI results

Below follows a quick overview of the available scripts and modules.
Some might no longer be used, but have been kept for reference.

HI fitting scripts
------------------

To perform continuum reconstruction fitting and determine NHI, spectral data were downloaded in an automated way.

``download_data_list.csv``: list of stars for which to search and download data

``download_HST_data.py``: searches MAST for HST STIS data and downloads it if available.
The data are placed in ``data/<target>/mastDownload/HST/``
Also downloads IUE SWP data (high and low dispersion), into ``data/<target>/mastDownload/IUE``.

This script did not find data for HD216898, so I downloaded those from the IUE archive manually. - Dries

``download_IUE_data.py``: obsolete script, used before the above was generalized.

``get_spectrum.py``: functions to load and process the downloaded spectral data, given the name of a star.
The data to use for each star can be chosen by manually filling out a dictionary at the top of the file.
The function ``processed()`` automatically chooses the right functions to load the data based on the filename, and performs coadding and rebinning as necessary.

``lyafitting.py``: performs the HI fitting.
Wavelength ranges used to fit the continuum and the Lyman alpha line are individually chosen per star, using two manually filled-in dictionaries at the top of the file.
    ``--target``: specify a single star and interactively look at the plot.
    ``--update_catalog``: replace the HI values in the given file (one of the data files in ``data/*hi_h2*``), so that they can be used for the rest of the analysis (next section)

Sample analysis
---------------

Most of the analysis consists of scatter plots, sometimes with a line fit through them.

The uncertainties in x and y are both significant, and some of the chosen quantities are inherently correlated because they have a common factor.
For example, NH/AV and AV are anti-correlated because of the common AV factor.
We visualize the errors and covariance between x and y for each point as an ellipse.

The chi2 function that is minimized to perform the linear fits is one presented in Hogg et al. (2010).
It uses the orthogonal distance to the line as the deviation, and the orthogonally projected covariance as the standard deviation.
This way, the shape of the uncertainty ellipse of each point is taken into account when estimating the slope of the scatter plot.

``linear_ortho_fit.py`` contains functions to perform the fitting described above

``covariance.py`` contains the function that draws the scatter plot and the covariance ellipses, as well as helper functions to calculate the covariance between certain quantities

``get_data.py`` contains functions to retrieve the data and uncertainties for all the sightlines, and calculate some derived quantities

``plot_fuse_results.py`` makes use of the above modules to do the analysis and make the desired plots

Workflow
========

The following is a reminder on how to do the analysis from start to finish, using the available scripts.

Starting point
--------------

The following work and data files were already done by one or more co-authors, when I took over the project:

* H2 fitting to FUSE data. Results in ``data/fuse_h2_details.dat``.
* Gathering results from earlier work into files

  * ``bohlin78_copernicus.dat``
  * ``fuse_ext_details.dat`` and ``fuse_ext_fm90.dat`` from Gordon+2009

* Collection of HI data from the literature; see ``hiref`` column in ``data/fuse_h1_h2.dat``, and reference numbers in paper.


HI data and fitting
-------------------

The HI data needed to be improved, since some of these data were based purely on the HI-extinction relation.
The data were replaced by new values in several ways, with priority as numbered below.

1. Use values from Jenkins+2019, table 5.
   Download table 5 from this paper ``jenkins2019_hi_h2.fits`` using::

     cd data
     curl "https://cdsarc.unistra.fr/viz-bin/nph-Cat/fits?J/ApJ/872/55/table5.dat" -o jenkins2019_hi_h2.fits

   Then go back to the root directory, and run::
     python update_data_with_jenkins2019.py

   which will create ``data/fuse_h1_h2_update.dat``

2. Lyman alpha fitting.
   The stars that did not have sufficiently accurate HI data are listed in ``download_data_list.csv``
   
   a. Download IUE and HST STIS UV spectroscopy using::
        python download_HST_data.py

      The results are stored in ``data/<star name>

   b. Choose which spectra to use by editing the dict ``target_use_which_spectrum`` at the top of ``get_spectrum.py``
      This script will co-add data if multiple files are listed using an asterisk wildcard.

   c. Choose which wavelength ranges to use for the continuum fit and the line profile fit by editing
      ``target_continuum_wav_ranges`` and ``target_lya_wav_ranges`` at the top of ``lyafitting.py``.

      * To run for one star and inspect the fitting ranges and result interactively::
          python lyafitting.py --target <name>

      * To run for all stars and write the results into the main table::
          python lyafitting.py --target all --update_catalog data/fuse_h1_h2_update.dat
        The results will be saved to ``data/fuse_h1_h2_with_lyafitting.dat``

Distances
---------

To calculate the average number density along each line of sight, the distance of each star is required.

1. First I downloaded data from Gaia DR2 using ``python get_gaia.py``
This data is saved at ``data/gaia/``, one file per star, and is merged into ``data/gaia/merged.dat``.

2. Since Gaia parallaxes are known to be inaccurate for OB stars, we instead use photometric distances with the following priority

   a. From Shull+2021 (about half the sample)

   b. Using AV and spectral types from Gordon+2009, combined with absolute magnitudes for those spectral types from Bowen+2008, appendix 3B, and Wegner+2007, Table 8.
      These tables were copied into ``data/ob_mags.dat``
      The equation is simply ``d = 1 pc * 10 ** ((V - AV - MV) / 5)``.

Merged data and derived columns
-------------------------------

The data in the files mentioned above is loaded in the ``get_data`` module.
Derived columns, such as linear (instead of log) densities, uncertainties, calculated photometric distances, are added.
The main function to retrieve everything is `get_merged_table()`.
Some functions to load data from other works are also available.

A more complex part of the code is where the covariances are calculated.

Scatter plots and fits
----------------------

In ``covariance.py``, a function was implemented to draw scatter plots where every point is an ellipse representing the covariance between x and y.

In ``linear_ortho_fit.py``, a line fitting method based on Hogg+2010 was implemented, which takes uses the perpendicular distance to calculate chi2.
It properly takes into account the uncertainty ellipse (with xy covariance) of each data point).

The main drawing and fitting calls are in ``plot_fuse_results.py``.
The typical workflow for making a plot and fitting the data (with covariance) for that plot is::

  from plot_fuse_results import plot_results_scatter, plot_results_fit
  ax = axs[0, 0]
  xs, ys, covs = plot_results_scatter(
      ax,
      data,
      "AV",
      "nhtot",
      data_comp=comp,
      data_bohlin=bohlin,
      ignore_comments=["lo_h_av", "hi_h_av"],
  )
  plot_results_fit(xs, ys, covs, ax)


Paper plots
-----------

One function per plot in ``paper_scatter.py``.

In Development
==============

Scripts and data will be changing until paper is written.
Use at your own risk.

Contributors
============

Dries Van De Putte, Karl Gordon

License
=======

This code is licensed under a 3-clause BSD style license (see the
``LICENSE`` file).

