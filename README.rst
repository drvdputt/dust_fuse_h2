Analysis for FUSE H2/Extinction paper
=====================================

.. image:: https://zenodo.org/badge/376953484.svg
   :target: https://zenodo.org/badge/latestdoi/376953484

This is the data and code used for the paper by Van De Putte, Clayton, Gordon, et al. (in prep).
The code requires Python 3 (tested with >3.8), and an untested list of dependencies can be found in ``requirements_minimal.txt``

In this readme, I describe the most important data files, scripts, and modules, with the goal of providing a reference for later.
Some of the smaller files have been skipped over, if they are self-explanatory, or just small experiments.
The various notebooks are also not described here, as all of them contain highly experimental data exploration code.

Starting data
=============

The initial work for this paper, done before 2009, resulted in several data files.
In late 2021, work on this paper was continued.
We describe the data files included at that point here.
For the H2 absorption line modeling to the FUSE data, the results are included in `data/fuse_h2_details.dat`.

The total H2 columns, and literature data for NHI were collected in `data/fuse_h1_h2.dat`.
Of the NHI values, 23 were replaced by the results of the Lyman alpha absorption fitting in this work.

The details about the extinction and the FM90 extinction curve fits are in
`fuse_ext_details.dat` and `fuse_ext_fm90.dat`.

The analysis compares the extinction results from Gordon+2009 with the newly obtained H2 and HI results.
During the analysis, other data were downloaded and included in this repo.
Those will be described further below, in the relevant part of the analysis.

HI fitting
----------

To perform continuum reconstruction fitting and determine NHI, spectral data were downloaded in
an automated way. Here are the relevant files and a short description.

Scripts
.......

``download_data_list.csv``: list of stars which need lya fits

``download_HST_data.py``: searches MAST for HST STIS data and downloads it if available.
The data are placed in ``data/<target>/mastDownload/HST/``
Also downloads IUE SWP data (high and low dispersion), into ``data/<target>/mastDownload/IUE``.

``lyafitting.py``: performs the HI fitting.
Wavelength ranges used to fit the continuum and the Lyman alpha line are individually chosen per star, using two manually filled-in dictionaries at the top of the file.

``--target``: specify a single star and interactively look at the plot.

``--update_catalog``: replace the HI values in the given file (one of the data files in ``data/*hi_h2*``), so that they can be used for the rest of the analysis (next section)

Modules
.......

``get_spectrum.py``: functions to load and process the downloaded spectral data, given the name of a star.
The data to use for each star can be chosen by manually filling out a dictionary at the top of the file.
The function ``processed()`` automatically chooses the right functions to load the data based on the filename, and performs coadding and rebinning as necessary.

Correlation and slope analysis
------------------------------

The uncertainties in x and y are both significant, and some of the chosen quantities are inherently correlated because they have a common factor.
For example, NH/AV and AV are anti-correlated because of the common AV factor.
We visualize the errors and covariance between x and y for each point as an ellipse.

Modules
.......

``covariance.py`` contains the function that draws the scatter plot and the covariance ellipses, as well as helper functions to calculate the covariance between certain quantities

``get_data.py`` contains functions to retrieve the data and uncertainties for all the sightlines, and calculate some derived quantities.
Some functions to load data from other works are also available.

``linear_ortho_fit.py`` contains the line fitting method.
The chi2 function that is minimized to perform the linear fits is the one presented in Hogg et al. (2010).
It uses the perpendicular distance and projected variance to calculate chi2.
This way, the shape of the uncertainty ellipse of each point (x,y) is taken into account when estimating the slope of one of our scatter plots.

``pearson.py`` contains our method to calculate the pearson correlation coefficient r, as described in the paper.
The significance of r is expressed as a number of sigma, where sigma is the width of the r distribution if the data set was fully uncorrelated (r=0).
There is also a method to deal with correlations induced by correlated error bars between x and y.

``plot_fuse_results.py`` defines functions that make use of the above modules to do the analysis and make the desired plots

Scripts
.......

``paper_scatter.py`` calls functions from the above modules to do the analysis and make the plots for the various xy pairs of interest.

Misc scripts and modules
========================

``get_gaia.py`` looks up gaia data for each source. Results are written to ``data/gaia/merged.dat``. Distances are derived from this in ``get_data.py``.

``jenkins2009_fstar.py`` figure out the overlap with the sample of Jenkins (2009), and make some plots that involve the depletion ``F*``. Plots not used in paper, but they are mentioned.

``update_data_with_jenkins2019.py`` replace some of the values in ``data/fuse_h1_h2.dat`` with those provided by Jenkins et al. (2019).
Needs ``jenkins2019_hi_h2.fits``, which can be obtained by running ``get_jenkins2019_hi_h2.sh``.

``match_shull21`` compares some quantities of our sample to the values of Shull et al. (2021)

Any script starting in ``paper_`` makes plots that are used in the paper. They are run without arguments.

``paper_tables.tex`` generates Table 2 for the paper. It is no longer up to date with the headers and footnotes etc. Those edited updated manually before submission.

``photometric_distance.py`` contains some methods to calculate our own photometric distance for some sources.
It makes use of the file ``data/ob_mags.dat``, which contains magnitude models from Wegner 2007 Table 8 and Bowen 2008 appendix 3B.
In the end it went unused, as most of the sources had distances from Gaia DR2 and/or Shull et al. (2021). 

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

      The results are stored in ``data/<star name>``.
      This script did not find data for HD216898, so I downloaded those from the IUE archive manually.

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

2. Since Gaia parallaxes are known to be inaccurate for OB stars, we instead use photometric distances from Shull et al. 2021 for the 39 stars that overlap with our sample.
   The distance data are calculated / combined somewhere in ``get_data.py``

Scatter plots and fits
----------------------

One function per group of plots in ``paper_scatter.py``. It makes use of modules described above.

The data in the files mentioned above is loaded function in the ``get_data`` module, and main function to retrieve everything is `get_merged_table()`.
Derived columns, such as linear (instead of log) densities, uncertainties, calculated photometric distances, are calculated and added while this function is executed.

A more complex part of the code is where the covariances are calculated.

The main drawing and fitting functions are defined in ``plot_fuse_results.py``.
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
