Analysis for FUSE H2/Extinction paper
=====================================

This paper is Van De Putte, Clayton, Gordon, et al. (in prep).

The H2 absorption line modeling to the FUSE data is done.
The results are used to determine NH2

More recently, Lyman alpha absorption fitting was done to determine NHI.
For this, IUE and HST STIS data were used.

The rest of the analysis compares the extinction results from Gordon et al. (2009) with the newly obtained H2 and HI results

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

