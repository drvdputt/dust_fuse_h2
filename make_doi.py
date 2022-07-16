"""This script gathers all the obsids of the spectra that I used. This
list can then be given to the MAST DOI tool."""

from get_spectrum import target_use_which_spectrum
from pathlib import Path
from astropy.io import fits


def obsid_for_file(fn):
    base_name = Path(fn).name
    
    if 'mxlo' in fn:
        obsid = base_name.split("mxlo")[0]
    
    if 'mxhi' in fn:
        # for IUE, the obsid is swpxxxxx
        obsid = base_name.split('.')[0]

    if 'x1d' in fn:
        # for HST, obsid is obicxxxxx
        obsid = base_name.split('_')[0]
    
    return obsid
    

def obsids_for_target(target):
    file_spec = target_use_which_spectrum[target]
    if isinstance(file_spec, list):
        files = file_spec
    elif isinstance(file_spec, str):
        file_globstring = target_use_which_spectrum[target]
        files = [str(p) for p in Path(".").glob(file_globstring)]
    return [obsid_for_file(f) for f in files]
    
results = {}
for target in target_use_which_spectrum:
    results[target] = obsids_for_target(target)

print(results)
plain_list = []
for t in results:
    plain_list += results[t]

print(plain_list)
