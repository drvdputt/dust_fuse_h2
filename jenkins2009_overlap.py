import numpy as np
from astropy.table import Table
tjenkins = Table.read('data/jenkins2009-table2.dat', format='ascii.cds')

# the first two BD stars overlap

from get_data import get_merged_table

data = get_merged_table()

our_hd_numbers = [int(data['Name'][i][2:]) for i in range(len(data)) if 'HD' in data['Name'][i]]

jenkins_hd_numbers = tjenkins['HD'][~tjenkins['HD'].mask]

hd_set = set(our_hd_numbers) & set(jenkins_hd_numbers[~jenkins_hd_numbers.mask])

print(f"The following {len(hd_set)} HD stars overlap")
print(hd_set)
print(f"And also these two that I found manually: BD+35d4258 and BD+53d2820")

# this does not work without converting to list first, for some reason
tjenkins_overlap = np.isin(tjenkins['HD'], list(hd_set)) 
# our data overlap based on number
data_overlap = [name[:2] == 'HD' and int(name[2:]) in hd_set for name in data['Name']]

nhav = data['NH_AV'][data_overlap]

# to get fstar, we need to load another table...
fstar = tjenkins_depletion['Fstar'][tjenkins_overlap]
