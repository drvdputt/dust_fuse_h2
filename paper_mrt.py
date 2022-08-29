"""Machine readable tables"""

from get_data import get_merged_table, get_fuse_h2_details

data = get_merged_table()
keep_columns = ['Name']
q_with_unc = ['lognhtot', 'lognhi', 'lognh2', 'd']
for q in q_with_unc:
    keep_columns += [q, q + '_unc']
q_other = ['fh2', 'denhtot', 'hiref']
for q in q_other:
    keep_columns += [q]
data[keep_columns].write("paper-tables/mrt_test.dat", format="mrt", overwrite=True)

h2data = get_fuse_h2_details(components=True, stars=data['Name'])
keep_columns_h2 = ['Name']
keep_columns_h2 += [c for c in h2data.colnames if 'log' in c]
h2data.write("paper-tables/mrt_h2.dat", format="mrt", overwrite=True)
