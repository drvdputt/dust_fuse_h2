"""Make latex tables for paper.

This script loads in the data, and uses it to create the latex tables
for the paper. Useful to update them, if some of our calculations
change.

Some notes about the output: 

1. It is not the most recent version!. The data can be copy pasted into
the tables used for the paper, but everything around it has been
manually edited (including some of the footnotes!). So watch out when
updating whatever is in the paper.

2. Distance is missing for HD099872 (no shull or gaia data). Could
replace this with my own photometric method, but the error bar was very
large anyway (.1 errorbar on .2 kpc distance).

"""
import get_data
import itertools
import numpy as np

# columns we want from the astropy Table (with and without uncertainties)
quantities_and_unc = ["lognhtot", "lognhi", "lognh2", "d"]
quantities_only = ["fh2", "denhtot"]
qu_headers = [
    r"$\log_{10} N(\text{H})$",
    r"$\log_{10} N(\text{\ion{H}{1}})$",
    r"$\log_{10} N(\text{H}_2)$",
    r"$d_{\mathrm{phot}}$",
]
q_headers = [
    r"$f(\text{H}_2)$",
    r"$\log_{10} n(\text{H})$",
]

# last column is hiref
num_columns = len(quantities_and_unc) * 2 + len(quantities_only) + 1


def format_row(table, index):

    name = table["Name"][index]
    values_and_unc = itertools.chain.from_iterable(
        (table[q][index], table[q + "_unc"][index]) for q in quantities_and_unc
    )
    values_only = [table[q][index] for q in quantities_only]
    hiref = table["hiref"][index]
    # subsitutions to make new reference legend work
    hiref = np.where(hiref == 9, 5, hiref)
    hiref = np.where(hiref == 15, 6, hiref)
    # hiref[hiref == 9] = 5
    # hiref[hiref == 15] = 6
    # then double check that no unexplained references are there
    if (hiref > 6).any():
        print("Warning! Problem with hiref")

    everything = [name, *values_and_unc, *values_only, hiref]
    fmt = (
        "{} "
        + "& {:.2f} " * (len(quantities_and_unc) * 2 + len(quantities_only))
        + "& {}"
        + " \\\\"
    )
    return fmt.format(*everything)


def header():
    column_format = (
        "l"  # name
        + r"r@{$\pm$}l" * len(qu_headers)  # with uncertainties
        + "c" * len(q_headers)  # without uncertainties
        + "c"  # hiref
    )
    lines = [
        r"\startlongtable" f"\\begin{{deluxetable*}}{{{column_format}}}",
        r"""\tablecaption{Total Gas Column Densities\label{tab:gasdetails}}
\tablewidth{0pt}
\tablehead{
    \colhead{Star}""",
    ]
    for label in qu_headers:
        lines.append(f" & \\multicolumn{{2}}{{c}}{{{label}}}")
    for label in q_headers:
        lines.append(f" & \\colhead{{{label}}}")
    lines.append(r" & \colhead{HI Ref.}")

    lines.append(
        r"""}
\startdata"""
    )
    return [l + "\n" for l in lines]


def footer():
    """This footer needs to contain the hi refs.

    The old version is
    (1) This paper
    (2) \citet{2006ApJ...641..327C};
    (3) \citet{1994ApJ...427..274D};
    (4) \citet{1990ApJS...72..163F};
    (5) \citet{1997AaA...321..531T};
    (6) \citet{2003ApJ...591.1000A};
    (7) \citet{1996MNRAS.279..788A};
    (8) \citet{2001ApJ...555..839R};
    (9) \citet{1994ApJ...430..630B};
    (10) \citet{2002ApJ...577..221R};
    (11) \citet{2005ApJ...619..891J};
    (13) \citet{1992ApJS...81..795G};
    (14) \citet{2007ApJ...669..378J};
    (15) \citet{2021ApJ...911...55S};

    But we're only using 1, 2, 3, 4, 9, and 15, so reorder here.
    Substitutions made: 9 --> 5, and 15 --> 6. This is done here,
    ad-hoc, in the main line formatting function.

    """
    return r"""\enddata
\tablerefs{
    (1) This paper
    (2) \citet{2006ApJ...641..327C};
    (3) \citet{1994ApJ...427..274D};
    (4) \citet{1990ApJS...72..163F};
    (5) \citet{1994ApJ...430..630B};
    (6) \citet{2021ApJ...911...55S};}
\end{deluxetable*}"""


def divider_message(s):
    return f"\\multicolumn{{{num_columns}}}{{c}}{{{s}}} \\\\ \\hline \n"


def data_lines(comp):
    data = get_data.get_merged_table(comp)
    return [format_row(data, i) + "\n" for i in range(len(data))]


def main():
    lines = header()
    lines.append(divider_message("Reddened Stars"))
    lines += data_lines(False)
    lines.append(divider_message("Comparison Stars"))
    lines += data_lines(True)
    lines += footer()

    with open("gastable.tex", "w") as f:
        f.writelines(lines)

main()
