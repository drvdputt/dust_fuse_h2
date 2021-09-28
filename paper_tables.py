"""Make latex tables for paper.

This script loads in the data, and uses it to create the latex tables
for the paper. Useful to update them, if some of our calculations
change.

"""
import get_data
import itertools

# columns we want from the astropy Table (with and without uncertainties)
quantities_and_unc = ["lognhtot", "lognhi", "lognh2"]
quantities_only = ["fh2", "denhtot", "d"]
qu_headers = [
    r"$\log_{10} N(\text{H})$",
    r"$\log_{10} N(\text{\ion{H}{1}})$",
    r"$\log_{10} N(\text{H}_2)$",
]
q_headers = [
    r"$f(\text{H}_2)$",
    r"$\log_{10} n(\text{H})$",
    "$d$",
]

num_columns = len(quantities_and_unc) * 2 + len(quantities_only) + 1


def format_row(table, index):

    name = table["Name"][index]
    values_and_unc = itertools.chain.from_iterable(
        (table[q][index], table[q + "_unc"][index]) for q in quantities_and_unc
    )
    values_only = [table[q][index] for q in quantities_only]
    hiref = table["hiref"][index]

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

    lines.append(
        r"""}
\startdata"""
    )
    return [l + "\n" for l in lines]


def footer():
    return r"""
\enddata
\tablenotetext{a}{Where estimated by reddening, the $N$(\ion{H}{1}) uncertainty
    is derived from adopting a 0.15 dex error allowance in $N$(H), including the
    30\% scatter noted by \citet{1978ApJ...224..132B} in the linear relationship between
    $N$(H$_{\rm total}$) and $E$($\bv$) and a small margin for $E$($\bv$)
    uncertainty, and the $N$(H$_2$) error allowance.}
\tablerefs{
    (1) This paper, either measured or by $N$(\ion{H}{1}) = 5.8$\times$10$^{21}
    E$($\bv$)$ - 2N$(H$_2$);
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
    (14) \citet{2007ApJ...669..378J}.}

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

    with open("test.tex", "w") as f:
        f.writelines(lines)
