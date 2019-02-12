import numpy as np
from astropy.io import ascii
from astropy.table import Table

MEu_MBC = 0.00711 # Arnould
#MEu_MBC = 0.00636 # Sneden

def print_collapsar():
    df = ascii.read("collapsar.tab").to_pandas()
    df["MFe"] = df["MNi"]
    MABC = df["M_A"] + df["M_BC"]
    df["f"] = df["M_A"]/df["M_BC"]
    df["logf"] = np.log10(df["f"])
    df["XLa"] = 0.14/(1+df["f"])
    df["logXLa"] = np.log10(df["XLa"])
    df["MEu"] = MEu_MBC * df["M_BC"]
    df["MEu/MFe"] = df["MEu"]/df["MFe"]
    df["nEu/nFe"] = df["MEu/MFe"] * 56/152.
    df["[Eu/Fe]"] = np.log10(df["nEu/nFe"]) - 0.52 + 7.5
    # Assuming 1e5 Msun of dilution
    df["[Fe/H]_5"] = np.log10(df["MFe"]/56) - 7.5 + 12 - 5.0
    df["[Eu/H]_5"] = np.log10(df["MEu"]/152) - 0.52 + 12 - 5.0
    df["[Fe/H]_6"] = np.log10(df["MFe"]/56) - 7.5 + 12 - 6.0
    df["[Eu/H]_6"] = np.log10(df["MEu"]/152) - 0.52 + 12 - 6.0
    print(df)
    printcols = ["model","MNi","f","logXLa","[Eu/Fe]","[Eu/H]_5","[Eu/H]_6"]
    print(df[printcols])
    


if __name__=="__main__":
    df = ascii.read("rproc_models.tab").to_pandas()
    
    df["f"] = df["M_A"]/df["M_BC"]
    df["logf"] = np.log10(df["f"])
    df["XLa"] = 0.14/(1+df["f"])
    df["logXLa"] = np.log10(df["XLa"])
    df["MEu"] = MEu_MBC * df["M_BC"]
    df["MEu/MFe"] = df["MEu"]/df["MFe"]
    df["nEu/nFe"] = df["MEu/MFe"] * 56/152.
    df["[Eu/Fe]"] = np.log10(df["nEu/nFe"]) - 0.52 + 7.5
    df["[Eu/H]_5"] = np.log10(df["MEu"]/152) - 0.52 + 12 - 5.0
    df["Mrtot"] = df["M_A"] + df["M_BC"]
    
    df["logXLa2"] = np.log10(df["M_La"]/df["Mrtot"])
    df["logXLa_diff"] = df["logXLa"] - df["logXLa2"]
    dfp = df[df["flag"]==1]
    print(dfp[["model","Mrtot","f","logXLa","logXLa2","logXLa_diff","Ref"]])#,"[Eu/H]_5","[Eu/Fe]"]])
    print()
    print(df[["model","Mrtot","f","logXLa","logXLa2","logXLa_diff","Ref"]])#,"[Eu/H]_5","[Eu/Fe]"]])
    
    Table.from_pandas(df).write("rproc_models_calc.tab", format='ascii.fixed_width_two_line')
