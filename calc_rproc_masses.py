from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from astropy.io import ascii
from alexmods.smhutils import element_to_species
import pandas as pd

def load_lodders():
    lodders = ascii.read("lodders10_isotopes.txt").to_pandas()
    lodders.index = zip(lodders["Z"].values.astype(int), lodders["A"].values.astype(int))
    return lodders
def load_solar(model):
    assert model in ["lodders"]
    return load_lodders()

def load_bisterzo_tab():
    path = "./rproc_patterns/bisterzo14/"
    tab = ascii.read(path+"table1.dat", readme=path+"ReadMe")
    tab = tab[np.array(list(map(lambda x: "^" in x, tab["Isotope"])))]
    A = list(map(lambda x: x.split("^")[1], tab["Isotope"]))
    El = list(map(lambda x: x.split("^")[2], tab["Isotope"]))
    Z = list(map(lambda x: int(element_to_species(x)), El))
    tab.add_columns([tab.Column(A, "A"), tab.Column(Z, "Z"), tab.Column(El,"El")])
    df = tab.to_pandas()
    df.index = zip(df["Z"].values.astype(int), df["A"].values.astype(int))
    return df

def load_arlandini_tab():
    path = "./rproc_patterns/"
    df = ascii.read(path+"/arlandini99.txt").to_pandas()
    df["Z"] = list(map(lambda x: int(element_to_species(x)), df["El"]))
    df.index = zip(df["Z"].values.astype(int), df["A"].values.astype(int))
    return df

def load_bisterzo_rpat(model="B14",solar="lodders"):
    assert model in ["T04", "B14"], model
    assert solar in ["lodders"]
    if model=="B14":
        dcol = "TW+"
        ecol = "e_TW+"
    if model=="T04":
        dcol = "T04+"
        ecol = "e_T04+"
    b14 = load_bisterzo_tab()
    solar = load_solar(solar)
    
    f_s = b14[dcol]*.01
    th_u_keys = [(90,232),(92,234),(92,235),(92,238)]
    th_u = pd.Series([0 for _ in th_u_keys], index=th_u_keys)
    f_s.append(th_u)
    f_r = 1.0-f_s
    rpat = solar["logN"] + np.log10(f_r)
    rpat = rpat[pd.notnull(rpat)]
    return rpat

def load_arlandini_rpat(model="stellar",solar="lodders"):
    assert model in ["classical", "stellar"], model
    assert solar in ["lodders"]
    if model == "classical":
        Nrcol = "Nr2"
    elif model == "stellar":
        Nrcol = "Nr1"
    a99 = load_arlandini_tab()
    solar = load_solar(solar)
    
    f_s = b14[dcol]*.01
    th_u_keys = [(90,232),(92,234),(92,235),(92,238)]
    th_u = pd.Series([0 for _ in th_u_keys], index=th_u_keys)
    f_s.append(th_u)
    f_r = 1.0-f_s
    rpat = solar["logN"] + np.log10(f_r)
    rpat = rpat[pd.notnull(rpat)]
    return rpat

if __name__=="__main__":
    lodders = load_lodders()
    bisterzo= load_bisterzo_tab()
    arlandini = load_arlandini_tab()
    b14 = load_bisterzo_rpat()
    
