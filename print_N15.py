from astropy.io import ascii
import numpy as np
import pandas as pd

from imp import reload
import calc_rproc_masses as crm; reload(crm);
from rstar_scatter import Arange1, Arange2, ArangeLa, Arange3, ArangeAc, Aranges

MEu_MBC = 0.00711 # Arnould
datadir = "/Users/alexji/Dropbox/RetII/FullPattern/rproc_yields/"

def load_N15(model):
    assert model in [1,2,3,4,5], model
    model_names = ["b11tw0.25","b11tw1.00","b12tw0.25","b12tw1.00","b12tw4.00"]
    model_name = model_names[model-1]
    masses = {"b11tw0.25":2.68000e-02,"b11tw1.00":2.15000e-02,"b12tw0.25":3.55000e-02,
              "b12tw1.00":4.37000e-02,"b12tw4.00":8.57000e-02}
    total_mass = masses[model_name]
    fname = datadir+"/Nishimura15/{}.dat".format(model_name)
    # abundance is Y = X/A, multiply by total mass to get number of atoms of that isotope
    df = ascii.read(fname, names=["name","Z","N","A","X","abundance"]).to_pandas()
    return df
def load_N15_Z(model):
    df = load_N15(model)
    # Sum out isotopes
    s = df.groupby("Z").sum()["abundance"] * total_mass
    s.name = model_name
    return s
def load_N17(model):
    # L=1.0 is h(eating); L=1.25 is h+
    # L=0.6 is i(intermediate); L=0.4/0.75 is i-/+
    # L=0.2 is m(agnetic); L=0.4 is m+
    all_L = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25]
    assert model in range(1, len(all_L)+1), model
    L = all_L[model-1]
    model_name = "L{:.2f}".format(L)
    masses = {"L0.10":7.66225e-03,"L0.20":1.34505e-02,"L0.30":2.09197e-02,
              "L0.40":3.19371e-02,"L0.50":6.17773e-02,"L0.60":1.19465e-01,
              "L0.75":2.00467e-01,"L1.00":2.38585e-01,"L1.25":2.61565e-01}
    total_mass = masses[model_name]
    fname = datadir+"/Nishimura17/{}.dat".format(model_name)
    # abundance is Y = X/A, multiply by total mass to get number of atoms of that isotope
    df = ascii.read(fname, names=["name","Z","N","A","X","abundance"]).to_pandas()
    return df
def load_N17_Z(model, retmass=False):
    df = load_N17(model)
    # Sum out isotopes
    if retmass:
        s = df.groupby("Z").sum()["X"] * total_mass
    else:
        s = df.groupby("Z").sum()["abundance"] * total_mass
    s.name = model_name
    return s
def load_W16(model):
    assert model in [1,2,3,4,5]
    #def is fiducial
    #m0.01, 0.10: disk mass
    #s10, s6: entropy
    model_names = ["def","m0.01","m0.10","s10","s6"]
    model_name = model_names[model-1]
    fname = datadir+"/Wu2016/s_{}_tgyr_elemplot_intd".format(model_name)
    tab = ascii.read(fname, names=["Z","abundance"])
    s = pd.Series(tab["abundance"], index=tab["Z"], name=model_name)

    rpat3 = crm.load_arnould_rpat(log=False)
    masses3, massesA3, mean_masses3, numbers3, numbersA3 = crm.compute_masses(rpat3, True)
    mean_mass = masses3/numbers3 # amu/number
    for Z in s.index:
        try:
            s.loc[Z] = s.loc[Z] * mean_mass.loc[Z]
        except:
            s.loc[Z] = np.nan
    s = s[pd.notnull(s)]
    return s

def load_N15_custom(model):
    df = load_N15(model)
    index = [(Z,A) for ix,(Z,A) in df[["Z","A"]].iterrows()]
    df.index = index
    df = df["abundance"]
    df = df.rename(columns={"abundance":"N"})
    return df
def load_N17_custom(model):
    df = load_N17(model)
    index = [(Z,A) for ix,(Z,A) in df[["Z","A"]].iterrows()]
    df.index = index
    df = df["abundance"]
    df = df.rename(columns={"abundance":"N"})
    return df



if __name__=="__main__":
    names = ["B11W025","B11W100","B12W025","B12W100","B12W400"]
    for model in [1,2,3,4,5]:
        df = load_N15_custom(model)
        masses, massesA, mean_masses, numbers, numbersA = crm.compute_masses(df, True)
        Mranges = crm.calc_mass_in_ranges(massesA,Aranges)
        MA, MBC = Mranges[0], Mranges[1]+Mranges[2]+Mranges[3]
        MFe = masses[26]
        MLa = Mranges[3]
        print("mrdsn_{}   {:>5.2f} {:>5.2f} {:>5.3f} {:>5.2f} N15".format(names[model-1], MA, MBC, MLa, MFe))

    all_L = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25]
    for model in range(9):
        model = model+1
        df = load_N17_custom(model)
        masses, massesA, mean_masses, numbers, numbersA = crm.compute_masses(df, True)
        Mranges = crm.calc_mass_in_ranges(massesA,Aranges)
        MA, MBC = Mranges[0], Mranges[1]+Mranges[2]+Mranges[3]
        MFe = masses[26]
        MLa = Mranges[3]
        print("mrdsn_L{:03}      {:>5.2f} {:>5.2f} {:>5.3f} {:>5.2f} N17".format(int(all_L[model-1]*100), MA, MBC, MLa, MFe))
        #print("MEu={} or {}".format(masses[63], MEu_MBC*MBC))

    for model in [1,2,3,4,5]:
        masses = load_W16(model)
        Zranges = [[31,50],[50,92],[57,72]]
        Mranges = crm.calc_mass_in_ranges(masses,Zranges)
        MA, MBC = Mranges[0], Mranges[1]
        #MFe = masses[26]
        MFe = np.nan
        MLa = Mranges[2]
        print("nsmwind_{}      {:>5.2f} {:>5.2f} {:>5.3f} {:>5.2f} W16".format("xx", MA, MBC, MLa, MFe))
        
    rpat3 = crm.load_arnould_rpat(log=False)
    masses3, massesA3, mean_masses3, numbers3, numbersA3 = crm.compute_masses(rpat3, True)
    Mranges3 = crm.calc_mass_in_ranges(massesA3,Aranges)
    MEu = masses3[63]
    MA3, MBC3 = Mranges3[0], Mranges3[1]+Mranges3[2]+Mranges3[3]
    print("Arnould MEu/MBC = {:.2e}/{:.2e} = {:.2e}".format(MEu, MBC3, MEu/MBC3))
    print("Error from forgetting XLa = {}, log={}".format(Mranges3[2]/MBC3, np.log10(1+Mranges3[2]/MBC3)))

    rpat3 = crm.load_sneden_rpat(log=False)
    masses3, massesA3, mean_masses3, numbers3, numbersA3 = crm.compute_masses(rpat3, True)
    Mranges3 = crm.calc_mass_in_ranges(massesA3,Aranges)
    MEu = masses3[63]
    MA3, MBC3 = Mranges3[0], Mranges3[1]+Mranges3[2]+Mranges3[3]
    print("Sneden MEu/MBC = {:.2e}/{:.2e} = {:.2e}".format(MEu, MBC3, MEu/MBC3))
    print("Error from forgetting XLa = {}, log={}".format(Mranges3[2]/MBC3, np.log10(1+Mranges3[2]/MBC3)))
