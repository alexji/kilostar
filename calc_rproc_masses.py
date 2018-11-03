from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import pickle

from astropy.io import ascii
from alexmods.smhutils import element_to_species, species_to_element


def load_lodders():
    lodders = ascii.read("lodders10_isotopes.txt").to_pandas()
    lodders.index = zip(lodders["Z"].values.astype(int), lodders["A"].values.astype(int))
    return lodders
def load_solar(model):
    assert model in ["lodders"]
    return load_lodders()
def load_sneden(whichdata):
    """ Loading the r-process isotopes from Sneden et al. 2008 (pickled in SMHR) """
    assert whichdata in ['rproc','sproc','sneden'], whichdata
    _datadir = "/Users/alexji/smhr/smh/data/isotopes"
    datamap = {'rproc':'sneden08_rproc_isotopes.pkl',
               'sproc':'sneden08_sproc_isotopes.pkl',
               'sneden':'sneden08_all_isotopes.pkl'}
    with open(_datadir+'/'+datamap[whichdata],'rb') as f:
        isotopes = pickle.load(f, encoding='latin1')
    return isotopes
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
def load_arnould_tab():
    path = "./rproc_patterns/"
    df = ascii.read(path+"/r_process_arnould_2007.txt",delimiter="&").to_pandas()
    A = df["A"]
    df["Z"] = list(map(lambda x: int(element_to_species(x)), df["Elem"]))
    df.index = zip(df["Z"].values.astype(int), df["A"].values.astype(int))
    return df

def load_bisterzo_rpat(model="B14",solar="lodders",log=True):
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
    f_s = f_s.append(th_u)
    f_r = 1.0-f_s
    if log:
        rpat = solar["logN"] + np.log10(f_r)
    else:
        rpat = solar["N"] * f_r
    rpat = rpat[pd.notnull(rpat)]
    return rpat
def load_arlandini_rpat(model="stellar", log=True):
    assert model in ["classical", "stellar"], model
    if model == "classical":
        Nrcol = "Nr2"
    elif model == "stellar":
        Nrcol = "Nr1"
    a99 = load_arlandini_tab()
    if log:
        rpat = np.log10(a99[Nrcol])
    else:
        rpat = a99[Nrcol]
    rpat = rpat[pd.notnull(rpat)]
    return rpat
def load_arnould_rpat(log=True):
    df = load_arnould_tab()
    if log:
        rpat = np.log10(df["N"])
    else:
        rpat = df["N"]
    rpat = rpat[pd.notnull(rpat)]
    return rpat
def load_sneden_rpat(log=True):
    df = ascii.read("sneden08_isotopes.txt").to_pandas()
    df.rename(columns={"mass":"A"}, inplace=True)
    index = [(Z,A) for _,(Z,A) in df[["Z","A"]].iterrows()]
    df.index = index
    rpat = df["Nr"] #df.groupby("A")["Nr"].sum()
    rpat.name = "N"
    if log:
        rpat = np.log10(rpat)
    rpat = rpat[pd.notnull(rpat)]
    return rpat

def find_mass_ratios(masses, Zmin=30, Z12=50):
    Z = masses.index.values
    M = masses.values
    ii1 = (Z >= Zmin) & (Z <= Z12)
    ii23 = Z > Z12
    ZLaMin, ZLaMax = 57, 71
    iiLa = (Z >= ZLaMin) & (Z <= ZLaMax)
    ZAcMin, ZAcMax = 89, 103
    iiAc = (Z >= ZAcMin) & (Z <= ZAcMax)
    
    M1  = M[ii1].sum()
    M23 = M[ii23].sum()
    MLa = M[iiLa].sum()
    MAc = M[iiAc].sum()
    return M1, M23, MLa, MAc

def find_massA_ratios(masses, Amin=70, A12=120):
    A = masses.index.values
    M = masses.values
    ii1 = (A >= Amin) & (A <= A12)
    ii23 = A > A12
    ALaMin, ALaMax = 139, 176
    iiLa = (A >= ALaMin) & (A <= ALaMax)
    AAcMin = 232
    iiAc = (A >= AAcMin)
    
    M1  = M[ii1].sum()
    M23 = M[ii23].sum()
    MLa = M[iiLa].sum()
    MAc = M[iiAc].sum()
    return M1, M23, MLa, MAc

def compute_masses(rpat, get_numbers=False):
    ## For each Z, calculate the total mass isotope by isotope.
    ## For each A, calculate the total mass isotope by isotope.
    masses = {}
    numbers = {}
    mean_masses = {}
    massesA = {}
    numbersA = {}
    allZ = []
    for (Z,A) in rpat.index:
        if Z in masses:
            masses[Z] += rpat[(Z,A)]*A
            mean_masses[Z] += rpat[(Z,A)]
            numbers[Z] += rpat[(Z,A)]
        else:
            masses[Z] = rpat[(Z,A)]*A
            mean_masses[Z] = rpat[(Z,A)]
            numbers[Z] = rpat[(Z,A)]
            allZ.append(Z)
        if A in massesA:
            massesA[A] += rpat[(Z,A)]*A
            numbersA[A] += rpat[(Z,A)]
        else:
            massesA[A] = rpat[(Z,A)]*A
            numbersA[A] = rpat[(Z,A)]
    for Z in allZ:
        mean_masses[Z] = masses[Z]/mean_masses[Z]
    masses = pd.Series(masses)
    massesA = pd.Series(massesA)
    numbers = pd.Series(numbers)
    numbersA = pd.Series(numbersA)
    if get_numbers:
        return masses, massesA, mean_masses, numbers, numbersA
    return masses, massesA, mean_masses

def calc_mass_in_ranges(massesA, Aranges):
    assert len(Aranges)==4
    Mranges = []
    for Arange in Aranges:
        Amin, Amax = Arange
        ix = np.logical_and(massesA.index >= Amin, massesA.index < Amax)
        Mranges.append(np.sum(massesA[ix]))
    return Mranges

def calc_ratio_ranges(massesA, Aranges, logMAscale=0.0):
    Mranges = calc_mass_in_ranges(massesA, Aranges)
    MA = Mranges[0] * (10**logMAscale)
    MB = Mranges[1] + Mranges[3]
    MC = Mranges[2]
    f = MA/(MB+MC)
    H = MC/(MB+MC)
    XLa = MC/(MA+MB+MC)
    return f, H, XLa

if __name__=="__main__":
#def tmp():
    lodders = load_lodders()
    bisterzo= load_bisterzo_tab()
    arlandini = load_arlandini_tab()
    b14 = load_bisterzo_rpat()
    a99 = load_arlandini_rpat()
    
    pattern_label = "bisterzo"
    rpat = load_bisterzo_rpat(log=False)
    pattern_label = "arlandini"
    rpat = load_arlandini_rpat(log=False)
    #pattern_label = "arnould"
    #rpat = load_arnould_rpat(log=False)
    rpatZarr = [_[0] for _ in rpat.index]
    rpatAarr = [_[1] for _ in rpat.index]
    
    masses, massesA, mean_masses = compute_masses(rpat)
    
    M1, M23, MLa, MAc = find_mass_ratios(masses, Z12=50)
    MLa = MLa + MAc
    M1 *= 10**-0.6
    print("Default by Z (Z12=50, Ascale=-0.6)")
    print("M1, M23, Mtot, MLa = {:.3f} {:.3f} {:.3f} {:.3f}".format(M1, M23, M1+M23, MLa))
    print("M1/M23 = {:.3f}".format(M1/M23))
    print("MLa/M23 = {:.3f}".format(MLa/M23))
    print("MLa/Mtot = {:.3f}".format(MLa/(M1+M23)))
    
    ### Figure showing effect of Z12
    #fig, ax = plt.subplots()
    fig, axes = plt.subplots(2,2, figsize=(12,8))
    ax = axes[0,0]
    ax2 = axes[1,0]
    ax.axvspan(49.5,50.5,ymin=0,ymax=.66,color='c', alpha=.15)
    ax.axvspan(53.5,54.5,ymin=0,ymax=1.,color='k', alpha=.2)
    Z12range = np.arange(46,56)
    for mass_lower_factor_exponent in [-0.6, -0.3, -0.1]:
        mass_lower_factor = 10**mass_lower_factor_exponent
        M1M23 = []
        MLaM23 = []
        MLaMtot = []
        for Z12 in Z12range:
            out = find_mass_ratios(masses, Z12=Z12)
            M1M23.append(mass_lower_factor * out[0]/out[1])
            MLaM23.append((out[2]+out[3])/out[1])
            MLaMtot.append((out[2]+out[3])/(out[0]+out[1]))
        l, = ax.plot(Z12range, M1M23, 'o-', label=r"$M_A/M_{{A,\odot}}=10^{{{:+.1f}}}$".format(mass_lower_factor_exponent))
        ax2.plot(Z12range, MLaM23, 'ko-')
    
        maxHobs = (np.array(M1M23)+1) * 10**-1.5
        Hobs = (np.array(M1M23)+1) * 10**-2.
        minHobs = (np.array(M1M23)+1) * 10**-2.5
        if mass_lower_factor_exponent==-0.1:
            ax2.fill_between(Z12range, minHobs, maxHobs, step='mid', color=l.get_color(), alpha=.3, lw=.7)
        ax2.plot(Z12range, Hobs, color=l.get_color(), linestyle='--', drawstyle='steps-mid')

    for Z12 in Z12range:
        ax.text(Z12,0.07,species_to_element(float(Z12)).split()[0],ha='center',va='bottom')
    for _ax in [ax, ax2]:
        _ax.xaxis.set_major_locator(MultipleLocator(1))
    ax2.set_xlabel(r"$Z_{AB}$ (boundary between 1st and 2nd peak)")
    ax.set_ylim(0,3.5)
    ax.set_ylabel(R"$f = M_A/M_{{B+C}}$")
    ax.legend()
    ax2.set_ylim(0, 0.3)
    ax2.set_ylabel(r"$H = M_C/M_{{B+C}}$")
    #ax2.axhspan(10**-2.5, 10**-1.5, color='red', alpha=.3)

    # Loading the mean masses to see the difference
    sneden = load_sneden("rproc")
    mean_masses_sneden = {}
    for elem in sneden.keys():
        Z = int(element_to_species(elem))
        isodict = sneden[elem]
        A = np.array(list(isodict.keys()))
        f = np.array(list(isodict.values()))
        mean_masses_sneden[Z] = np.sum(A*f)
    mean_masses = pd.Series(mean_masses)
    mean_masses_sneden = pd.Series(mean_masses_sneden)
    
    ## For each A, calculate the total mass. Use a sliding A cutoff.

    M1, M23, MLa, MAc = find_massA_ratios(massesA, A12=120)
    MLa = MLa + MAc
    M1 *= 10**-0.6
    print("Default by A (A12=120, Ascale=-0.6)")
    print("M1, M23, Mtot, MLa = {:.3f} {:.3f} {:.3f} {:.3f}".format(M1, M23, M1+M23, MLa))
    print("M1/M23 = {:.3f}".format(M1/M23))
    print("MLa/M23 = {:.3f}".format(MLa/M23))
    print("MLa/Mtot = {:.3f}".format(MLa/(M1+M23)))
    
    ### Figure showing effect of A12
    # Ye > 0.3, don't produce much of the 2nd peak
    # Ye 0.25 to 0.3 starts making significant amounts of 2nd peak
    #fig, ax = plt.subplots()
    ax = axes[0, 1]
    ax2 = axes[1, 1]
    ax.axvspan(119,121,ymin=0,ymax=.66,color='c', alpha=.15)
    ax.axvspan(129,131,ymin=0,ymax=1.,color='k', alpha=.2)
    Amin, Amax = 115,135
    A12range = np.arange(Amin, Amax+1)
    for mass_lower_factor_exponent in [-0.6, -0.3, -0.1]:
        mass_lower_factor = 10**mass_lower_factor_exponent
        M1M23 = []
        MLaM23 = []
        MLaMtot = []
        for A12 in A12range:
            out = find_massA_ratios(massesA, A12=A12)
            M1M23.append(mass_lower_factor * out[0]/out[1])
            MLaM23.append((out[2]+out[3])/out[1])
            MLaMtot.append((out[2]+out[3])/(out[0]+out[1]))
        l, = ax.plot(A12range, M1M23, 'o-', label=r"$\frac{{M_A}}{{M_{{A,\odot}}}}=10^{{{:+.1f}}}$".format(mass_lower_factor_exponent))
        ax2.plot(A12range, MLaM23, 'ko-')
        maxHobs = (np.array(M1M23)+1) * 10**-1.5
        Hobs = (np.array(M1M23)+1) * 10**-2.
        minHobs = (np.array(M1M23)+1) * 10**-2.5
        if mass_lower_factor_exponent==-0.1:
            ax2.fill_between(A12range, minHobs, maxHobs, step='mid', color=l.get_color(), alpha=.3, lw=.7,
                             label=r"$H$ for $X_{\rm{La}} = 10^{-2 \pm 0.5}$")
        ax2.plot(A12range, Hobs, color=l.get_color(), linestyle='--', drawstyle='steps-mid')
    ax.set_ylabel(R"$f = M_A/M_{{B+C}}$")
    for _ax in [ax, ax2]:
        _ax.set_xlim(Amin, Amax)
        _ax.xaxis.set_major_locator(MultipleLocator(5))
        _ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax2.set_xlabel(r"$A_{AB}$ (boundary between 1st and 2nd peak)")
    ax2.set_ylim(0, 0.3)
    ax2.set_ylabel(r"$H = M_C/M_{{B+C}}$")
    ax2.legend()
    #ax2.set_ylabel(r"$X_{\rm{La}} = M_C/M_{{A+B+C}}$")

    for _ax in [axes[1,0], axes[1,1]]:
        _ax.set_ylim(0,.3)
        _ax.yaxis.set_major_locator(MultipleLocator(.1))
        _ax.yaxis.set_minor_locator(MultipleLocator(.02))
    for _ax in [axes[0,0], axes[0,1]]:
        _ax.set_ylim(0,3.5)
        _ax.yaxis.set_major_locator(MultipleLocator(.5))
        _ax.yaxis.set_minor_locator(MultipleLocator(.1))

    fig.savefig("ZA12La_{}.pdf".format(pattern_label), bbox_inches="tight")
    
    fig, ax = plt.subplots()
    rpat1 = load_bisterzo_rpat(log=False)
    rpat2 = load_arlandini_rpat(log=False)
    rpat3 = load_arnould_rpat(log=False)
    masses1, massesA1, mean_masses1, numbers1, numbersA1 = compute_masses(rpat1, True)
    masses2, massesA2, mean_masses2, numbers2, numbersA2 = compute_masses(rpat2, True)
    masses3, massesA3, mean_masses3, numbers3, numbersA3 = compute_masses(rpat3, True)
    ax.plot(numbers1.index.values, np.log10(numbers1.values), 'o-', label="Bisterzo+14")
    ax.plot(numbers2.index.values, np.log10(numbers2.values), 'o-', label="Arlandini+99")
    ax.plot(numbers3.index.values, np.log10(numbers3.values)+1.9, 'o-', label="Arnould+07")
    ax.set_xlabel("Z")
    ax.set_ylabel(r"$\log N$")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend()
    fig.savefig("rpatZ_numbers.png", bbox_inches="tight")
    fig, ax = plt.subplots()
    ax.plot(numbersA1.index.values, np.log10(numbersA1.values), '.', label="Bisterzo+14")
    ax.plot(numbersA2.index.values, np.log10(numbersA2.values), '.', label="Arlandini+99")
    ax.plot(numbersA3.index.values, np.log10(numbersA3.values)+1.9, '.', label="Arnould+07")
    ax.set_xlabel("A")
    ax.set_ylabel(r"$\log N$")
    ax.legend()
    ax.set_ylim(-3,2)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    fig.savefig("rpatA_numbers.png", bbox_inches="tight")
    fig, ax = plt.subplots()
    ax.plot(masses1.index.values, np.log10(masses1.values), 'o-', label="Bisterzo+14")
    ax.plot(masses2.index.values, np.log10(masses2.values), 'o-', label="Arlandini+99")
    ax.plot(masses3.index.values, np.log10(masses3.values)+1.9, 'o-', label="Arnould+07")
    ax.set_xlabel("Z")
    ax.set_ylabel(r"$\log M$")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend()
    fig.savefig("rpatZ_masses.png", bbox_inches="tight")
    fig, ax = plt.subplots()
    ax.plot(massesA1.index.values, np.log10(massesA1.values), '.', label="Bisterzo+14")
    ax.plot(massesA2.index.values, np.log10(massesA2.values), '.', label="Arlandini+99")
    ax.plot(massesA3.index.values, np.log10(massesA3.values)+1.9, '.', label="Arnould+07")
    ax.set_xlabel("A")
    ax.set_ylabel(r"$\log M$")
    ax.legend()
    ax.set_ylim(-1,4)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    fig.savefig("rpatA_masses.png", bbox_inches="tight")

    plt.show()
