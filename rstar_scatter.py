
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import ascii
from astropy.table import Table
from imp import reload
from alexmods import read_data as rd; reload(rd);
import calc_rproc_masses as crm; reload(crm);
from astropy.stats import biweight_scale
import seaborn as sns

# Full range of everything
ZrangeA_1 = [31,47]
ZrangeBC_1= [56,77]

# Minimum range: Good Lanthanides
ZrangeA_2 = [38,40]
ZrangeBC_2= [57,66]

## Copied from Results.ipynb
Arange1 = [70,115]
Arange2 = [115,140]
ArangeLa= [140,176]
Arange3 = [176,211]
ArangeAc = [228,240]
Aranges = [Arange1,Arange2,ArangeLa,Arange3]

rpat3 = crm.load_arnould_rpat(log=False)
rpat4 = crm.load_sneden_rpat(log=False)
masses3, massesA3, mean_masses3, numbers3, numbersA3 = crm.compute_masses(rpat3, True)
masses4, massesA4, mean_masses4, numbers4, numbersA4 = crm.compute_masses(rpat4, True)
Mranges3 = crm.calc_mass_in_ranges(massesA3,Aranges)
Mranges4 = crm.calc_mass_in_ranges(massesA4,Aranges)
MA3, MBC3 = Mranges3[0], Mranges3[1]+Mranges3[3]+Mranges3[2] # OOPS I FORGET TO ADD THE LANTHANIDES THIS WAS JUST MB
MA4, MBC4 = Mranges4[0], Mranges4[1]+Mranges4[3]+Mranges4[2] # It increases MBC by 0.07 and 0.06 dex, increases XLa by that much too

a = set(numbers3.index)
b = set(numbers4.index)
Zs = np.sort(list(a.union(b)))

def process_df(df, verbose=True, Nelemmin=5):
    for errcol in rd.errcolnames(df):
        elem = rd.getelem(errcol)
        epscol = rd.epscol(elem)
        ii = df[errcol] < 0
        df.loc[ii,epscol] = np.nan
        df.loc[ii,errcol] = np.nan
        if verbose: print("{}: removed {} upper limits".format(elem, np.sum(ii)))
    epscols = rd.epscolnames(df)
    epsZs = [rd.epscol(rd.getelem(Z)) for Z in Zs]
    epscols = np.array(epscols)[np.in1d(epscols,epsZs)]
    df["Nelems"] = np.sum(pd.notnull(df[epscols]), axis=1)
    iiminelem = np.array(df["Nelems"] >= Nelemmin)
    df = df[iiminelem]
    print("Cut to {}/{} stars with >= {} elems".format(len(df),len(iiminelem),Nelemmin))
    
    offsets = np.zeros((len(df),2))
    df["offsetA_1_A07"] = np.nan
    df["offsetBC_1_A07"] = np.nan
    df["offsetA_1_S08"] = np.nan
    df["offsetBC_1_S08"] = np.nan
    df["offsetA_2_A07"] = np.nan
    df["offsetBC_2_A07"] = np.nan
    df["offsetA_2_S08"] = np.nan
    df["offsetBC_2_S08"] = np.nan
    for i in range(len(df)):
        mystar, mystarerr = make_star_series(df,i)
        for rangenum, source in [[1,"A07"],[1,"S08"],[2,"A07"],[2,"S08"]]:
            if source == "A07": rpat = np.log10(numbers3)
            elif source=="S08": rpat = np.log10(numbers4)
            if rangenum == 1: ZrangeA, ZrangeBC = ZrangeA_1, ZrangeBC_1
            elif rangenum==2: ZrangeA, ZrangeBC = ZrangeA_2, ZrangeBC_2
            out = find_offsets(mystar, rpat, ZrangeA, ZrangeBC)
            col1 = "offsetA_{}_{}".format(rangenum,source)
            col2 = "offsetBC_{}_{}".format(rangenum,source)
            col3 = "offseterr_{}_{}".format(rangenum,source)
            df.loc[df.index[i],col1] = out[0]
            df.loc[df.index[i],col2] = out[1]
            df.loc[df.index[i],col3] = out[4]

            out = find_offsets(mystar, rpat, ZrangeA, ZrangeBC, starerrs=mystarerr)
            col1 = "woffsetA_{}_{}".format(rangenum,source)
            col2 = "woffsetBC_{}_{}".format(rangenum,source)
            col3 = "woffseterr_{}_{}".format(rangenum,source)
            df.loc[df.index[i],col1] = out[0]
            df.loc[df.index[i],col2] = out[1]
            df.loc[df.index[i],col3] = out[4]

    for rangenum, source in [[1,"A07"],[1,"S08"],[2,"A07"],[2,"S08"]]:
        if source == "A07": MA, MBC = MA3, MBC3
        elif source=="S08": MA, MBC = MA4, MBC4
        col1 = "offsetA_{}_{}".format(rangenum,source)
        col2 = "offsetBC_{}_{}".format(rangenum,source)
        col3 = "offseterr_{}_{}".format(rangenum,source)
        df["delta_{}_{}".format(rangenum,source)] = df[col1]-df[col2]
        df["f_{}_{}".format(rangenum,source)] = 10**(df[col1]-df[col2]) * MA/MBC
        df["ferr_{}_{}".format(rangenum,source)] = 10**(df[col1]-df[col2]) * MA/MBC * np.log(10.) * df[col3]

        col1 = "woffsetA_{}_{}".format(rangenum,source)
        col2 = "woffsetBC_{}_{}".format(rangenum,source)
        col3 = "woffseterr_{}_{}".format(rangenum,source)
        df["wdelta_{}_{}".format(rangenum,source)] = df[col1]-df[col2]
        df["wf_{}_{}".format(rangenum,source)] = 10**(df[col1]-df[col2]) * MA/MBC
        df["wferr_{}_{}".format(rangenum,source)] = 10**(df[col1]-df[col2]) * MA/MBC * np.log(10.) * df[col3]

    df["[Ba/Eu]"] = df["epsba"] - 2.18 - df["epseu"] + 0.52
    return df


def find_offsets(star, rpat, ZrangeA, ZrangeBC, starerrs=None):
    """ Find dlogepsA, dlogepsBC """
    output = []
    def get_solar_value(Z):
        if Z in rpat: return rpat[Z]
        else: return np.nan
    for Zrange in [ZrangeA, ZrangeBC]:
        _star = star[np.array([np.logical_and(Z >= Zrange[0], Z <= Zrange[1]) for Z in star.index])]
        alldlogeps = np.array([logeps-rpat[Z] for Z, logeps in _star.items()])
        if np.all(np.isnan(alldlogeps)):
            output.append(np.nan)
            output.append(np.nan)
        else:
            mind, maxd = round(np.nanmin(alldlogeps),2)-0.1, round(np.nanmax(alldlogeps),2)+0.1
            searchd = np.arange(mind,maxd+0.1,0.01) # brute force minimization
            if starerrs is None:
                absdev = np.nansum(np.abs(alldlogeps[:,np.newaxis] - searchd[np.newaxis,:]), axis=0)
            else:
                allerrs = np.array([starerrs[Z] for Z, logeps in _star.items()])
                absdev = np.nansum(np.abs(alldlogeps[:,np.newaxis] - searchd[np.newaxis,:])/allerrs[:,np.newaxis], axis=0)
            dlogeps = searchd[np.argmin(absdev)]
            output.append(dlogeps)
            e_dlogeps = alldlogeps - dlogeps
            e_dlogeps = biweight_scale(e_dlogeps[np.isfinite(e_dlogeps)])
            output.append(e_dlogeps)
    dlogepsA, e_dlogepsA, dlogepsBC, e_dlogepsBC = output
    return dlogepsA, dlogepsBC, e_dlogepsA, e_dlogepsBC, np.sqrt(e_dlogepsA**2 + e_dlogepsBC**2)

def make_star_series(df, irow):
    mystar = df.iloc[irow]
    epscols = [rd.epscol(rd.getelem(Z)) for Z in Zs]
    epscolii = np.array([col in df.columns for col in epscols])
    epscols = np.array(epscols)[epscolii]
    errcols = np.array([rd.errcol(rd.getelem(col)) for col in epscols])
    mystar2 = pd.Series(mystar[epscols].values, index=Zs[epscolii])
    mystar2err = pd.Series(mystar[errcols].values, index=Zs[epscolii])
    return mystar2, mystar2err

def make_samples(df,flogerr=0.2,Nsamples=100,ftype="wf_1"):
    f = df[ftype+"_A07"].dropna()
    #XLa = np.concatenate([(0.14+x)/(1+f) for x in [-0.03,0.00,+0.03]])
    #plt.plot(np.tile(f,3),XLa,'.')
    XLa = 0.14/(1+f)
    #print("wf_1_A07",np.min(f),np.max(f), np.min(XLa),np.max(XLa))
    print("{} f={:.1f}-{:.1f} XLa={:.3f}-{:.3f}".format(ftype+"_A07",np.min(f),np.max(f),np.min(XLa),np.max(XLa)))
    # Add independent errors for H (0.03) and f (logf=0.2)
    #fsample = np.tile(f,100) + np.random.randn(len(f)*100)*0.1
    fsample = np.tile(f,Nsamples) * 10**(flogerr*np.random.randn(len(f)*Nsamples))
    Hsample = 0.14 + np.zeros_like(fsample)*0.03
    XLasample = Hsample/(1+fsample)

    fS08 = df[ftype+"_S08"].dropna()
    XLaS08 = 0.14/(1+fS08)
    print("{} f={:.1f}-{:.1f} XLa={:.3f}-{:.3f}".format(ftype+"_S08",np.min(fS08),np.max(fS08),np.min(XLaS08),np.max(XLaS08)))
    #fsampleS08 = np.tile(fS08,100) + np.random.randn(len(fS08)*100)*0.1
    fsampleS08 = np.tile(fS08,Nsamples) * 10**(flogerr*np.random.randn(len(fS08)*Nsamples))
    HsampleS08 = 0.14 + np.zeros_like(fsampleS08)*0.03
    XLasampleS08 = HsampleS08/(1+fsampleS08)
    
    samples = [[fsample, Hsample, XLasample], [fsampleS08, HsampleS08, XLasampleS08]]
    return samples

def make_plot(df,
              XLamin=0.00, XLamax=0.08,
              fmin=0.0, fmax=30.0,
              percentile_levels = [50,70,90,99.7,100],
              plot_gw170817=True,
              flogerr=0.2, plot_log=False,
              samples=None, get_samples=False):
    
    if samples is None: samples = make_samples(df,flogerr=flogerr)
    [fsample, Hsample, XLasample], [fsampleS08, HsampleS08, XLasampleS08] = samples
    
    xbins = np.arange(fmin,fmax,10**flogerr)
    ybins = np.arange(XLamin,XLamax,.005)
    X, Y = np.meshgrid(xbins,ybins)
    H, xe, ye = np.histogram2d(fsampleS08,XLasampleS08,bins=[xbins,ybins])
    xx = (xe[1:]+xe[:-1])/2.; yy = (ye[1:]+ye[:-1])/2.; XX, YY = np.meshgrid(xx,yy)
    
    import seaborn as sns
    g = sns.jointplot(fsample, XLasample, kind="kde",stat_func=None, #label="A07",
                      xlim=[fmin,fmax],ylim=[XLamin,XLamax], shade_lowest=False,
                      color='orange')
    for ax in [g.ax_joint, g.ax_marg_x]:
        ax.set_xlabel("f", fontsize=20)
    for ax in [g.ax_joint, g.ax_marg_y]:
        ax.set_ylabel("XLa", fontsize=20)
    
    ax = g.ax_joint
    
    cmap = sns.light_palette("browny orange", as_cmap=True, input="xkcd", reverse=True)
    color2 = cmap(0)
    
    _H = np.ravel(H[H > 0])
    levels = [np.percentile(_H, p) for p in percentile_levels]
    #ax.contour(XX,YY,H.T,levels,colors='orange', label="S08", linestyles=':')
    sns.kdeplot(fsampleS08,XLasampleS08,shade=False,shade_lowest=False,linestyles=':',
                ax=g.ax_joint,legend=False,cmap=cmap)
    sns.distplot(fsampleS08,color=color2,ax=g.ax_marg_x,hist=False,bins=xbins, kde_kws={'ls':':'})
    sns.distplot(XLasampleS08,color=color2,ax=g.ax_marg_y,hist=False,vertical=True, kde_kws={'ls':':'})
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    if plot_gw170817:
        for ax in [g.ax_joint, g.ax_marg_y]:
            ax.axhline(10**-2,color='red',linestyle='--')
            ax.axhline(10**-2.5,color='red',linestyle=':')
            ax.axhline(10**-1.5,color='red',linestyle=':')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='orange', label="A07", lw=2),
        Patch(facecolor='none',edgecolor=color2,label="S08",linestyle=':', lw=2)
    ]
    g.ax_joint.legend(handles=legend_elements, loc='upper right', fontsize=20)

    fA07sun = MA3/MBC3; XLaA07sun = .14/(1+fA07sun)
    fS08sun = MA4/MBC4; XLaS08sun = .14/(1+fS08sun)
    ax = g.ax_joint
    #ax.plot(fA07sun, XLaA07sun, 's', mfc='none', mec=sns.color_palette()[0], zorder=9, ms=15, mew=3)
    #ax.plot(fS08sun, XLaS08sun, 's', mfc='none', mec='orange', zorder=9, ms=15, mew=3)
    ax.plot(fA07sun, XLaA07sun, 's', mfc='none', mec='orange', zorder=9, ms=15, mew=3)
    ax.plot(fS08sun, XLaS08sun, 's', mfc='none', mec=color2, zorder=9, ms=15, mew=3)
    
    fmin, fmax = g.ax_joint.get_xlim()
    fplot = np.linspace(fmin,fmax)
    XLaplot = 0.14/(1+fplot)
    g.ax_joint.plot(fplot, XLaplot, 'c', lw=1)
    
    if plot_log:
        g.ax_joint.set_yscale('log')
        g.ax_joint.set_ylim(10**-3.5,10**-0.5)
        g.ax_marg_y.set_yscale('log')
        g.ax_marg_y.set_ylim(10**-3.5,10**-0.5)        
    
    if get_samples: return samples, g.fig
    return g.fig
def add_eufe_panel(fig,df,ylim=(0.5,2.0),xlim=(0,20)):
    ax = fig.axes[0]
    fig.set_figheight(10)
    fig.subplots_adjust(bottom=.4)
    ax2 = fig.add_axes((fig.subplotpars.left,.1,0.725,.24))
    ax2.plot(df["f_1_A07"], df["[Eu/Fe]"], 's')
    orange = sns.color_palette()[1]
    ax2.plot(df["f_1_S08"], df["[Eu/Fe]"], 's', mfc='none', mec=orange, mew=1.5)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_xlabel("f", fontsize=20)
    ax2.set_ylabel("[Eu/Fe]", fontsize=20)
    ax2.axhline(1.0,color='k',ls=':',lw=1)
    return fig
