import numpy as np
import matplotlib.pyplot as plt

def simple_model(A,B,C):
    Mtot = A+B+C
    XLa = C/(Mtot)
    f = A/(B+C)
    H = C/(B+C)
    return Mtot, XLa, f, H

def inverse_model1(XLa,f,H):
    """ Calculate MA, MB, MC, and Mtot given the other three """
    MA = Mtot * f/(1+f)
    MB = Mtot * (1-XLa-f*XLa)/(1+f)
    MC = Mtot * XLa
    H = MC/(MB+MC)
    return MA, MB, MC, H

def inverse_model4(Mtot,XLa,f):
    """ Calculate MA, MB, MC, and H given the other three """
    MA = Mtot * f/(1+f)
    MB = Mtot * (1-XLa-f*XLa)/(1+f)
    MC = Mtot * XLa
    H = MC/(MB+MC)
    return MA, MB, MC, H

if __name__=="__main__":
    mMtot = .035
    eMtot = .015
    mlogXLa = -2.
    elogXLa = 0.5
    mf = 0.6
    ef = 0.1
    mH = 0.13
    eH = .03
    
    XLa_arr = 10**np.array([mlogXLa + x for x in [-elogXLa, 0., elogXLa]])
    Mtotarr = [mMtot] #[mMtot + x for x in [-eMtot, 0., eMtot]]:
    farr = [0.5, 1.0, 2.0] #[mf + x for x in [-ef, 0., ef]]:    
    #farr = [0.5, 0.6, 0.7] #[mf + x for x in [-ef, 0., ef]]:    
    Harr = [.10, .13, .16]
    
    for XLa in XLa_arr:
        for Mtot in Mtotarr:
            for f in farr:
                _, _, _, H = inverse_model4(Mtot, XLa, f)
                extra = ""
                if np.abs(H-mH) < 2*eH: extra="<---"
                print("H={:.3f}: Mtot={:.03f}, XLa={:.03f}, f={:.2f}{}".format(
                    H, Mtot, XLa, f, extra))

    print()
    XLa_out = []
    for f in farr:
        for H in Harr:
            print H, f, H/(1.+f)
            XLa_out.append((H/(1.+f)))
    print("XLa = {:.3f}-{:.3f}".format(min(XLa_out), max(XLa_out)))
