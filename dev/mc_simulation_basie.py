from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import norm, logistic, norm
from scipy.special import ndtri, expit, erf, erfinv, erfc, erfcinv
import matplotlib.pyplot as plt
import datetime
import math
import time
import random
import glob

from basie_functions import *

# run MC simulations

NMC=10 # 1000 MC simulations
nt=20 # of 300 trials each

modelp = np.array([[0.5], [0.04], [0.2], [-20], [20], [0.01], [0.5]])
basiep={'cs': 0.5}

nmodels=3

availsnr=np.linspace(-20, 20, 41).T
snrlist=[]
for i in range(nmodels):
    snrlist.append(availsnr)
snrlist=np.asarray(snrlist)

srterror=np.empty((NMC, nt,nmodels))
slopeerror=np.empty((NMC,nt,nmodels))
srtvar=np.empty((NMC,nt,nmodels))
slopevar=np.empty((NMC,nt,nmodels))

for imc in range(NMC):
	[snr, ii, __, __]=v_psycest(-nmodels,  np.repeat(modelp, nmodels, axis=1), basiep, snrlist,1)

	truemodel=np.array([[0.5], [0.0], [0.01], [0.02], [0.01], [1]]) # prob@thresh, thresh(dB), slope@thresh(db-1), prob(miss), prob(guess), fct type

	for i in range(nt):
	    [response, __] = v_psychofunc('r',truemodel,np.array([snr]));
	#     response = np.array([[bool(listofresponses[i])]])
	    # print('snr: ',snr, 'response: ',response, 'model: ', ii)
	    [snr, ii, m, v] = v_psycest(ii, snr, response);
	    srterror[imc, i] = m[0,:,0]-truemodel[1]
	    slopeerror[imc, i] = m[1,:,0]-truemodel[2]
	    srtvar[imc, i] = v[0,:]
	    slopevar = v[2,:]


srtRMSE=np.sqrt(np.mean(srterror**2, axis=0))
srtbias=np.mean(srterror, axis=0)
slopeRMSE=np.sqrt(np.mean(slopeerror**2, axis=0))
slopebias=np.mean(slopeerror, axis=0)

enssrtvar=np.mean(srtvar, axis=0)
ensslopevar=np.mean(slopevar, axis=0)

plt.plot(srtRMSE)
plt.show()