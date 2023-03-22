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


modelp = np.array([[0.5], [0.04], [0.2], [-20], [20], [0.01], [0.5]])

# nmodels=modelp.shape[1]
nmodels=3

availsnr=np.linspace(-20, 20, 41).T

snrlist=[]
for i in range(nmodels):
    snrlist.append(np.linspace(-10, 10, 21).T)

snrlist=np.asarray(snrlist)




[snr, __, __, __]=v_psycest(-nmodels,  np.repeat(modelp, nmodels, axis=1), None, snrlist,1)
truemodel=np.array([[0.5], [0.0], [0.1], [0.01], [0], [1]])

nt = 100
listofresponses = np.array([[1], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [0], [1], [1], [0],[1],[0]])
ii=1

vartoplot=[]
meantoplot=[]
for i in range(nt):
    [response, __] = v_psychofunc('r',truemodel,np.array([snr]));
#     response = np.array([[bool(listofresponses[i])]])
    # print('snr: ',snr, 'response: ',response, 'model: ', ii)
    [snr, ii, m, v] = v_psycest(ii, snr, response);
    vartoplot.append(v[0,:])
    meantoplot.append(m[0,:,0])

fig, axs = plt.subplots(1,2)
axs[0].plot(range(nt), np.sqrt(vartoplot))
plt.grid()
axs[1].plot(range(nt), meantoplot)
plt.grid()
# plt.ylim(0,1)
plt.show()
