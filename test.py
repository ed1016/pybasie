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
modelp = np.array([[0.5, 0.5], [0.01, 0.04], [0, 0.1], [-20, -20], [20,20], [0,0], [0.5,0.5]])
# modelp = np.array([[0.5], [0.01], [0], [-20], [20], [0], [0.5]])

nmodels=modelp.shape[1]

availsnr=np.linspace(-10, 10, 21).T

snrlist=[]
for i in range(nmodels):
    snrlist.append(np.linspace(-10, 10, 21).T)

snrlist=np.asarray(snrlist)
print(snrlist)
[snr, __, __, __]=v_psycest(-2, modelp, None, snrlist,1)
truemodel=np.array([[0.5], [0.0], [0.1], [0.01], [0], [1]])

nt = 10
listofresponses = np.array([[1], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [0], [1], [1], [0],[1],[0]])
ii=1
for i in range(nt):
    [response, __] = v_psychofunc('r',truemodel,np.array([snr]));
#     response = np.array([[bool(listofresponses[i])]])
    print('snr: ',snr, 'response: ',response, 'model: ', ii)
    [snr, ii, m, v] = v_psycest(ii, snr, response);
    plt.show()