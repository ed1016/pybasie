import matlab.engine
from python_run import python_run
import numpy as np
import random
import warnings
import yaml
import io
import contextlib
import sys

config_file=sys.argv[1]
print(config_file)
with open(config_file, 'r') as file:
	configvars=yaml.safe_load(file)

N=configvars['trial_structure']['Nrealisations']
nt=configvars['trial_structure']['Ntrials']

mdlsrtprct=configvars['model_parameters']['srt_percentage']
mdlmiss=configvars['model_parameters']['miss_rate']
mdlguess=configvars['model_parameters']['guess_rate']
mdlminsnr=configvars['model_parameters']['min_snr']
mdlmaxsnr=configvars['model_parameters']['max_snr']
mdlslopemin=configvars['model_parameters']['min_slope']
mdlslopemax=configvars['model_parameters']['max_slope']

nx=configvars['basie_parameters']['nx']
ns=configvars['basie_parameters']['ns']
nh=configvars['basie_parameters']['nh']
cs=configvars['basie_parameters']['cs']
dh=configvars['basie_parameters']['dh']
sl=configvars['basie_parameters']['sl']
kp=configvars['basie_parameters']['kp']
hg=eval(configvars['basie_parameters']['hg'])
cf=configvars['basie_parameters']['cf']
pm=configvars['basie_parameters']['pm']
lg=configvars['basie_parameters']['lg']
pp=configvars['basie_parameters']['pp']
pf=configvars['basie_parameters']['pf']
ts=configvars['basie_parameters']['ts']
dp=configvars['basie_parameters']['dp']
it=configvars['basie_parameters']['it']
at=configvars['basie_parameters']['at']
la=configvars['basie_parameters']['la']
op=configvars['basie_parameters']['op']
rx=configvars['basie_parameters']['rx']

results=[]
eng=matlab.engine.start_matlab()

for j in range(N):
	listofresponses=[]
	for i in range(nt):
		listofresponses.append(random.randint(0,1))

	matresult=eng.matlab_run(nt,listofresponses,mdlsrtprct,mdlmiss,mdlguess,mdlminsnr,mdlmaxsnr,mdlslopemin,mdlslopemax,
		nx,ns,nh,cs,dh,sl,kp,hg,cf,pm,lg,pp,pf,ts,dp,it,at,la,op,rx,nargout=1,stdout=io.StringIO())
	with contextlib.redirect_stdout(io.StringIO()):
		with warnings.catch_warnings(record=True) as w:
			pyresult=python_run(nt,listofresponses,mdlsrtprct,mdlmiss,mdlguess,mdlminsnr,mdlmaxsnr,mdlslopemin,mdlslopemax,
				nx,ns,nh,cs,dh,sl,kp,hg,cf,pm,lg,pp,pf,ts,dp,it,at,la,op,rx)

	w = list(filter(lambda i: issubclass(i.category, UserWarning), w))
	if len(w):
		warningmsg=w[0].message
	else:
		warningmsg=''

	matresult=[d for d in matresult[0]]

	results.append({'matresults': matresult, 'pyresult': pyresult, 'difference': sum(abs(matresult-pyresult)), 'warning':warningmsg, 'conditionlist':listofresponses})
eng.quit()

print(sorted(results, key=lambda k: k['difference']))



