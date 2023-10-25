import sys
import warnings
sys.path.append('../utils')
from basie_class import *

def python_run(nt,listofresponses,mdlsrtprct,mdlmiss,mdlguess,mdlminsnr,mdlmaxsnr,mdlslopemin,mdlslopemax,nx,ns,nh,cs,dh,sl,kp,hg,cf,pm,lg,pp,pf,ts,dp,it,at,la,op,rx):


	modelp = np.array([[mdlsrtprct], [mdlmiss], [mdlguess], [mdlminsnr], [mdlmaxsnr], [mdlslopemin], [mdlslopemax]])
	basiep={'nx':nx,'ns':ns,'nh':nh,'cs':cs,'dh':dh,'sl':sl,'kp':kp,'hg':hg,'cf':cf,'pm':pm,'lg':lg,'pp':pp,'pf':pf,'ts':ts,'dp':dp,'it':it,'at':at,'la':la,'op':op,'rx':rx}

	nmodels=1


	basieest=basie_estimator()
	availsnr=np.linspace(mdlminsnr, mdlmaxsnr, int((mdlmaxsnr-mdlminsnr))+1).T
	[snr1,__,__,__]=basieest.initialise(1, modelp=modelp, availsnr=availsnr, basiep=basiep)

	ii=1
	for i in range(nt):
		response=np.array([[bool(listofresponses[i])]])
		[snr1, ii, m, v, mrob, vrob]=basieest.update(ii, probesnr=snr1, response=response, robust=True)

	return basieest.summary()[-1][-1]
