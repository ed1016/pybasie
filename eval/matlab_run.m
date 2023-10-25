function endresults=matlab_run(nt,listofresponses,mdlsrtprct,mdlmiss,mdlguess,mdlminsnr,mdlmaxsnr,mdlslopemin,mdlslopemax,nx,ns,nh,cs,dh,sl,kp,hg,cf,pm,lg,pp,pf,ts,dp,it,at,la,op,rx)
addpath(genpath('~/Documents/MATLAB'))
clear params

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     define test parameters      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% listofresponses=repmat([1 0 0 0 0 0 1 1 1 1], 1,10);
listofresponses=cell2mat(listofresponses);
nt=double(nt);
mdlsrtprct=double(mdlsrtprct);
mdlmiss=double(mdlmiss);
mdlguess=double(mdlguess);
mdlminsnr=double(mdlminsnr);
mdlmaxsnr=double(mdlmaxsnr);
mdlslopemin=double(mdlslopemin);
mdlslopemax=double(mdlslopemax);

q.nx=double(nx);  %number of SNR values in pdf [40]
q.ns=double(ns);  %number of slope values in pdf [21]
q.nh=double(nh);  %number of probe SNR values to evaluate [30]
q.cs=double(cs);   %weighting of slope relative to SRT in cost function [1]
q.dh=double(dh); %minimum step size in dB for probe SNRs [0.2]
q.sl=double(sl); %min slope at threshold (must be >0) [0.005]
q.kp=double(kp); %number of std deviations of the pdfs to keep [4]
q.hg=double(hg); %amount to grow expected gains in ni trials [1.3]
q.cf=double(cf); %cost function: 1=variance, 2=v_entropy [2]
q.pm=pm; %psychometric model: 1=logistic, 2=cumulative gaussian [1]
q.lg=lg; %use log slope in pdf: 0=no, 1=yes [1]
q.pp=double(pp); %Number of prior standard deviations in initial semi-range [1]
q.pf=double(pf); %Probability floor (integrated over entire grid) [0.0001]
q.ts=double(ts); %Number of std devs to explore [2]
q.dp=double(dp); %Maximum probe SNR shift (dB) [10]
q.it=double(it); %Grid interpolation threshold [0.5]
q.at=double(at); %Axis change threshold [0.1]
q.la=double(la); %Look 2-ahead when choosing probe [1]
q.op=double(op); %Outlier probability [0.01]
q.rx=double(rx); %Minimum range factor per iteration [0.5]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     define model parameters     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% truemodel = [gtsrtprct gtsrt gtslp gtmiss gtguess gtfct]';                  % ground truth model
modelp = [mdlsrtprct mdlmiss mdlguess mdlminsnr mdlmaxsnr mdlslopemin mdlslopemax];
availablesnrs=mdlminsnr:mdlmaxsnr;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Now perform the tests       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
snr=v_psycest(-1,modelp.',q, availablesnrs);              % initialize the model (only estimating a single model)
ii=1;

for i=1:nt
    response = listofresponses(i);
    [snr,ii, m, v, mrob, vrob]=v_psycest(ii,snr,response);                % supply the response, update the model and find next probe snr
end

[p,q,msr]=v_psycest(0);                             % output model parameters and record of trial results
endresults=msr(end,:);
%msr
end
