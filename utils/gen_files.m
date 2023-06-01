function out=gen_files(fs,speech, rir, noise, snr)

% read speech and resample
[speechsig, fsspeech] = v_readwav(speech);
speechsig=resample(origsig,fs,fsspeech);

% read RIR and resample
[RIR, fsRIR] = v_readwav(rir);
nchans=size(RIR,2);
RIR=resample(RIR,fs,fsRIR);

% read noise and resample
[noisesig, fsnoise] = v_readwav(noise);
noisesig=resample(noisesig, fs, fsnoise);

% generate reverberant speech
for ichan=1:nchans
    reverbspeech(:,ichan)=filter(RIR(:,ichan),1,speechsig);
end

out=zeros(size(reverbspeech));
nchansnoise=size(noise,2);
% if multichannel noise
if nchansnoise==nchans
    for ichan=1:nchans
        out(:,ichan)=v_addnoise(reverbspeech(:,ichan), fs, snr, 'doAEpk', noisesig(:,ichan), fs);
    end
% otherwise
else
            x = zeros(1.5*length(reverbspeech)+nfft, nchans);
            tmp = 1.5*length(reverbspeech);
            for ichan=1:nchans
                x(:,ichan) = noisesig(mod((ichan-1)*tmp+1:ichan*tmp, length(noisesig)));
                out(:,ichan)=v_addnoise(reverbspeech(:,ichan), fs, snr, 'doAEpk', x(:,ichan), fs);
            end
end



%
% addpath(genpath('~/Documents/MATLAB/sap-voicebox'))
%
% rirpath='~/OneDrive - Imperial College London/Data/External/Ace/Single';
% sentencepath='~/OneDrive - Imperial College London/Data/External/IEEE sentences/Male';
% babblepath='~/OneDrive - Imperial College London/Data/External/NatoNoise0';
% n_type='babble';
% devices=[1,1,1];
% snr=[-20:1:20];
%
% nfiles=length(snr);
%
%
% fs=8000;
% sentdir = dir(fullfile(sentencepath, '*ieee*.wav'));
% sentnames = {sentdir.name};
% nfft = 2*(round(0.016*fs)-1);
% % load sentence s
% for iFile=1:nfiles
%     [origsig, fssig] = v_readwav(fullfile(sentencepath, sentnames{iFile}));
%     origsig = resample(origsig, fs, fssig);
%
%     [rir, fsrir]=v_readwav(fullfile(rirpath, 'Office_1/1',dir(fullfile(rirpath, "Office_1/1", '*RIR*.wav')).name));
%     rir=resample(rir, fs, fsrir);
%
%     sig=filter(rir,1, origsig);
%
%     switch n_type
%         case 'speech-shaped'
%             x = zeros(length(sig)+nfft, size(devices,1));
%             for iArr=1:size(devices,1)
%                 x(:,iArr)=v_stdspectrum(11,'t',fs,length(sig)+nfft); % generate n samples of speech-shaped noise
%             end
%         case 'white'
%             x = randn(length(sig)+nfft, size(devices,1));
%         case 'babble'
%             [babble,fsb] = v_readwav(fullfile(babblepath, 'babble.wav'));
%             if fsb~=fssig
%                 babble = resample(babble, fs, fsb);
%             end
%             x = zeros(length(sig)+nfft, size(devices,1));
%             tmp = length(sig)+nfft;
%             for iArr=1:size(devices,1)
%                 x(:,iArr) = babble((iArr-1)*tmp+1:iArr*tmp);
%             end
%         otherwise
%             error('unknown noise')
%     end
%     snr(iFile)
%     out = v_addnoise(sig(:,1), fs, snr(iFile), 'doAEpk', x(:,1), fs);
%
% end
%
%
