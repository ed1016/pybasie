addpath(genpath('~/Documents/MATLAB/sap-voicebox'))

rirpath='~/OneDrive - Imperial College London/Data/External/Ace/Single';
sentencepath='~/OneDrive - Imperial College London/Data/External/Hurricane/quiet_mrt';
babblepath='~/OneDrive - Imperial College London/Data/External/NatoNoise0';
n_type='babble';
devices=[1,1,1]; % can add other devices later
snr=[5:5:20];
reverb='Office_1'; %'Office_1'; % 'Meeting_Room_1' 'Building_Lobby
% nfiles=length(snr);
nsnr=length(snr);

fs=16000;
sentdir = dir(fullfile(sentencepath, '*mrt*.wav'));
sentnames = {sentdir.name};
nfiles=length(sentnames);
%%
% sentnames= sentnames(randperm(length(sentnames)));
nfft = 2*(round(0.016*fs)-1);
% load sentence s
for iFile=1:nfiles
    [origsig, fssig] = v_readwav(fullfile(sentencepath, sentnames{iFile}));
    origsig = resample(origsig, fs, fssig);

    [rir, fsrir]=v_readwav(fullfile(rirpath,reverb, '1',dir(fullfile(rirpath, reverb, "1", '*RIR*.wav')).name));
    rir=resample(rir, fs, fsrir);

    sig=filter(rir,1, origsig);

    switch n_type
        case 'speech-shaped'
            x = zeros(length(sig)+nfft, size(devices,1));
            for iArr=1:size(devices,1)
                x(:,iArr)=v_stdspectrum(11,'t',fs,length(sig)+nfft); % generate n samples of speech-shaped noise
            end
        case 'white'
            x = randn(length(sig)+nfft, size(devices,1));
        case 'babble'
            [babble,fsb] = v_readwav(fullfile(babblepath, 'babble.wav'));
            if fsb~=fssig
                babble = resample(babble, fs, fsb);
            end
            x = zeros(length(sig)+nfft, size(devices,1));
            tmp = length(sig)+nfft;
            for iArr=1:size(devices,1)
                x(:,iArr) = babble((iArr-1)*tmp+1:iArr*tmp);
            end
        otherwise
            error('unknown noise')
    end
%     snr(iFile)
    for iSnr=1:nsnr
        out = v_addnoise(sig(:,1), fs, snr(iSnr), 'doAEpk', x(:,1), fs);
        v_writewav(out, fs, sprintf('audio/mrt/practice_%s_reverb_%s_snr_%i_db.wav',extractBefore(sentnames{iFile}, '.wav'), ...
            reverb, snr(iSnr)), 'g')
    end
end


