addpath(genpath('~/Documents/MATLAB/sap-voicebox'))

rirpath='~/OneDrive - Imperial College London/Data/External/Ace/Single';%'~/OneDrive - Imperial College London/Data/External/OpenAIR/saint-lawrence-church-molenbeek-wersbeek-belgium/stereo'%'~/OneDrive - Imperial College London/Data/External/Ace/Single';
sentencepath='~/OneDrive - Imperial College London/Data/External/Hurricane_high_quality/quiet_mrt';
babblepath='~/OneDrive - Imperial College London/Data/External/Ace/Single';
n_type='babble';
devices=[1,1,1]; % can add other devices later
snr=[5:5:20];
reverbid='Office_1';%'%ir_church_saint-laurentius_molenbeek_bekkevoort_belgium.wav'; %'Office_1'; % 'Meeting_Room_1' 'Building_Lobby
reverboutname='office';
% nfiles=length(snr);
nsnr=length(snr);

fs=48000;
sentdir = dir(fullfile(sentencepath, '*mrt*.wav'));
sentnames = sort({sentdir.name});
nfiles=length(sentnames);
mkdir(sprintf('data/mrt_hq/%s', reverboutname))

%%
% sentnames= sentnames(randperm(length(sentnames)));
nfft = 2*(round(0.016*fs)-1);
% load sentence s
for iFile=1:49
    [origsig, fssig] = v_readwav(fullfile(sentencepath, sentnames{iFile}));
    origsig = resample(origsig, fs, fssig);

    if ~strcmp(reverbid, 'anechoic')
        if ~contains(rirpath,'OpenAIR')
            [rir, fsrir]=v_readwav(fullfile(rirpath,reverbid, '1',dir(fullfile(rirpath, reverbid, "1", '*RIR*.wav')).name));
            rir=resample(rir, fs, fsrir);
        else
            [rir, fsrir]=v_readwav(fullfile(rirpath,reverbid));
            rir=resample(rir(:,1), fs, fsrir);
        end
        sig=filter(rir,1, origsig);
    else
        sig = origsig;
    end

    switch n_type
        case 'speech-shaped'
            x = zeros(length(sig)+nfft, size(devices,1));
            for iArr=1:size(devices,1)
                x(:,iArr)=v_stdspectrum(11,'t',fs,length(sig)+nfft); % generate n samples of speech-shaped noise
            end
        case 'white'
            x = randn(length(sig)+nfft, size(devices,1));
        case 'babble'
            [babble,fsb] = v_readwav(fullfile(babblepath, reverbid, '1', dir(fullfile(rirpath, reverbid, '1', '*Babble.wav')).name),'p');
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
        v_writewav(out./10, fs, sprintf('data/mrt_hq/%s/practice_%s_reverb_%s_snr_%i_db.wav',reverboutname,extractBefore(sentnames{iFile}, '.wav'), ...
            reverboutname, snr(iSnr)))
    end
end


