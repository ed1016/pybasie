%% Description
% This script can be used to generate the data used in the experiments.
% Requirements:
%   voicebox: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%       for signal processing
%   Hurricane sentences: https://doi.org/10.7488/ds/2482
%       for the stimuli MRT sentences
%   Ace corpus: http://www.ee.ic.ac.uk/naylor/ACEweb/index.html
%       for Room Impulse Respones (RIR)
%   Babble file: http://svr-www.eng.cam.ac.uk/comp.speech/Section1/Data/noisex.html
%       for noise generation. Could be any noise file

voiceboxpath=uigetdir('~', 'Select path to voicebox folder');
addpath(genpath(voiceboxpath))

sentencepath=uigetdir('~', 'Select path to Hurricane sentences folder');
sentencepath=fullfile(sentencepath, 'quiet_mrt');

rirpath=uigetdir('~', 'Select path to Ace Corpus folder');
rirpath=fullfile(rirpath, 'Single');

noisepath=uigetdir('~', 'Select path to noise file (.wav)');

%% parameters
n_type='file';

snr=-20:1:20; % list of SNR values to generate
reverbid='Office_1'; %'anechoic' 'Office_1' 'Meeting_Room_1' 'Building_Lobby'
reverboutname='office';

nsnr=length(snr);

fs=48000;
sentdir = dir(fullfile(sentencepath, '*mrt*.wav'));
sentnames = sort({sentdir.name});
nfiles=length(sentnames);

outfolder='./data/mrt_hq';
mkdir(sprintf('%s/%s',outfolder, reverboutname))
%% generate the data and save
nfft = 2*(round(0.016*fs)-1);

excludedfiles=[50,5,105,255,8,158,208,10,60,160,62,16,266,168,269,20,221,175,26,129,134,86,37,187,142,94,48];

% load rir
if ~contains(reverbid,'anechoic')
    [rir, fsrir]=v_readwav(fullfile(rirpath,reverbid, '1',dir(fullfile(rirpath, reverbid, "1", '*RIR*.wav')).name));
    for ichan=1:size(rir,2)
        rir(:,ichan)=resample(rir(:,ichan), fs, fsrir);
    end
else
    rir=1;
end

% load noise
switch n_type
    case 'speech-shaped'
        x = zeros(fs*1, size(rir,2));
        for iArr=1:size(devices,1)
            x(:,iArr)=v_stdspectrum(11,'t',fs,fs*1); % generate n samples of speech-shaped noise
        end
    case 'white'
        x = randn(fs*1, size(rir,2)));
    case 'file'
        [noise,fsb] = v_readwav(noisepath,'p');
        if fsb~=fs
            noise = resample(noise, fs, fsb);
        end
        x = zeros(fs*15, size(rir,2));
        tmp = fs*15;

        wrapN = @(x, N) (1 + mod(x-1, N));
        N = length(noise);

        for iArr=1:size(rir,2)
            x(:,iArr) = filter(rir(:,iArr), 1,noise(wrapN(10*fs+(iArr-1)*tmp+1):wrapN(10*fs+iArr*tmp)));
        end
    otherwise
        error('unknown noise')
end


% generate the practice files
snr=[10:5:20];
for iFile=1:2
    [origsig, fssig] = v_readwav(fullfile(sentencepath, sentnames{iFile}));
    origsig = resample(origsig, fs, fssig);

    if ~strcmp(reverbid, 'anechoic')
        for ichan=1:size(rir,2)
            sig(:,ichan)=filter(rir(:,ichan),1, origsig);
        end
    else
        sig = origsig;
    end

    for iSnr=1:length(snr)
        for ichan=1:size(rir,2)
            out(:,ichan) = v_addnoise(sig(:,ichan), fs, snr(iSnr), 'doAEpk', x(:,ichan), fs);
        end
        v_writewav(out./10, fs, sprintf('%s/%s/practice_%s_reverb_%s_snr_%i_db.wav',outfolder,reverboutname,extractBefore(sentnames{iFile}, '.wav'), ...
            reverboutname, snr(iSnr)))
    end
end

% generate main sentences
snr=[-20:1:20];
% load sentence s
for iFile=3:nfiles
    fprintf('file %i/%i\n', iFile, nfiles)
    if ~ismember(iFile, excludedfiles)
        [origsig, fssig] = v_readwav(fullfile(sentencepath, sentnames{iFile}));
        origsig = resample(origsig, fs, fssig);

        if ~strcmp(reverbid, 'anechoic')
            for ichan=1:size(rir,2)
                sig(:,ichan)=filter(rir(:,ichan),1, origsig);
            end
        else
            sig = origsig;
        end

        %     snr(iFile)
        for iSnr=1:length(snr)
            for ichan=1:size(rir,2)
                out(:,ichan) = v_addnoise(sig(:,ichan), fs, snr(iSnr), 'doAEpk', x(:,ichan), fs);
            end
            v_writewav(out./10, fs, sprintf('%s/%s/%s_reverb_%s_snr_%i_db.wav',outfolder,reverboutname,extractBefore(sentnames{iFile}, '.wav'), ...
                reverboutname, snr(iSnr)))
        end
    end
end

% generate max loudness
snr=-20;
if strcmp(reverbid,'anechoic')
    for iFile=50
        [origsig, fssig] = v_readwav(fullfile(sentencepath, sentnames{iFile}));
        origsig = resample(origsig, fs, fssig);

        sig = origsig;
        %     snr(iFile)
        for iSnr=1:length(snr)
            out = v_addnoise(sig(:,1), fs, snr(iSnr), 'doAEpk', x(:,1), fs);
            v_writewav(sig./10, fs, sprintf('%s/clearspeech.wav', outfolder))
            v_writewav(out./10, fs, sprintf('%s/maxloudness.wav', outfolder))
        end
    end
end



