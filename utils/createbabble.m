addpath(genpath('~/Documents/MATLAB/sap-voicebox'))

vctkpath='~/OneDrive - Imperial College London/Data/External/VCTK/wav48_silence_trimmed';
nspeakers=12;

speakerdata=readtable(strrep(vctkpath, 'wav48_silence_trimmed', 'speaker-info.txt'), 'ExpectedNumVariables',5);


listfemale=cellfun(@(x) string(x), table2cell(speakerdata(cellfun(@(x) x=='F', speakerdata.Var3),1)));
listmale=cellfun(@(x) string(x), table2cell(speakerdata(cellfun(@(x) x=='M', speakerdata.Var3),1)));

speakerlist=[listfemale(randperm(length(listfemale), int32(floor(nspeakers/2))));
    listmale(randperm(length(listmale), int32(ceil(nspeakers/2))))];

babble=0;
minlength=0;
for i=1:nspeakers
    speaker=[];
    for j=1:30
        filelist=sort({dir(fullfile(vctkpath, speakerlist(i), "*mic2.flac")).name});
        [speech, fs]=audioread(char(fullfile(vctkpath, speakerlist(i), filelist{mod((i-1)*30+j, length(filelist)-1)})));
        speaker=[speaker;speech(0.3*fs:end-0.2*fs)];
    end
    if i==1
        babble=speaker;
    else
        try
            babble=babble+speaker(1:length(babble));
        catch
            babble=babble(1:length(speaker))+speaker;
            minlength=length(speaker);
        end
    end
end

babble=babble(5*fs:minlength);
v_writewav(babble,fs, 'data/babble.wav')



