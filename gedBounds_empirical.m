%% gedBounds method to obtain empirical frequency boundaries based on spatial correlations of eigenvectors.
%
% IMPORTANT NOTE! This is designed to work for empirical EEG data in eeglab
%    format. It is highly likely that you will need to make at least minor
%    modifications to the code for it to work on your data.
%
%  Questions? -> mikexcohen@gmail.com
%

clear

%% load data file

load([ homedir 'proc\' num2str(subIClist{datai,1}) '_proc.mat' ])

%% possible additional data cleaning or preparation...

%% frequency parameters

% frequency resolution and range
numfrex  = 100;
lowfreq  = 2;  % Hz
highfreq = 80; % Hz
frex = logspace(log10(lowfreq),log10(highfreq),numfrex);

% standard deviations for the Gaussian filtering
stds = linspace(2,5,numfrex);

% onset times for epoching resting-state data
onsets = EEG.srate*2:2*EEG.srate:EEG.pnts-EEG.srate*4;
snipn  = 2*EEG.srate;

% initialize some variables
[evals,evecs,maps] = deal(zeros(numfrex,EEG.nbchan));

%% create R covariance matrix

% full R
R = zeros(length(onsets),EEG.nbchan,EEG.nbchan);
for segi=1:length(onsets)
    snipdat = EEG.data(:,onsets(segi):onsets(segi)+snipn);
    snipdat = bsxfun(@minus,snipdat,mean(snipdat,2));
    R(segi,:,:) = snipdat*snipdat'/snipn;
end

% clean R
meanR = squeeze(mean(R));
dists = zeros(1,size(R,1));
for segi=1:size(R,1)
    r = R(segi,:,:);
    dists(segi) = sqrt( sum((r(:)-meanR(:)).^2) );
end
R = squeeze(mean( R(zscore(dists)<3,:,:) ,1));

% regularized R
gamma = .01;
Rr = R*(1-gamma) + eye(EEG.nbchan)*gamma*mean(eig(R));

%% loop over frequencies

for fi=1:numfrex
    
    % filter data
    fdat = filterFGx(EEG.data,EEG.srate,frex(fi),stds(fi));
    
    %%% compute S
    % full S
    S = zeros(length(onsets),EEG.nbchan,EEG.nbchan);
    for segi=1:length(onsets)
        snipdat = fdat(:,onsets(segi):onsets(segi)+snipn);
        snipdat = bsxfun(@minus,snipdat,mean(snipdat,2));
        S(segi,:,:) = snipdat*snipdat'/snipn;
    end
    
    % clean S
    meanS = squeeze(mean(S));
    dists = zeros(1,size(S,1));
    for segi=1:size(S,1)
        s = S(segi,:,:);
        dists(segi) = sqrt( sum((s(:)-meanS(:)).^2) );
    end
    S = squeeze(mean( S(zscore(dists)<3,:,:) ,1));
    
    % global variance normalize
    S = S / (std(S(:))/std(R(:)));
    
    
    % GED
    [W,L] = eig(S,Rr);
    [evals(fi,:),sidx] = sort(diag(L),'descend');
    W = W(:,sidx);
    
    % store top component map and eigenvector
    maps(fi,:) = W(:,1)'*S;
    evecs(fi,:) = W(:,1);
end

%% correlation matrices for clustering

E = zscore(evecs,[],2);
evecCorMat = (E*E'/(EEG.nbchan-1)).^2;

%% determine the optimal epsilon value

% range of epsilon parameter values
nepsis = 50;
epsis  = linspace(.001,.05,nepsis);
qvec   = nan(nepsis,1);

for epi=1:length(epsis)
    
    % scan
    freqbands = dbscan(evecCorMat,epsis(epi),3,'Distance','Correlation');
    if max(freqbands)<4, continue; end
    
    % compute q
    qtmp = zeros(max(freqbands),1);
    MA = false(size(evecCorMat));
    for i=1:max(freqbands)
        M = false(size(evecCorMat));
        M(freqbands==i,freqbands==i) = 1;
        qtmp(i) = mean(mean(evecCorMat(M))) / mean(mean(evecCorMat(~M)));
        MA = MA+M;
    end
    qvec(epi) = mean(qtmp) + log(mean(MA(:)));
end

% run it again on the best epsilon value
[~,epsiidx] = findpeaks(qvec,'NPeaks',1,'SortStr','descend');
if isempty(epsiidx), epsiidx = round(nepsis/2); end
freqbands = dbscan(evecCorMat,epsis(epsiidx),3,'Distance','Correlation');

% dissolve tiny clusters, and renumber all clusters consecutively
newc = cell(4,1); n=1;
for i=1:max(freqbands)
    cc = bwconncomp(freqbands==i);
    for ci=1:cc.NumObjects
        if length(cc.PixelIdxList{ci})>2
            newc{n} = cc.PixelIdxList{ci};
            n = n+1;
        end
    end
end
freqbands = -ones(size(frex));
for ni=1:n-1
    freqbands(newc{ni}) = ni;
end

%% average correlation coefficient within each cluster

avecorcoef = zeros(max(freqbands),2);
for i=1:max(freqbands)
    submat = evecCorMat(freqbands==i,freqbands==i);
    avecorcoef(i,1) = mean(nonzeros(tril(submat,-1)));
    avecorcoef(i,2) = mean(frex(freqbands==i));
end

%% save outputs

chanlocs = EEG.chanlocs;
save(outfilename,'maps','evecs','evals','frex','evecCorMat','evecCorMat','groupidx','chanlocs','avecorcoef','epsis','qvec','epsiidx')


%% some plotting
%

%% correlation matrix and band boundaries

figure(1), clf, colormap bone

imagesc(1-evecCorMat), hold on
f2u = round(linspace(1,length(frex),10));
set(gca,'clim',[.2 1],'xtick',f2u,'xticklabel',round(frex(f2u),1),'ytick',f2u,'yticklabel',round(frex(f2u),1))
axis square, axis xy

for i=1:max(freqbands)
    
    tbnds = frex(freqbands==i);
    tbnds = dsearchn(frex',tbnds([1 end])');
    
    % box
    plot(tbnds,[1 1]*tbnds(1),'m','linew',2)
    plot(tbnds,[1 1]*tbnds(2),'m','linew',2)
    plot([1 1]*tbnds(1),tbnds,'m','linew',2)
    plot([1 1]*tbnds(2),tbnds,'m','linew',2)
end

%% plot the average maps

figure(2), clf
for i=1:max(freqbands)
    subplot(3,3,i)
    m = pca(maps(freqbands==i,:));
    topoplotIndie(m(:,1),chanlocs,'numcontour',0,'electrodes','off','plotrad',.6);
    title([ num2str(round(mean(frex(freqbands==i)),2)) ' Hz' ])
end

%%

