%% gedBounds method to obtain empirical frequency boundaries based on spatial correlations of eigenvectors.
% You can run the entire script to produce figures, and modify the
% simulation as you like.
% 
%  Questions? -> mikexcohen@gmail.com
% 

clear

%% preliminaries

% mat file containing EEG, leadfield and channel locations
load emptyEEG

%% simulation parameters

% boundaries of frequency bands
dipfrex{1} = [  4  7 ];
dipfrex{2} = [  9 11 ];
dipfrex{3} = [ 11 13 ];

% indices of dipole locations
dipoleLoc(1) = 1350;
dipoleLoc(2) =   94;
dipoleLoc(3) =  205;

%% create the dipole time series

% create narrowband nonstationary time series
hz = linspace(0,EEG.srate,EEG.pnts);

% brain of white noise
dipdat = randn(size(EEG.lf.Gain,3),EEG.pnts)*3;

%% optional: brain of pink noise (1/f)
% The method does not depend on the color of the background noise spectrum.
% This is illustrated by using pink noise instead of white noise.

% ed = 2000; % exponential decay parameter
% for di=1:size(EEG.lf.Gain,3)
%     as = rand(1,EEG.pnts) .* exp(-(0:EEG.pnts-1)/ed);
%     data = real(ifft(as .* exp(1i*2*pi*rand(size(as)))));
%     dipdat(di,:) = (data-mean(data))/std(data);
% end

%%

figure(1), clf


for di=1:length(dipoleLoc)
    
    % FIR1 filtered noise
    ford  = round( 30*(EEG.srate/3) );
    fkern = fir1(ford,dipfrex{di}./(EEG.srate/2));
    filtdat = filter(fkern,1,randn(EEG.pnts,1)) * 200;
    
    % put in dipole time series
    dipdat(dipoleLoc(di),:) = filtdat;
    
    subplot(3,1,di), hold on
    plot(linspace(0,EEG.srate,length(fkern)),abs(fft(fkern)),'ko-')
    plot([0 dipfrex{di}(1) dipfrex{di} dipfrex{di}(2) EEG.srate/2],[0 0 1 1 0 0],'r')
    sigX = abs(fft(filtdat));
    plot(linspace(0,EEG.srate,EEG.pnts),sigX/max(sigX),'m')
    set(gca,'xlim',[0 20])
    xlabel('Frequency (Hz)'), ylabel('Amplitude')
    title([ 'Spectrum, dipole ' num2str(di) ])
end

EEG.data = squeeze(EEG.lf.Gain(:,1,:))*dipdat;

%% now for the analysis


%% frex params

numfrex = 80;
frex = logspace(log10(2),log10(30),numfrex);
stds = linspace(1,1,numfrex);

onsets = EEG.srate*2:2*EEG.srate:EEG.pnts-EEG.srate*2;
snipn = 2*EEG.srate;

% initialize
[evals,evecs,maps] = deal(zeros(numfrex,EEG.nbchan));

%% create R

% full R
R = zeros(EEG.nbchan,EEG.nbchan);
for segi=1:length(onsets)
    snipdat = EEG.data(:,onsets(segi):onsets(segi)+snipn);
    snipdat = bsxfun(@minus,snipdat,mean(snipdat,2));
    R = R + snipdat*snipdat'/snipn;
end
R = R/segi;

% regularized R
gamma = .01;
Rr = R*(1-gamma) + eye(EEG.nbchan)*gamma*mean(eig(R));

%% loop over frequencies

for fi=1:numfrex
    
    % filter data
    fdat = filterFGx(EEG.data,EEG.srate,frex(fi),stds(fi));
    
    %%% compute S
    % full S
    S = zeros(EEG.nbchan,EEG.nbchan);
    for segi=1:length(onsets)
        snipdat = fdat(:,onsets(segi):onsets(segi)+snipn);
        snipdat = bsxfun(@minus,snipdat,mean(snipdat,2));
        S = S + snipdat*snipdat'/snipn;
    end
    % global variance normalize (optional; this scales the eigenspectrum)
    S = S / (std(S(:))/std(R(:)));
    
    % GED
    [W,L] = eig(S,Rr);
    [evals(fi,:),sidx] = sort(diag(L),'descend');
    W = W(:,sidx);
    
    % store top component map and eigenvector
    maps(fi,:) = W(:,1)'*S;
    evecs(fi,:) = W(:,1);
    
end

%% correlation matrix for clustering

E = zscore(evecs,[],2);
evecCorMat = (E*E'/(EEG.nbchan-1)).^2;

%%

figure(2), clf, set(gcf,'color','w')
contourf(frex,frex,1-evecCorMat,40,'linecolor','none'), hold on
set(gca,'clim',[0 .8],'xscale','log','yscale','log','xtick',round(logspace(log10(1),log10(numfrex),14),1),'ytick',round(logspace(log10(1),log10(numfrex),14),1),'fontsize',15)
xlabel('Frequency (Hz)'), ylabel('Frequency (Hz)') 
axis square, axis xy, colormap bone
title('Eigenvectors correlation matrix')

% box
for i=1:length(dipfrex)
    tbnds = dipfrex{i}; dsearchn(frex',dipfrex{i}');
    plot(tbnds,[1 1]*tbnds(1),'r--','linew',2)
    plot(tbnds,[1 1]*tbnds(2),'r--','linew',2)
    plot([1 1]*tbnds(1),tbnds,'r--','linew',2)
    plot([1 1]*tbnds(2),tbnds,'r--','linew',2)
end


e = evals(:,1);
e = e-min(e); e=e./max(e);
plot(frex,1.5*e+frex(1),'b','linew',2)

h = colorbar;
set(h,'ticklabels',{'1','.8','.6','.4','.2'},'Ticks',0:.2:.8,'fontsize',15)

%% determine the optimal epsilon value

% range of epsilon parameters
nepsis  = 50;
epsis   = linspace(.0001,.05,nepsis);
qvec    = nan(nepsis,1);

for thi=1:length(epsis)
    
    % scan
    freqbands = dbscan(evecCorMat,epsis(thi),3,'Distance','Correlation');
    
    % compute q
    qtmp = zeros(max(freqbands),1);
    MA = false(size(evecCorMat));
    for i=1:max(freqbands)
        M = false(size(evecCorMat));
        M(freqbands==i,freqbands==i) = 1;
        qtmp(i) = mean(mean(evecCorMat(M))) / mean(mean(evecCorMat(~M)));
        MA = MA+M;
    end
    qvec(thi) = mean(qtmp) + log(mean(MA(:)));
end

% run it again on the best epsilon value
[~,epsiidx] = findpeaks(qvec,'NPeaks',1,'SortStr','descend');
if isempty(epsiidx), epsiidx = round(nepsis/2); end
freqbands = dbscan(evecCorMat,epsis(epsiidx),3,'Distance','Correlation');

%% draw empirical bounds on correlation map

for i=1:max(freqbands)
    
    tbnds = frex(freqbands==i);
    tbnds = tbnds([1 end]);
    
    % box
    plot(tbnds,[1 1]*tbnds(1),'m','linew',2)
    plot(tbnds,[1 1]*tbnds(2),'m','linew',2)
    plot([1 1]*tbnds(1),tbnds,'m','linew',2)
    plot([1 1]*tbnds(2),tbnds,'m','linew',2)
end

%% plot the average maps

figure(3), clf, set(gcf,'color','w')
for i=1:3
    
    groundTruth = -squeeze(EEG.lf.Gain(:,1,dipoleLoc(i)));
    
    % ground truth
    subplot(3,3,i)
    topoplotIndie(groundTruth,EEG.chanlocs,'numcontour',0,'electrodes','off');
    title([ 'GT: ' num2str(mean(dipfrex{i})) ' Hz' ])
    
    % maps
    subplot(3,3,i+3)
    m = pca(maps(freqbands==i,:));
    m = m(:,1)*sign(corr(m(:,1),groundTruth));
    topoplotIndie(m,EEG.chanlocs,'numcontour',0,'electrodes','off');
    title([ 'Maps: ' num2str(round(mean(frex(freqbands==i)),2)) ' Hz' ])
    
    % eigenvectors
    subplot(3,3,i+3+3)
    m = pca(evecs(freqbands==i,:));
    m = m(:,1)*sign(corr(m(:,1),groundTruth));
    topoplotIndie(m,EEG.chanlocs,'numcontour',0,'electrodes','off');
    title([ 'E-vecs: ' num2str(round(mean(frex(freqbands==i)),2)) ' Hz' ])
end

set(findall(gcf,'type','axes'),'fontsize',12)

%% done.
