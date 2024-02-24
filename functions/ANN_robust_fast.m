function [T_gf,T_pred,T_ext,unc,c,ANNstats] = ANN_robust_fast(X,T,arch,clmax,nmin,clund,Next,Ninit,f,extout)
% Gap filling with representative training data selection, optimization of 
% ANN architecture complexity, and uncertainty estimation of gap-filled 
% estimates. 
% Representative data selection: Data used to train, test, and validate
%   the ANN are proportional representative of the instances of similar 
%   conditions in the output series. 
% ANN complexity optimization: Varying levels of ANN architecture 
%   complexity are tested, with the selected architecture being the 
%   simplest to achieve robust estimation. 
% Uncertainty estimatation: Bootstrapping of the data used to train, test, 
%   and validate the optimum ANN is used to obtain uncertainty estimates of
%   the output.
% 
% -------------- Inputs -----------------
% X: Column matrix of explanatory variables. Each new column is a different
%    explanatory variable. It is important that there be no missing data in
%    this matrix.
% T: Column matrix of the dataset to be gap-filled. If the dataset contains
%    multiple columns (variables), the input-output relationships 
%    determined by the neural network will be a compromise between all
%    output variables. Therefore, it is important that multiple outputs be
%    logically related in some way.
% arch: Specifies the hidden layer ANN architectures to test (optional). 
%    This is a cell column matrix where each row is a vector of the hidden 
%    layer nodes. Each successive row should be more complex than the 
%    previous. Enter [] to test the 4 default architectures: 
%           [N inputs*1]
%           [N inputs*1.5]
%           [N inputs*1  N inputs*0.5]
%           [N inputs*1.5  N inputs*0.75]
% clmax: maximum number of clusters to divide the explanatory data into for
%    representative sampling of data fed into the ANN. The code will start
%    with this number of clusters and decrease if necessary.
% nmin: Minimum sample size of target data in each explanatory data
%    cluster. Only clund explanatory data clusters are allowed to be
%    underrepresented below this level.Otherwise, the number of explanatory
%    data clusters is reduced.
% clund: the number of explanatory data clusters allowed to be under-
%    represented (n examples < nmin) 
% Next: The number of random data extractions to perform. From these N
%    extractions, the median outcome is used as the gap-filled estimate, 
%    and the uncertainty is estimated as the interquartile range.
% Ninit: The number of initializations to test for each network in order to
%       avoid local minima.
% f: a 3-element vector of the fraction of the random sample going to the
%    training, validating, and testing subsamples. Ex. [0.5 0.25 0.25]
% extout: binary (0 or 1) indicating whether to output all ANN-predicted 
%         datasets from each data extraction.
%
% -------------- Outputs ----------------
% T_gf: The gap-filled dataset.
% T_pred: The median ANN-predicted dataset
% T_ext: All ANN-predicted datasets from each data extraction during the
%        uncertainty estimation. If T_gf is a single column array, each
%        extraction will be a new column. If T_gf is a 2D matrix, each 
%        extraction will be a new layer in the 3rd dimension. 
%        If extout = 0, this output will be an empty matrix. 
% unc: Interquartile range of all ANN-predicted datasets (from each data 
%      extraction during the uncertainty estimation). This will be a nx2
%      column matrix of [lower upper] interquartile range for each ANN
%      prediction.
% c: the cluster classes of the explanatory data
% ANNstats: A structure containing information about the ANN optimization
%      procedure and the chosen ANNs for each extraction:
%      ANNstats.arch = The tested hidden layer architectures
%      ANNstats.bestai = The index within arch indicating the chosen
%            architecture for each extraction
%      ANNstats.mse_val = m.s.e. of each validation set, size
%            [1,length(arch),Ninit]
%      ANNstats.perc_better = the percentage improvement in m.s.e. for each
%            tested architecture, size [1,length(arch)]
%      ANNstats.bestnet = the final ANN for each extraction
%
% ---------------------------------------
% The methods used in this code were inspired by instruction from Dario
% Papale.
% Copyright 2013, Cove Sturtevant. All rights reserved. 
% Updated September 2014 to include ANNstats output

%% Prep the data

% Scale all inputs/outputs between -1 and 1
X = mapminmax(X'); X = X';
%[X,~] = fixunknowns(X'); % Turn NaNs in predictors to mean value
%X = X(1:2:end,:)';

T_gf = T; % Save original 
[T,PS] = mapminmax(T'); T = T';

% Create several levels of network architecture complexity to test
if isempty(arch)
    arch{1} = ceil(size(X,2)*1);
    arch{2} = ceil(size(X,2)*1.5);
    arch{3} = [ceil(size(X,2)*1) ceil(size(X,2)*0.5)];
    arch{4} = [ceil(size(X,2)*1.5) ceil(size(X,2)*0.75)];
end


%% Cluster explanatory data for representative sampling
% Divide the dataset into natural ranges (of the explanatory data), as many 
% as possible with enough data to train/test/validate ANN 
disp('Creating explanatory data clusters for representative sampling...')
warning('off','stats:kmeans:FailedToConvergeRep')

% Start with max clusters, decreasing from there
cl = clmax;
keepgoing = 1;
while keepgoing
    
    % Find natural clusters
    kopts = statset; 
    kopts.maxiter = 300;
    [c,~] = kmeans(X,cl,'replicates',5);
    
    % Determine how much target data we have in each of these clusters
    nT = NaN(cl,1);
    for cli = 1:cl
        nT(cli,1) = sum(~isnan(sum(T(c==cli,:),2)));
    end
    
    % If more than clund clusters are unrepresented, fall back to a 
    % lower number of clusters. 
    if sum(nT < nmin) > clund && cl > 1
        cl = cl-1;
    else
        keepgoing = 0;
    end
end

% Determine the sample size for random sampling within each cluster
nTs = sort(nT);
if cl < clund+1
    nrs = nTs(end);
else
    nrs = nTs(clund+1);
end

disp(['Using ' num2str(cl) ' clusters.'])

figure(1); clf
scatter((1:size(T,1))',T,3,c,'filled')
ylabel('Rescaled target data')
cb = colorbar;
set(get(cb,'ylabel'),'string','Cluster')
axis tight
title('Environmental data clusters projected on target data')


%% Find best neural network architecture
disp('Finding best neural network architecture. This will take some time...')

mse_val = NaN(length(arch),10);
mse_ts = NaN(length(arch),1);
perc_better = NaN(length(arch),1);

% Representatively sample target examples within each explanatory data 
% cluster and divide into training, validation, and testing indices.
[tri,vali,tsi] = repsample(T,c,nrs,f);


for ai = 1:length(arch)
    disp(['Testing ' num2str(ai) ' of ' num2str(length(arch)) '. Hidden layer architecture: [' num2str(arch{ai}) '] ...'])
    
    % Create the network
    net = fitnet(arch{ai},'trainlm');
    net.trainParam.showWindow = 0;
    net.trainParam.goal = 0;
    net.trainParam.mu = 0.001;
    net.trainParam.mu_inc = 10;

    % Set the training/val/test indices to the representative samples - for
    % some reason the ANN is MUCH faster if you reduce the dataset to just
    % the training, validating, and testing indices.
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = 1:length(tri);
    net.divideParam.valInd = (max(net.divideParam.trainInd)+1):(max(net.divideParam.trainInd)+length(vali));
    net.divideParam.testInd = (max(net.divideParam.valInd)+1):(length(tsi)+max(net.divideParam.valInd));
    %net.divideParam.trainInd = tri';
    %net.divideParam.valInd = vali';
    %net.divideParam.testInd = tsi';

    % Test multiple initializations of the network
    for Initi = 1:Ninit
        % Train the network
        net = train(net,[X(tri,:);X(vali,:);X(tsi,:)]',[T(tri,:);T(vali,:);T(tsi,:)]'); % train it
        %net = train(net,X',T'); % train it
        Ty = net(X')'; % Simulate all outputs
        figure(2); clf; plot((1:size(T))',T,'k',(1:size(T))',Ty,'r',tri,T(tri),'b*'); 
        legend('data','fit','training set'); 
        title(['Hidden layer architecture: [' num2str(arch{ai}) '], Initialization ' num2str(Initi) ' of 10']); pause(0.1) 
        
        % Get mse of validation set (paritially independent)
        mse_val(ai,Initi) = perform(net,T(vali)',Ty(vali)');
        
        % Plot the training and test regressions
        figure(3); clf
        subplot(1,2,1)
        plot([-1 1],[-1 1],'k',T(tri),Ty(tri),'k.'); 
        title(['Training set: R^2 = ' num2str(corr(T(tri),Ty(tri)).^2,2)])
        xlabel('measured'); ylabel('modeled')
        subplot(1,2,2)
        plot([-1 1],[-1 1],'k',T(tsi),Ty(tsi),'k.'); 
        title(['Test set: R^2 = ' num2str(corr(T(tsi),Ty(tsi)).^2,2)])

        % Check if this is the best initialization of this architecture
        % and possibly overall
        if mse_val(ai,Initi) <= nanmin(mse_val(ai,:))
            % Save the best independent (test) set error
            mse_ts(ai) = perform(net,T(tsi)',Ty(tsi)');
            
            % Evaluate if this is the best overall architecture (best of
            % the best test errors)
            if ai == 1
                bestnet = net;
                bestai = 1;
                perc_better(1) = 0;
            else
                % This network must be at least 5% better (w.r.t. the
                % simplest network) than the previous best to be the new 
                % best
                perc_better(ai) = -(mse_ts(ai)-mse_ts(1))/mse_ts(1)*100;
                if (perc_better(ai) - perc_better(bestai)) >= 5
                    bestnet = net;
                    bestai = ai;
                end
            end
        end
        
        % Re-initialize network weights for next pass
        net = init(net); 

    end
end

clear net
net0 = bestnet;
perc_better
disp(['     Best architecture: [' num2str(arch{bestai}) ']'])
close(3)
    
%% Use best architecture to gap-fill and get uncertainty
disp('Architecture found, gap-filling with uncertainty estimation...')

figure(2); clf;
h0 = plot(T_gf,'k','linewidth',2); hold on


% Start with the best architecture, re-train with multiple data extractions
T_ext = NaN(length(T),Next); % initialize
for Nexti = 1:Next
    disp(['Run ' num2str(Nexti) ' of ' num2str(Next)])
    
    % Representatively sample target examples within each explanatory data 
    % cluster and divide into training, validation, and testing indices.
    [tri,vali,tsi] = repsample(T,c,nrs,f);


    % Set the training/val/test indices to the representative samples
    net0.divideFcn = 'divideind';
    net0.divideParam.trainInd = 1:length(tri);
    net0.divideParam.valInd = (max(net0.divideParam.trainInd)+1):(max(net0.divideParam.trainInd)+length(vali));
    net0.divideParam.testInd = (max(net0.divideParam.valInd)+1):(length(tsi)+max(net0.divideParam.valInd));
    %net0.divideParam.trainInd = tri';
    %net0.divideParam.valInd = vali';
    %net0.divideParam.testInd = tsi';
    
    % Train the network
    net = train(net0,[X(tri,:);X(vali,:);X(tsi,:)]',[T(tri,:);T(vali,:);T(tsi,:)]');

    ANNstats.bestnet = net;
    
    Ty = net(X')'; % Simulate all outputs
    T_ext(:,Nexti) = mapminmax('reverse',Ty',PS)';
    
    figure(2); 
    h1 = plot(T_ext(:,Nexti),'color',[0.5 0.5 0.5]);
    legend([h0;h1],'data','run fit')
    title(['Fit ' num2str(Nexti) ' of ' num2str(Next)])
    ylabel('Target data')
    pause(0.1)
    
end

% Save the median output
T_pred = median(T_ext,2); % Entire predicted output
T_gf(isnan(T_gf)) = T_pred(isnan(T_gf)); % Gap-filled output

figure(2); hold on
h2 = plot(T_pred,'color',[1 0 0]);
legend([h0;h1;h2],'data','run fit','median fit')
title('Uncertainty of ANN fit')

% Get interquartile range of predicted datasets
unc = [prctile(T_ext,25,2) prctile(T_ext,75,2)];

% If not output the results from all extractions, clear the output
if ~extout
    T_ext = NaN(size(T_gf));
end

% Output the ANN stats from the optimization procedure
ANNstats.arch = arch;
ANNstats.bestai = bestai;
ANNstats.mse_val = mse_val;
ANNstats.perc_better = perc_better;

end

function [tri,vali,tsi] = repsample(T,c,nrs,f)
% ----- Inputs ------
% T: the target data
% c: the cluster number corresponding to each target data observation
% nrs: the total sample size for random sampling within each cluster
% f: a 3-element vector of the fraction of the random sample going to the
%    training, validating, and testing subsamples. Ex. [0.5 0.25 0.25]
% 
% ----- Outputs -----
% tri: the indices for training the ANN
% vali: the indices for validating the ANN
% tsi: the indices for testing the ANN
% 
% -------------------

% Get total number of clusters
cl = length(unique(c));

% Representatively sample target examples within each explanatory data 
% cluster and divide into training, validation, and testing indices.
tri = cell(cl,1); % set up training indices
vali = cell(cl,1); % validation indices
tsi = cell(cl,1); % test indices

for cli = 1:cl
    % Indices of target examples in this explanatory data cluster
    ci = find(c==cli & ~isnan(sum(T,2)));
    
    % Pick which indices to use
    if length(ci) < nrs
        % If under-represented, use all indices
        rsi = ci;
    else
        % If fully-represented, pick a representative sample
        rsi = randsample(ci,nrs);
    end
    
    % Divide the sample into training, testing, and validation data
    [trainInd,valInd,testInd] = dividerand(length(rsi),f(1),f(2),f(3));
    tri{cli} = rsi(trainInd);
    vali{cli} = rsi(valInd);
    tsi{cli} = rsi(testInd);

    % If any are empty, make sure they are empty in a certain way
    if isempty(tri{cli})
        tri{cli} = [];
    end
    if isempty(vali{cli})
        vali{cli} = [];
    end
    if isempty(tsi{cli})
        tsi{cli} = [];
    end

end

% Concatenate all the training, testing, and validation indices from each
% explanatory data cluster into single matrices
tri = cell2mat(tri);
vali = cell2mat(vali);
tsi = cell2mat(tsi);

end