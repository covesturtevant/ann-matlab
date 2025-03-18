function [T_gf,T_pred,T_ext,unc,c,ANNstats] = ANN_robust_full(X,T,arch,clmax,nmin,clund,Next,Ninit,f,extout)
%  --------------- Usage -----------------
% 
% [T_gf,T_pred,T_ext,unc,c,ANNstats] = ANN_robust_full(X,T,arch,clmax,nmin,clund,Next,Ninit,f,extout)
%
% ------------- Description --------------
% NOTE: The Neural Network Toolbox and the Statics and Machine Learning
% Toolbox are required.
% Gap filling with representative training data selection, optimization of 
% ANN architecture complexity, and uncertainty estimation of gap-filled 
% estimates. 
% Representative data selection: Data used to train, test, and validate
%   the ANN are proportionally representative of the  conditions present 
%   in the entire gap-filled series of explanatory variables. 
% ANN complexity optimization: Varying levels of ANN architecture 
%   complexity are tested, with the selected architecture being the 
%   simplest to achieve robust estimation (< 5% decrease in mean square 
%   error for more complex architectures). 
% Uncertainty estimation: Resampling of the data used to train, test, 
%   and validate the optimum ANN is used to obtain uncertainty estimates of
%   the output.
% 
% This 'full' version does everything right. The optimization is nested so
% that the best network architecture and initialization are saved for each
% training data extraction. The downside to this version is that it is
% slow, especially for large datasets and greater network complexity.
%
% -------------- Inputs -----------------
% X: Column matrix of explanatory variables. Each new column is a different
%    explanatory variable. It is important that there be no missing data in
%    this matrix.
% T: Column matrix of the target dataset to be gap-filled. If the dataset contains
%    multiple columns (variables), the input-output relationships 
%    determined by the neural network will be a compromise between all
%    output variables. Therefore, it is important that multiple outputs be
%    logically related in some way. NOTE: right now the code is not set to
%    deal with multiple output variables. Keep it to 1 for now.
% arch: Specifies the hidden layer ANN architectures to test (optional). 
%    This is a cell matrix where each cell row is a vector of the hidden 
%    layer nodes. Each successive row should be more complex than the 
%    previous. Enter [] or {} to test the 4 default architectures: 
%           [Ninputs*1]
%           [Ninputs*1.5]
%           [Ninputs*1  Ninputs*0.5]
%           [Ninputs*1.5  Ninputs*0.75]
%           where Ninputs is the number of explanatory data variables.
% clmax: the number of clusters to divide the explanatory data into for
%    representative sampling of data fed into the ANN. The code will start
%    with this number of clusters and decrease if necessary. 10-15 is a
%    good number to start with for several months of data. Use more for
%    longer or extremely variable explanatory conditions, less for more
%    uniform conditions or less data.
% nmin: Minimum sample size of target data in each explanatory data
%    cluster. Only clund explanatory data clusters are allowed to be
%    underrepresented below this level. Otherwise, the number of explanatory
%    data clusters is reduced. Recommended nmin: 1-3% of the total (non-nan) 
%    target data size. 
% clund: the number of explanatory data clusters allowed to be under-
%    represented (n samples < nmin). 2 seems reasonable for 15
%    explanatory data clusters.
% Next: The number of random data extractions to perform. From these Next
%    extractions, the median outcome is used as the gap-filled estimate, 
%    and the uncertainty is estimated as the interquartile range. 20 seems
%    good.
% Ninit: The number of initializations to test for each network in order to
%       avoid local minima. 10 usually suffices.
% f: a 3-element vector of the fraction of the random sample going to the
%    training, validating, and testing subsamples, respectively. 
%    Equal partitioning seems reasonable: [0.333 0.333 0.333]
% extout: binary (0 or 1) indicating whether to output all ANN-predicted 
%         datasets from each data extraction. This is helpful in order to
%         compute the uncertainty of cumulative estimates.
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
%            [Next,length(arch),Ninit]
%      ANNstats.perc_better = the percentage improvement in m.s.e. for each
%            tested architecture, size [Next,length(arch)]
%      ANNstats.bestnet = the final ANN for each extraction
%      ANNstats.PSX = the scale settings returned from mapminmax on X in 
%      order to apply the same mappings to a different explanatory dataset
%      ANNstats.PST = the scale settings returned from mapminmax on T in 
%      order to apply the same mapping to a different target dataset
%
% ---------------------------------------
% The methods used in this code were inspired by instruction from Dario
% Papale.
% Copyright 2013, Cove Sturtevant. All rights reserved. 
% Updated September 2014 to include ANNstats output
% Updated January 2024 to include mapminmax scaling settings in ANNstats

%% Prep the data

% Scale all inputs/outputs between -1 and 1
[X,PSX] = mapminmax(X'); X = X';
T_gf = T; % Save original 
[T,PST] = mapminmax(T'); T = T';

% Create several levels of network architecture complexity to test
if isempty(arch)
    arch{1} = ceil(size(X,2)*1);
    arch{2} = ceil(size(X,2)*1.5);
    arch{3} = [ceil(size(X,2)*1) ceil(size(X,2)*0.5)];
    arch{4} = [ceil(size(X,2)*1.5) ceil(size(X,2)*0.75)];
end


%% Cluster explanatory data for representative sampling
% Divide the dataset into natural ranges (of the explanatory data) with 
% enough data to train/test/validate ANN 
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

figure(1); clf; hold on
for i = 1:size(T,2)
    scatter((1:size(T,1))',T(:,i),3,c,'filled')
    ylabel('Rescaled target data')
    cb = colorbar;
    set(get(cb,'ylabel'),'string','Cluster')
    axis tight
    title('Environmental data clusters projected on target data')
end

%% Run the neural network
tic

% Initialize performance metrics
mse_val = NaN(Next,length(arch),Ninit);
R2_val = NaN(Next,length(arch),Ninit);
mse_ts = NaN(Next,length(arch),1);
bestIniti = NaN(Next,length(arch),1);
perc_better = NaN(Next,length(arch));
bestai = NaN(Next,1);

figure(4); clf;
h0 = plot(T_gf,'k','linewidth',2); hold on; 
title('Best ANN prediction from each data extraction')

% initialize final output matrix
if size(T,2) > 1
    T_ext = NaN([size(T) Next]); 
else
    T_ext = NaN(length(T),Next); 
end

% Extract the training data for the network multiple times. For each
% extraction, find the best network architecture and initialization.  
for Nexti = 1:Next
    disp(['Training data extraction ' num2str(Nexti) ' of ' num2str(Next)])

    bestnet = []; % Reset the best network
    
    % Representatively sample target examples within each explanatory data 
    % cluster and divide into training, validation, and testing indices.
    [tri,vali,tsi] = repsample(T,c,nrs,f);

    % Run through the different architectures
    for ai = 1:length(arch)
        disp(['     Testing hidden layer architecture [' num2str(arch{ai}) '] (' num2str(ai) ' of ' num2str(length(arch)) ')...'])

        % Create the network
        net = fitnet(arch{ai},'trainlm');
        net.trainParam.showWindow = 0;
        net.trainParam.goal = 0;
        net.trainParam.mu = 0.001;
        net.trainParam.mu_inc = 10;

        % Set the training/val/test indices to the representative samples - for
        % some reason the ANN is MUCH faster if you reduce the dataset to just
        % the training, validating, and testing indices, each indexed 
        % consecutively.
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
            Ty = net(X')'; % Simulate all outputs
            figure(2); clf; plot((1:size(T))',T,'k',(1:size(T))',Ty,'r',tri,T(tri),'b*'); 
            legend('data','fit','training set'); 
            title(['Hidden layer architecture: [' num2str(arch{ai}) '], Initialization ' num2str(Initi) ' of ' num2str(Ninit)]); pause(0.1) 

            % Get mse of validation set (paritially independent)
            %mse_val(Nexti,ai,Initi) = perform(net,T(vali,:)',Ty(vali,:)');
            mse_val(Nexti,ai,Initi) = nanmean((T(vali,:)-Ty(vali,:)).^2);
            R2_val(Nexti,ai,Initi) = corr(T(vali,:),Ty(vali,:),'rows','complete').^2;
            
            % Plot the training and test regressions
            figure(3); clf
            subplot(1,3,1)
            plot([-1 1],[-1 1],'k',T(tri,:),Ty(tri,:),'k.'); 
            title(['Training set: R^2 = ' num2str(mean(diag(corr(T(tri,:),Ty(tri,:)).^2)),2)])
            subplot(1,3,2)
            plot([-1 1],[-1 1],'k',T(vali,:),Ty(vali,:),'k.'); 
            title(['Validation set: R^2 = ' num2str(mean(diag(corr(T(vali,:),Ty(vali,:)).^2)),2)])
            xlabel('measured'); ylabel('modeled')
            subplot(1,3,3)
            plot([-1 1],[-1 1],'k',T(tsi,:),Ty(tsi,:),'k.'); 
            title(['Independent Test set: R^2 = ' num2str(mean(diag(corr(T(tsi,:),Ty(tsi,:)).^2)),2)])
            
            % Check if this is the best initialization of this architecture
            % and possibly overall
            if mse_val(Nexti,ai,Initi)/R2_val(Nexti,ai,Initi) <= nanmin(mse_val(Nexti,ai,:)./R2_val(Nexti,ai,:))
%             if mse_val(Nexti,ai,Initi) <= nanmin(mse_val(Nexti,ai,:))
                
                % Save the best independent (test) set error
                mse_ts(Nexti,ai) = perform(net,T(tsi)',Ty(tsi)');
                bestIniti(Nexti,ai,1) = Initi;

                % Save the plot of the absolute best network
                figure(5); clf
                subplot(1,3,1)
                plot([-1 1],[-1 1],'k',T(tri,:),Ty(tri,:),'k.'); 
                title(['Training set: R^2 = ' num2str(mean(diag(corr(T(tri,:),Ty(tri,:)).^2)),2)])
                subplot(1,3,2)
                plot([-1 1],[-1 1],'k',T(vali,:),Ty(vali,:),'k.'); 
                title(['Best [' num2str(arch{ai}) '] Validation set: R^2 = ' num2str(mean(diag(corr(T(vali,:),Ty(vali,:)).^2)),2)])
                xlabel('measured'); ylabel('modeled')
                subplot(1,3,3)
                plot([-1 1],[-1 1],'k',T(tsi,:),Ty(tsi,:),'k.'); 
                title(['Independent Test set: R^2 = ' num2str(mean(diag(corr(T(tsi,:),Ty(tsi,:)).^2)),2)])

                % Evaluate if this is the best overall architecture (best of
                % the best test errors)
                if ai == 1
                    bestnet = net;
                    bestai(Nexti) = 1;
                    perc_better(Nexti,1) = 0;
                    
                    % Save the plot of the absolute best network
                    figure(6); clf
                    subplot(1,3,1)
                    plot([-1 1],[-1 1],'k',T(tri,:),Ty(tri,:),'k.'); 
                    title(['Training set: R^2 = ' num2str(mean(diag(corr(T(tri,:),Ty(tri,:)).^2)),2)])
                    subplot(1,3,2)
                    plot([-1 1],[-1 1],'k',T(vali,:),Ty(vali,:),'k.'); 
                    title(['Best Overall Validation set ([' num2str(arch{ai}) ']): R^2 = ' num2str(mean(diag(corr(T(vali,:),Ty(vali,:)).^2)),2)])
                    xlabel('measured'); ylabel('modeled')
                    subplot(1,3,3)
                    plot([-1 1],[-1 1],'k',T(tsi,:),Ty(tsi,:),'k.'); 
                    title(['Independent Test set: R^2 = ' num2str(mean(diag(corr(T(tsi,:),Ty(tsi,:)).^2)),2)])
                    
                else
                    % This network must be at least 5% better (w.r.t. the
                    % simplest network) than the previous best to be the new 
                    % best
                    perc_better(Nexti,ai) = -(mse_ts(Nexti,ai)-mse_ts(Nexti,1))/mse_ts(Nexti,1)*100;
                    % perc_better(Nexti,ai) = -(mse_val(Nexti,ai,Initi)-mse_val(Nexti,1,bestIniti(Nexti,1,1)))/mse_val(Nexti,1,bestIniti(Nexti,1,1))*100;
                    % perc_better(Nexti,ai) = -(mse_val(Nexti,ai,Initi)/R2_val(Nexti,ai,Initi)-mse_val(Nexti,1,bestIniti(Nexti,1,1))/R2_val(Nexti,1,1))/(mse_val(Nexti,1,bestIniti(Nexti,1,1))/R2_val(Nexti,1,1))*100; % Discount for crappy R2 
                    if (perc_better(Nexti,ai) - perc_better(Nexti,bestai(Nexti))) >= 5
                        bestnet = net;
                        bestai(Nexti) = ai;
                        
                        % Save the plot of the absolute best network
                        figure(6); clf
                        subplot(1,3,1)
                        plot([-1 1],[-1 1],'k',T(tri,:),Ty(tri,:),'k.'); 
                        title(['Training set: R^2 = ' num2str(mean(diag(corr(T(tri,:),Ty(tri,:)).^2)),2)])
                        subplot(1,3,2)
                        plot([-1 1],[-1 1],'k',T(vali,:),Ty(vali,:),'k.'); 
                        title(['Best Overall Validation set ([' num2str(arch{ai}) ']): R^2 = ' num2str(mean(diag(corr(T(vali,:),Ty(vali,:)).^2)),2)])
                        xlabel('measured'); ylabel('modeled')
                        subplot(1,3,3)
                        plot([-1 1],[-1 1],'k',T(tsi,:),Ty(tsi,:),'k.'); 
                        title(['Independent Test set: R^2 = ' num2str(mean(diag(corr(T(tsi,:),Ty(tsi,:)).^2)),2)])


                    end
                end
            end

            % Re-initialize network weights for next pass
            net = init(net); 

        end
        
    end
    
    % After all architectures and initializations have been run, save
    % the output from the best network from this extraction
    ANNstats.bestnet{Nexti,1} = bestnet;
    Ty = bestnet(X')'; % Simulate all outputs
    if size(T,2) > 1
        T_ext(:,:,Nexti) = mapminmax('reverse',Ty',PST)';
        
        figure(4); 
        h1 = plot(T_ext(:,:,Nexti),'color',[0.5 0.5 0.5]);
        legend([h0;h1],'data','run fit')
        title(['Best ANN predictions:' num2str(Nexti) ' of ' num2str(Next)])
        ylabel('Target data')
        pause(0.1)
        
    else
        T_ext(:,Nexti) = mapminmax('reverse',Ty',PST)';

        figure(4); 
        h1 = plot(T_ext(:,Nexti),'color',[0.5 0.5 0.5]);
        legend([h0;h1],'data','run fit')
        title(['Best ANN predictions:' num2str(Nexti) ' of ' num2str(Next)])
        ylabel('Target data')
        pause(0.1)
    end
    

    disp(['  Best architecture: [' num2str(arch{bestai(Nexti)}) ']'])
        
end

% Show the percent improvement of each ANN complexity compared to the simplest ANN
% perc_better
    
% Save the median output
if size(T,2) > 1
    T_pred = median(T_ext,3); % Entire predicted output
    T_gf(isnan(T_gf)) = T_pred(isnan(T_gf)); % Gap-filled output
    
    % Get interquartile range of predicted datasets
    unc = [prctile(T_ext,25,3) prctile(T_ext,75,3)];

else
    T_pred = median(T_ext,2); % Entire predicted output
    T_gf(isnan(T_gf)) = T_pred(isnan(T_gf)); % Gap-filled output
    
    % Get interquartile range of predicted datasets
    unc = [prctile(T_ext,25,2) prctile(T_ext,75,2)];

end

figure(4); hold on
h2 = plot(T_pred,'color',[1 0 0]);
legend([h0;h1;h2],'data','run fit','median fit')
title('Uncertainty of ANN fit')


% If not output the results from all extractions, clear the output
if ~extout
    T_ext = NaN(size(T_gf));
end

% Output the ANN stats from the optimization procedure
ANNstats.arch = arch;
ANNstats.bestai = bestai;
ANNstats.mse_val = mse_val;
ANNstats.R2_val = R2_val;
ANNstats.bestIniti = bestIniti;
ANNstats.mse_ts = mse_ts;
ANNstats.perc_better = perc_better;
ANNstats.PSX = PSX;
ANNstats.PST = PST;

toc
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