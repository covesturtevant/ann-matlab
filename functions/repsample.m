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
end

% Concatenate all the training, testing, and validation indices from each
% explanatory data cluster into single matrices
tri = cell2mat(tri);
vali = cell2mat(vali);
tsi = cell2mat(tsi);

end