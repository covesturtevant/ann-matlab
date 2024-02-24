function [RI,CW] = ANN_connection_weights(net)
% Uses the Garson 1991 algorithm for partitioning and quantifying 
% neural network conenction weights. Follows the procedure summarized by 
% Olden and Jackson 2002 (Box 1).
% Essentially, this method multiplies the connection weights over each 
% input-output network path and computes the relative contribution of each 
% input neuron to the path. The relative contributions over all network
% paths are summed and normalized to arrive at a relative importance metric 
% for each input variable to the network as a whole.
%
% NOTE: This function currently only supports ANNs with an input layer, 1
% or 2 hidden layers, and an output layer with a single node
% -------------- Inputs -----------------
% net: A neural network object as output by e.g. ANN_robust_full.m in the
%      ANNstats.bestnet output
%
% -------------- Outputs -----------------
% RI: A 1 X nI matrix, where nI is the number of input nodes. The relative
%     importance of the input node (variable) as a percent according to
%     Garson's formula.
% CW: A matrix of connection weights with dimensions [nI nH1 nH2 nO], 
% where nI is as above, nH1
% is the number of nodes in the first hidden layer, nH2 is the number of
% nodes in the 2nd hidden layer (only if applicable) and nO is the number
% of nodes in the output layer (presumably 1). These dimensions form all
% the network paths.
%
% -------------- Example -----------------
% net = ANNstats.bestnet{1}
% [RI,CW] = ANN_connection_weights(net)
% ---------------------------------------
% Authorship: 2023, Cove Sturtevant. 

num_layers = net.numLayers;  % Includes hidden layers and output layer. Does not include input layer.

% Get the network architecture: number of nodes in each layer 
% [input <hidden layers> output]
arch = [net.input.size]; % initialize
for iLayer = 1:num_layers
    arch = [arch net.layers{iLayer}.size];
end

% iI is node of input layer
% iL1 is node of 1st layer
% iL2 is node of 2nd hidden layer (if present)
% iO is node of output layer
% cW is the overall connection weight of the network path
CW = zeros(arch); % initialize
for iI = 1:arch(1)
    for iL1 = 1:arch(2)
        for iL2 = 1:arch(3)
            if numel(arch) > 3
                for iL3 = 1:arch(4)
                    if num_layers == 3
                        iO = iL3;
                        weight_i = abs(net.IW{1,1}(iL1,iI)*net.Lw{2,1}(iL2,iL1)*net.LW{3,2}(iO,iL2));
                        CW(iI,iL1,iL2,iO) = weight_i;
                    end
                end
            else 
                iO = iL2;
                weight_i = abs(net.IW{1,1}(iL1,iI)*net.Lw{2,1}(iO,iL1));
                CW(iI,iL1,iO) = weight_i;
            end
        end
    end
end

% Compute relative contribution of each input neuron to the output for each
% (otherwise identical) network path
CWabs = abs(CW);
matrep = arch; matrep(2:end) = 1;
CWrel = CWabs./repmat(sum(CWabs,1),matrep); % relative contribution
if num_layers == 3
    % First sum over 3rd dimension
    S = sum(CWrel,3);
else
    S = CWrel;
end
% Final sum
S = sum(S,2);

% Compute relative importance of each variable
RI = S/sum(S)*100;

end