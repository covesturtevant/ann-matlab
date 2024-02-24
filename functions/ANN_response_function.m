function [response,driver_fields_varied] = ANN_response_function(ANNstats,drivers_test)
% Show the response curve or response surface of the target variable to 
% one or two driver variables, setting all other drivers to specific
% values.
% -------------- Inputs -----------------
% net: A neural network object as output by e.g. ANN_robust_full.m in the
%   ANNstats.bestnet output
% drivers: The R x C matrix of C drivers exactly as input into ANN_robust_full.m
% driver_fields: A 1 x C cell array of the column names of drivers.
% target: A R x 1 matrix of the target variable for which the ANN was
%       created.
% ANNstats
% drivers_test = a structure of the driver values to test. The fields in
%       this structure MUST be in the same column order as the driver set 
%       used to train the ANN. 
%
% -------------- Outputs -----------------
% response: A N-dimensional predicted dataset at all combinations of the varying
%   drivers, with the other drivers held constant at their input values. N
%   is the number of drivers that vary, and the index in response
%   corresponds to the prediction at that combination of the varying
%   drivers. Thus, if 
%
% -------------- Example -----------------
% net = ANNstats.bestnet{1}
% [response,driver_fields_varied] = ANN_response_function(net,drivers,ANNstats.PSX,ANNstats.PST)
% ---------------------------------------
% Authorship: 2023, Cove Sturtevant. 

% Get the driver fields that have varied (max 2)
driver_fields = fields(drivers_test);
n_fields = length(driver_fields);
driver_fields_varied = {};
nValues_varied = [];
for fi = 1:length(driver_fields)
    field_i = driver_fields{fi};
    numVal = length(drivers_test.(field_i));
    if numVal > 1
        driver_fields_varied = [driver_fields_varied field_i];
        nValues_varied = [nValues_varied numVal];
    end
end
n_fields_vary = length(driver_fields_varied);
if n_fields_vary > 2
    error('Max 2 varying driver fields')
end

% Initialize the output. 
numNets = numel(ANNstats.bestnet); % number of networks run (typically to gather uncertainty)
response = NaN(nValues_varied);
response = repmat(response,1,1,numNets);

field_1 = driver_fields_varied{1};
values_1 = drivers_test.(field_1);
for vi = 1:length(values_1)

    if n_fields_vary == 2
        field_2 = driver_fields_varied{2};
        n_values_2 = length(drivers_test.(field_2));
    else
        n_values_2 = 1;
    end

    % Create the driver set to feed into the ANN
    drivers_test_mat = NaN(n_values_2,n_fields);
    for i = 1:n_fields
        fi = driver_fields{i};
        if strcmp(fi,field_1)
            drivers_test_mat(:,i) = values_1(vi);
        else 
            drivers_test_mat(:,i) = drivers_test.(fi);
        end
    end

    % Predict the output fluxes with the chosen driver values
    X = mapminmax('apply',drivers_test_mat',ANNstats.PSX);
    for iNet = 1:numNets
        net = ANNstats.bestnet{iNet};
        Ty = net(X);
        response(vi,:,iNet) = mapminmax('reverse',Ty,ANNstats.PST)'; % Rows are driver
    end

end


end