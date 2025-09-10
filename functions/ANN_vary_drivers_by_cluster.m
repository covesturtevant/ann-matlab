function [drivers_test] = ANN_vary_drivers_by_cluster(drivers,driver_fields,driver_fields_vary,driver_clusters,cluster,n)
% Create structure of driver values to test for a certain driver cluster.
% The clusters should have been created in one of the ANN_robust functions.
% One or two drivers are allowed to vary within the 1st and 99th percentile
% of the cluster, and all other drivers are set to their median values of
% the cluster.
% -------------- Inputs -----------------
% drivers: The exact same driver dataset used to train the ANN. R x C
%       numeric matrix. R = number of samples. C = number of variables.
% driver_fields: A 1 x C cell array of the column names of drivers.
% driver_fields_vary: A 1 x m cell subset of driver_fields, where m <= C. 
%       These are the drivers to vary.
% driver_clusters: The cluster category for each row of drivers.
% cluster: The cluster to test
% n: The number of values to vary from the 1st to 99th percentile for the 
%       driver_fields_vary.
%
% -------------- Outputs -----------------
% drivers_test = a structure of the driver values to test. The fields in
%       this structure MUST be in the same column order as the driver set 
%       used to train the ANN.
%
% -------------- Example -----------------
% [drivers_test] = ANN_vary_drivers_by_cluster(drivers,{'ta','pressure','water_level','salinity'},{'ta','salinity'},driver_clusters,1,10)
% ---------------------------------------
% Authorship: 2024, Cove Sturtevant. 

% Compute 1st percentile, 99th percentile, and median driver values for this cluster
% Note, we compute 1st -99th percentiles to avoid outliers
driverClust = drivers(driver_clusters == cluster,:);

% Assign the median values to the driver test set
n_fields = length(driver_fields);
drivers_test = struct();
minDriverClust = NaN(1,n_fields); % initialize
medDriverClust = NaN(1,n_fields); % initialize
maxDriverClust = NaN(1,n_fields); % initialize
for i = 1:n_fields
    minDriverClust(i) = prctile(driverClust(~isnan(driverClust(:,i)),i),1,1); % Get average values for this cluster
    medDriverClust(i) = nanmedian(driverClust(:,i),1); % Get average values for this cluster
    maxDriverClust(i) = prctile(driverClust(~isnan(driverClust(:,i)),i),99,1); % Get average values for this cluster
    drivers_test.(driver_fields{i}) = medDriverClust(i);
end

% Now set the chosen drivers to vary from their 1st to 99th percentile
n_fields_vary = length(driver_fields_vary);
if n_fields_vary > 0
    nValues = NaN(1,n_fields_vary);
    for i = 1:n_fields_vary
        driver_vary = driver_fields_vary{i};
        idx = strmatch(driver_vary,driver_fields,'exact');
        drivers_test.(driver_vary) = minDriverClust(idx):(maxDriverClust(idx)-minDriverClust(idx))/(n-1):maxDriverClust(idx);
        nValues(i) = length(drivers_test.(driver_vary));
    end
end

end

