% Run ANN analysis for DOE Typha dataset
% Run with Matlab 2016b
% Cove Sturtevant, 2025

clear

fileData = 'data_partitioned.mat'; % input dataset in structure "data"
fileSave = 'output_ANN_analyis.mat'; % File path for where to save the output 
flux_var = 'CH4_qc'; % Variable in the input dataset to run ANN analysis on


fileData='D:\ForbrichLab\DOE_Typha_project\ANNanalysis\Analysis_ThruOct2023\inputANNwgaps_ifAGUwCond_metGF_GPP_ER.mat';
%% Run ANN
load(fileData) % load input data

% Time variable
data.Mdate = datenum(data.timestamp);

% Constrain date range - uncomment the date range to analyze
%setUse = data.Mdate >= datenum(2022,06,01) & data.Mdate < datenum(2022,10,01); % Drop period
setUse = data.Mdate >= datenum(2023,06,01) & data.Mdate < datenum(2023,10,01); % Recovery period
data = data(setUse,:);
[year,month,day,hour,minute,sec]=datevec(data.timestamp);

%VPD
es=0.6108*exp(17.27*data.Tair./(data.Tair+273.15));
ea=(data.RH./100).*es;
data.VPD=(es-ea);

%DOY calculation
doy=data.Mdate-datenum(year-1,12,31);
ddate=datenum(year, month,day, hour, minute, sec);
ddoy=ddate-datenum(2021,12,31);

% Seasonal representation (sin and cos of doy/365)
GFvar(:,1) = sin((doy-1)/365*2*pi);
GFvar(:,2) = cos((doy-1)/365*2*pi);
data.sin=GFvar(:,1);
data.cos=GFvar(:,2);
data.ddoy=ddoy;

% Solar zenith and azimuth
gtG = 5/24; % Time offset to get to Greenwhich (decimal day)
Date_utc = data.Mdate+gtG;
location.longitude = -70.83022; 
location.latitude = 42.74244; 
location.altitude = 1;
data.az=NaN(size(data.Mdate));
data.ze=data.az;
for i = 1:1:length(Date_utc)
      
        zx = datestr(Date_utc(i),'dd-mmm-yyyy HH:MM:SS');
        sun = sun_position(zx, location);
        data.az(i)=sun.azimuth;
        data.ze(i)=sun.zenith;

end
clear date_utc Date_utc

% ANN settings for input variable selection
arch=[]; % Use default architectures
clmax=5; % Number of environmental clusters for representative training
nmin=ceil(sum(~isnan(data.(flux_var))).*0.02);
clund=2; % Clusters allowed to be under-represented
Next=5; % number of replicate networks to run for uncertainty
Ninit=5; % numbe of initializations to avoid local minima
f=[0.333 0.333 0.333]; % [training test validation] split
extout=1; % Plot the output?

% Full set of drivers to test
driver_fields_all = {'Tair','PPFD','VPD','SpCond','ustar','atm_pressure','wind_dir','WTD','T1_1','T2','T3','T4','T5','T6','T7','gpp_ANNnight'};

% Systematically choose the best drivers
% Start with singular drivers. Choosing the most predictive, continue to
% add drivers until the performance does not increase (based on RMSE)
RMSE_improve = 1; % initialize
MSE_base = 1E10; % initialize
MSE_varAdd = NaN(1,length(driver_fields_all)); % initialize
R2_varAdd = NaN(1,length(driver_fields_all)); % initialize
MSE_model = []; % initialize (selected drivers model)
R2_model = []; % initialize (selected drivers model)
driver_fields_use = {}; % initialize
while RMSE_improve >= 0.05
    for iF = 1:length(driver_fields_all)
        % Add a new variable to the existing set of driver fields
        driver_fields = [driver_fields_use,driver_fields_all(iF)];
        disp(driver_fields)
               
        % Pull the drivers
        drivers = NaN(size(data.Mdate,1),numel(driver_fields));
        for i = 1:numel(driver_fields)
            drivers(:,i) = getfield(data,driver_fields{i});
        end
        
        % Initialize output
        data.wm = data.(flux_var);
        vnan = NaN(size(data.Mdate));
        data.wm_gf = vnan; data.wm_ANN = vnan; data.wm_ANNext = NaN(size(vnan,1),20); 
        data.wm_gf_iqr = [vnan vnan]; data.wm_ANNc = vnan;

        % Run the ANN 
        disp(' ')
        disp('Running ANN...')
        %T_gf         T_pred      T_ext     unc                          c              stats    
        [data.wm_gf,data.wm_ANN,data.wm_ANNext,data.wm_gf_iqr(:,[1 2]),data.wm_ANNc(:,1),ANNstats.wm] = ANN_robust_full(drivers,data.wm,arch,clmax,nmin,clund,Next,Ninit,f,extout);        
            
        % Pull MSE
        MSE_varAdd(iF) = nanmean((data.wm_ANN - data.wm).^2);
        
        % Pull R2
        [rho,p] = corr(data.wm,data.wm_ANN,'rows','complete');
        R2_varAdd(iF) = rho^2;
    end
    
    % Select the best MSE
    [MSE_new,imin] = min(MSE_varAdd);
    
    % Is the best RMSE better by at least 5%?
    RMSE_improve = -(sqrt(MSE_new)-sqrt(MSE_base))/sqrt(MSE_base);
    if RMSE_improve >= 0.05
        % Add the new variable to the base set
        driver_fields_use = [driver_fields_use, driver_fields_all(imin)];
        
        % Remove chosen variable from the test set
        driver_fields_all = setdiff(driver_fields_all,driver_fields_all(imin));
        MSE_base = MSE_new;
        
        % Save resultant model performance
        MSE_model = [MSE_model MSE_new];
        R2_model = [R2_model R2_varAdd(imin)];
        
        % Reset for next iteration
        MSE_varAdd = NaN(1,length(driver_fields_all)); % initialize
        R2_varAdd = NaN(1,length(driver_fields_all)); % initialize

    end

end

% Run the chosen ANN with full uncertainty quantification
driver_fields = driver_fields_use;
drivers = NaN(size(data.Mdate,1),numel(driver_fields));
driver_fields_text = '';
for i = 1:numel(driver_fields)
    drivers(:,i) = getfield(data,driver_fields{i});
    driver_fields_text = [driver_fields_text ',' driver_fields{i}];
end
driver_fields_text = driver_fields_text(2:numel(driver_fields_text));

% Initialize output
Next=20; % number of replicate networks to run for uncertainty
Ninit=10; % number of network initializations to avoid local minima
data.wm = data.(flux_var);
vnan = NaN(size(data.Mdate));
data.wm_gf = vnan; data.wm_ANN = vnan; data.wm_ANNext = NaN(size(vnan,1),20); 
data.wm_gf_iqr = [vnan vnan]; data.wm_ANNc = vnan;

% Run the ANN 
disp(' ')
disp('Running ANN...')
%T_gf         T_pred      T_ext                 unc                 c              stats    
[data.wm_gf,data.wm_ANN,data.wm_ANNext,data.wm_gf_iqr(:,[1 2]),data.wm_ANNc(:,1),ANNstats.wm] = ANN_robust_full(drivers,data.wm,arch,clmax,nmin,clund,Next,Ninit,f,extout);

% Save the output
save(fileSave,'data','ANNstats','drivers','driver_fields','MSE_model','R2_model')
    
% Plot overall results/fit
if year(end)-year(1) > 0
    dfmt = 2; % mm/dd/yy
else
    dfmt = 6; % mm/dd
end
xticks = floor(data.Mdate(1)):round((data.Mdate(end)-data.Mdate(1))/6):ceil(data.Mdate(end));
zerox = [data.Mdate(1) data.Mdate(end)];
zeroy = [0 0];

f_gf = figure(1); clf;     
% Clusters
subplot(1,3,1);
scatter(data.Mdate,data.wm,3,data.wm_ANNc,'filled')
axis tight
set(gca,'xtick',xticks)
datetick('x',dfmt,'keeplimits','keepticks')    
ylabel('CH_4 flux (\mumol m^{-2} s^{-1})')
cb = colorbar;
set(get(cb,'ylabel'),'string','Cluster')
title('Environmental data clusters projected on wm')
text(max(get(gca,'xlim')),0.8*max(get(gca,'ylim')),[' Predictors: ' driver_fields_text],'color','k','horizontalalignment','right','verticalalignment','bottom')

% ANN fit
subplot(1,3,2); 
h = plot(data.wm,data.wm_ANN,'k.'); hold on
axis tight;
xlabel('CH_4 flux_{measured} (\mumol m^{-2} s^{-1})')
ylabel('CH_4 flux_{ANN} (\mumol m^{-2} s^{-1})')
ni = find(~isnan(data.wm+data.wm_ANN));
if ~isempty(ni)
    b = robustfit(data.wm(ni),data.wm_ANN(ni));
    [rho,p] = corr(data.wm(ni),data.wm_ANN(ni));
    text(0.9*max(get(gca,'xlim')),0.8*min(get(gca,'ylim')),{['Robust fit'];['y = ' num2str(b(2),3) 'x + ' num2str(b(1),2)];['R^2 = ' num2str((rho^2)*100,3) '%']},'color','r','horizontalalignment','right','verticalalignment','bottom')
    l11 = [min([get(gca,'xlim') get(gca,'ylim')]) max([get(gca,'xlim') get(gca,'ylim')])];
    plot(l11,l11,'k')
    axis tight
    plot(get(gca,'xlim'),get(gca,'xlim')*b(2)+b(1),'color',[1 0 0])
    
    % Can also try orthogonal fit
    b_ort = fliplr(linortfit2(data.wm(ni),data.wm_ANN(ni)));
    text(0.7*max(get(gca,'xlim')),0.5*min(get(gca,'ylim')),{['Orthogonal fit'];['y = ' num2str(b_ort(2),3) 'x + ' num2str(b_ort(1),2)]},'color','g','horizontalalignment','right','verticalalignment','bottom')
    plot(get(gca,'xlim'),get(gca,'xlim')*b_ort(2)+b_ort(1),'color',[0 1 0])
end

% Uncertainty
subplot(1,3,3); hold on
plot(zerox,zeroy,'k')
he = plot(data.Mdate,data.wm_ANNext,'color',[0.5 0.5 0.5]);
h = plot(data.Mdate,data.wm,'k-',data.Mdate,data.wm_ANN,'r-','linewidth',2);
axis tight; legend([h; he(1)],'data','ANN fit median','ANN fit variability')
title('ANN performance: CH_4 flux')
ylabel('CH_4 flux (nmol m^{-2} s^{-1})')
set(gca,'xtick',xticks)
datetick('x',dfmt,'keeplimits','keepticks')


%% Get response function for data space observed over a specific time range
% Note that the response function is derived from the model that was
% trained over the entire dataset. All we are doing is constraining
% the varaible space of two variables of interest to that observed for 
% specified time, setting all other inputs to the model to their median 
% values observed over the specified date range.

% Note - this section can be run without re-running the ANN by loading the 
% file saved above.
load(fileSave)

% Choose values for the next 4 lines. Then run the section.
time_range=[datenum(2022,05,01) datenum(2023,10,01)]; % Time range to constrain driver range 
driver_fields_vary = {'Tair','SpCond'}; % Which drivers do you want to vary (enter 2)
nval = 10; % Number of values to test across the range of the drivers for this cluster 
plot_errorbars = 1; % Set to 1 in order to plot error bars on the response plots. Set to 0 to only plot the median. % Errorbars show median absolute deviation of the response across ANN replicates.

% Make a fake clustering variable to isolate the time range we want
ANNc = zeros(size(data.Mdate));
ANNc(data.Mdate >= time_range(1) & data.Mdate < time_range(2)) = 1;
clust=1; 

% % Use the time range we specified to find reasonable combinations of 
% drivers (i.e. the medians), and vary the drivers we want to test. 
% Then run the ANN at these driver values
[drivers_test] = ANN_vary_drivers_by_cluster(drivers,driver_fields,driver_fields_vary,ANNc,clust,nval);
[response,driver_fields_varied] = ANN_response_function(ANNstats.wm,drivers_test);

% Set up plotting
[y,~,~,~,~,~] = datevec(data.Mdate);
if y(end)-y(1) > 0
dfmt = 2; % mm/dd/yy
else
dfmt = 6; % mm/dd
end
xticks = ceil(data.Mdate(1)):round((data.Mdate(end)-data.Mdate(1))/4):ceil(data.Mdate(end));
zerox = [data.Mdate(1) data.Mdate(end)];
zeroy = [0 0];

figure(3); clf
subplot(1,3,1)
plot(data.Mdate,data.wm,'.'); hold on
setCluster = ANNc == clust;
h = plot(data.Mdate(setCluster),data.wm(setCluster),'r.');
axis tight
set(gca,'xtick',xticks)
datetick('x',dfmt,'keeplimits','keepticks')    
ylabel('CH_4 flux (\mumol m^{-2} s^{-1})')
title(['Environmental data cluster ' num2str(clust)])

% Plot the median environmental conditions values for the cluster
[drivers_median] = ANN_vary_drivers_by_cluster(drivers,driver_fields,[],ANNc,clust,1);
driver_median_text = {'Median environmental conditions:'};
for i = 1:numel(driver_fields)
    driver_median_text = [driver_median_text ; {[driver_fields{i} ': ',num2str(drivers_median.(driver_fields{i}))]}];
end
xlims = get(gca,'xlim');
ylims = get(gca,'ylim');
text(min(xlims)+0.1*(max(xlims)-min(xlims)),min(ylims)+0.95*(max(ylims)-min(ylims)),driver_median_text,'color','k','horizontalalignment','left','verticalalignment','top')

% Create a colormap
cmap = colormap;
nc = size(colormap,1);

% Match the range of the varying drivers to the colormap
n_vary = length(driver_fields_varied);
driver_fields_scale = cell(1,length(driver_fields_varied));
for i = 1:n_vary
    rng = [drivers_test.(driver_fields_varied{i})(1) drivers_test.(driver_fields_varied{i})(end)]; % 0.1 to 99.9th percentile
    driver_fields_scale{i} = rng(1):(rng(2)-rng(1))/(nc-1):rng(2);
end

subplot(1,3,2)
response_median_1 = median(response,3);
p1 = plot(drivers_test.(driver_fields_varied{1}),response_median_1,'linewidth',1);
if plot_errorbars
    hold on; 
    for i = 1:nval
        ebars = NaN(nval,1);
        for iv = 1:nval
            ebars(iv) = mad(response(iv,i,:),1);
        end
        %ebars = std(response(:,i,:),0,3);
        p1e(i) = errorbar(drivers_test.(driver_fields_varied{1}),response_median_1(:,i), ebars,'k.','linewidth',0.5);
    end
end
xlabel(driver_fields_varied{1})
ylabel('FCH4 \mumol m-2 s-1')
title(['CH4 response to ' driver_fields_varied{1} ' and ' driver_fields_varied{2}])

% Set the color of the lines to scale with the 2nd driver
ic = dsearchn(driver_fields_scale{2}',drivers_test.(driver_fields_varied{2})'); % color index
for ip = 1:length(p1)
    set(p1(ip),'Color',cmap(ic(ip),:))
    if plot_errorbars
        set(p1e(ip),'Color',cmap(ic(ip),:))
    end
end
cb1 = colorbar;
caxis([driver_fields_scale{2}(1) driver_fields_scale{2}(end)])
set(get(cb1,'ylabel'),'string',driver_fields_varied{2})
axis tight

subplot(1,3,3)
response_median_2 = median(response,3)';
p2 = plot(drivers_test.(driver_fields_varied{2}),response_median_2,'linewidth',1); 
if plot_errorbars
    hold on; 
    for i = 1:nval
        ebars = NaN(nval,1);
        for iv = 1:nval
            ebars(iv) = mad(response(i,iv,:),1);
        end
        %ebars = std(response(i,:,:),0,3);
        p2e(i) = errorbar(drivers_test.(driver_fields_varied{2}),response_median_2(:,i), ebars,'k.','linewidth',0.5);
    end
end
xlabel(driver_fields_varied{2})
ylabel('FCH4 \mumol m-2 s-1')
title(['CH4 response to ' driver_fields_varied{2} ' and ' driver_fields_varied{1}])

% Set the color of the lines to scale with the 2nd driver
ic = dsearchn(driver_fields_scale{1}',drivers_test.(driver_fields_varied{1})'); % color index
for ip = 1:length(p2)
    set(p2(ip),'Color',cmap(ic(ip),:))
    if plot_errorbars
        set(p2e(ip),'Color',cmap(ic(ip),:))
    end
end
cb2 = colorbar;
caxis([driver_fields_scale{1}(1) driver_fields_scale{1}(end)])
set(get(cb2,'ylabel'),'string',driver_fields_varied{1})
axis tight
