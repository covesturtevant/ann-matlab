clear
fileFlux = 'D:\BaldocchiLab\FieldSiteData\MayberryWetland\MB_2010287to2015217_L3.mat';
load(fileFlux) % mat file with the input data
fileSave = 'D:\BaldocchiLab\FieldSiteData\MayberryWetland\Analysis\ANNanalysis\MB_2010287to2015217_L3_ANNanalysis.mat';

datelim = [datenum(2013,1,1) datenum(2014,1,1)];
seasonvar = 0;

driver_fields_met = {'TA','TS_TULE_32cm','PAR','VPD','WT'};
driver_fields_flux = {'LE','ustar','WD'};
driver_fields = [driver_fields_met, driver_fields_flux];

drivers = NaN(size(data.TA,1),numel(driver_fields));
for i = 1:numel(driver_fields_met)
    drivers(:,i) = getfield(Metdata,driver_fields{i});
end
for i = (numel(driver_fields_met)+1):numel(driver_fields)
    drivers(:,i) = getfield(data,driver_fields{i});
end

if seasonvar
    drivers(:,numel(driver_fields)+1) = sin((data.DOY-1)/365*2*pi);
    drivers(:,numel(driver_fields)+2) = cos((data.DOY-1)/365*2*pi);
    driver_fields = [driver_fields, {'seasonSin','seasonCos'}];
end

% Cut down to date range in question
setUse = find(data.Mdate >= datelim(1) & data.Mdate < datelim(2)); 
Mdate = data.Mdate(setUse);
flux = data.wm(setUse);
nameFlux = 'wm';
drivers = drivers(setUse,:);

% Plot the measured and predicted fluxes
xticks = floor(Mdate(1)):round((Mdate(end)-Mdate(1))/6):ceil(Mdate(end));
zerox = [Mdate(1) Mdate(end)];
zeroy = [0 0];
[y,m,d, ~, ~,~]=datevec(Mdate);
if y(end)-y(1) > 0
    dfmt = 2; % mm/dd/yy
else
    dfmt = 6; % mm/dd
end

figure(1); clf
plot(Mdate,flux,'k.')
axis tight
set(gca,'xtick',xticks)
datetick('x',dfmt,'keeplimits','keepticks')    
legend('Measured')

%% Run the ANN
arch=[];
clmax=10;
nmin=ceil(sum(~isnan(flux)).*0.02);
clund=2;
Next=5;
Ninit=10;
f=[0.333 0.333 0.333];
extout=1;

disp(' ')
disp('Running ANN...')
%T_gf                 T_pred              T_ext_day     unc                              c              stats    
[~,ANNpred,ANNext,ANNunc,ANNc,ANNstats] = ANN_robust_full(drivers,flux,arch,clmax,nmin,clund,Next,Ninit,f,extout);

save(fileSave,'fileFlux','fileSave','flux','nameFlux','Mdate','drivers','driver_fields','ANNpred','ANNext','ANNunc','ANNc','ANNstats')

%% Plot ANN performance
load(fileSave)

figure(3); clf
plot(flux,ANNpred,'.')
xlabel([nameFlux])
ylabel(['ANN median prediction of ' nameFlux])
min11 = min([flux;ANNpred]);
max11 = max([flux;ANNpred]);
hold on
plot([min11 max11],[min11 max11],'k')
axis tight

ni = find(~isnan(flux+ANNpred));
[rho,p] = corr(flux(ni),ANNpred(ni));
text(0.9*max(get(gca,'xlim')),0.8*min(get(gca,'ylim')),{['R^2 = ' num2str((rho^2)*100,3) '%']},'color','r','horizontalalignment','right','verticalalignment','bottom')

%% Do some analysis of the ANN connection weights
% Note - can be run without re-running the ANN by loading the file saved
% above
load(fileSave)


numNets = numel(ANNstats.bestnet); % number of networks run (typically to gather uncertainty)
RI = NaN(numNets,numel(driver_fields)); % initialize connection weights output
for iNet = 1:numel(ANNstats.bestnet)
    [RI(iNet,:),~] = ANN_connection_weights(ANNstats.bestnet{iNet});
end
RImean = mean(RI,1);
RIstd = std(RI,0,1);

[~,isort] = sort(RImean,'descend');

% Bar plot!!
ANNvar = figure(4); clf
bar([1:size(RI,2)],RImean(isort)); hold on
errorbar(1:size(RI,2), RImean(isort),RIstd,'k.')
legend({'mean','std'})
ax = gca;
ax.XTickLabel = driver_fields(isort);
xlabel('Input variable')
ylabel('Relative variable importance (%)')
title(['Analysis of ANN Connection Weights predicting ' nameFlux])

%% Get response function for a particular driver cluster
load(fileSave)

% Choose values for the next 4 lines. Then run the section.
cluster = 7; % driver cluster to use (probably in the range of 1-15
driver_fields_vary = {'WD','TA'}; % Which drivers do you want to vary (enter 2)
nval = 10; % Number of values to test across the range of the drivers for this cluster 
plot_errorbars = 1; % Set to 1 in order to plot error bars on the response plots. Set to 0 to only plot the median. % Errorbars show median absolute deviation of the response across ANN replicates.

% % Use the clustering performed on the drivers to find reasonable
% combinations of drivers, and vary the drivers we want to test. Then run 
% the ANN at these driver values
[drivers_test] = ANN_vary_drivers_by_cluster(drivers,driver_fields,driver_fields_vary,ANNc,cluster,nval);
[response,driver_fields_varied] = ANN_response_function(ANNstats,drivers_test);

% Set up plotting
xticks = floor(Mdate(1)):round((Mdate(end)-Mdate(1))/6):ceil(Mdate(end));
zerox = [Mdate(1) Mdate(end)];
zeroy = [0 0];
[y,m,d, ~, ~,~]=datevec(Mdate);
if y(end)-y(1) > 0
    dfmt = 2; % mm/dd/yy
else
    dfmt = 6; % mm/dd
end

figure(5); clf
subplot(1,3,1)
plot(Mdate,flux,'.'); hold on
setCluster = ANNc == cluster;
h = plot(Mdate(setCluster),flux(setCluster),'r.');
axis tight
set(gca,'xtick',xticks)
datetick('x',dfmt,'keeplimits','keepticks')    
ylabel([nameFlux])
title(['Driver data cluster ' num2str(cluster)])

% Plot the median Driver conditions values for the cluster
[drivers_median] = ANN_vary_drivers_by_cluster(drivers,driver_fields,[],ANNc,cluster,1);
driver_median_text = {'Median driver conditions:'};
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
ylabel([nameFlux])
title(['Flux relationship with ' driver_fields_varied{1} ' and ' driver_fields_varied{2}])

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
ylabel([nameFlux])
title(['Flux relationship with ' driver_fields_varied{2} ' and ' driver_fields_varied{1}])

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

%% Get response function for all clusters 
% This section is best to run for the major drivers so that the relationship
% is shown for their entire data range. Because the other drivers are more 
% minor, their variability does not obscure the relationship too much. 

% Note - this section can be run without re-running the ANN by loading the 
% file saved above.
%load(fileSave)

% Choose values for the next 2 lines. Then run the section (takes several
% seconds)
driver_fields_vary = {'TW_16cm','WT'}; % Which drivers do you want to vary (enter 2)
nval = 10; % Number of values to test across the range of the drivers for this cluster 


% Create a colormap
cmap = colormap;
nc = size(colormap,1);

% Match the full range of the varying drivers to the colormap
n_vary = length(driver_fields_vary);
driver_fields_scale = cell(1,length(driver_fields_vary));
for i = 1:n_vary
    ifield = strmatch(driver_fields_vary{i},driver_fields);
    rng = [prctile(drivers(:,ifield),.1) prctile(drivers(:,ifield),99.9)]; % 0.1 to 99.9th percentile
    driver_fields_scale{i} = rng(1):(rng(2)-rng(1))/(nc-1):rng(2);
end

figure(6); clf
clusters = sort(unique(ANNc(~isnan(ANNc))));
for ci = 1:length(clusters)
    disp(['Running cluster ' num2str(ci) ' of ' num2str(clusters(end))])
    cluster = clusters(ci);

    % Use the clustering performed on the drivers to find reasonable
    % combinations of drivers, and vary the drivers we want to test. Then run 
    % the ANN at these driver values
    [drivers_test] = ANN_vary_drivers_by_cluster(drivers,driver_fields,driver_fields_vary,ANNc,cluster,nval);
    [response,driver_fields_varied] = ANN_response_function(ANNstats,drivers_test);

    % Set up plotting
    xticks = floor(Mdate(1)):round((Mdate(end)-Mdate(1))/6):ceil(Mdate(end));
    zerox = [Mdate(1) Mdate(end)];
    zeroy = [0 0];
    [y,m,d, ~, ~,~]=datevec(Mdate);
    if y(end)-y(1) > 0
        dfmt = 2; % mm/dd/yy
    else
        dfmt = 6; % mm/dd
    end

    clear leg
    var_multiline = drivers_test.(driver_fields_varied{2});
    for i = 1:length(var_multiline)
    leg{i} = [num2str(var_multiline(i))];
    end

    figure(6);
    subplot(1,2,1)
    response_median_1 = median(response,3);
    p1 = plot(drivers_test.(driver_fields_varied{1}),response_median_1); hold on
    %hold on; errorbar(drivers_test.(driver_fields_varied{1}),median(response(:,1,:),3), std(response(:,1,:),0,3),'k.')
    xlabel(driver_fields_varied{1})
    ylabel([nameFlux])
    %legend(leg,'Location','best')
    title(['Flux relationship with ' driver_fields_varied{1} ' and ' driver_fields_varied{2} ' across all driver clusters'])

    % Set the color of the lines to scale with the 2nd driver
    i = strmatch(driver_fields_varied{2},driver_fields_vary);
    ic = dsearchn(driver_fields_scale{i}',drivers_test.(driver_fields_varied{2})'); % color index
    for ip = 1:length(p1)
        set(p1(ip),'Color',cmap(ic(ip),:))
    end
    if ci == length(clusters)
        cb1 = colorbar;
        caxis([driver_fields_scale{i}(1) driver_fields_scale{i}(end)])
        set(get(cb1,'ylabel'),'string',driver_fields_varied{2})
        axis tight
    end
    
    
    clear leg
    var_multiline = drivers_test.(driver_fields_varied{1});
    for i = 1:length(var_multiline)
    leg{i} = [num2str(round(var_multiline(i),2,'significant')) ':' num2str(cluster)];
    end

    subplot(1,2,2)
    response_median_2 = median(response,3)';
    p2 = plot(drivers_test.(driver_fields_varied{2}),response_median_2); hold on
    %hold on; errorbar(drivers_test.(driver_fields_varied{1}),median(response(:,1,:),3), std(response(:,1,:),0,3),'k.')
    xlabel(driver_fields_varied{2})
    ylabel([nameFlux])
    %legend(leg,'Location','best')
    title(['Flux relationship with ' driver_fields_varied{2} ' and ' driver_fields_varied{1} ' across all driver clusters'])

    % Set the color of the lines to scale with the 2nd driver
    i = strmatch(driver_fields_varied{1},driver_fields_vary);
    ic = dsearchn(driver_fields_scale{i}',drivers_test.(driver_fields_varied{1})'); % color index
    for ip = 1:length(p2)
        set(p2(ip),'Color',cmap(ic(ip),:))
    end
    if ci == length(clusters)
        cb2 = colorbar;
        caxis([driver_fields_scale{i}(1) driver_fields_scale{i}(end)])
        set(get(cb2,'ylabel'),'string',driver_fields_varied{1})
        axis tight
    end
    

end