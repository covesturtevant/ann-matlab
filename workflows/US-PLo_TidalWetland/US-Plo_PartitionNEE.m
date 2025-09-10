% Partition NEE into GPP and ER using ANN method
% Run with Matlab 2016b
% Cove Sturtevant, 2025

clear
fileData=''; % mat file with the input data in structure "data"
fileSave = 'data_partitioned.mat'; % File path for where to save the output 

% ----- Run program ------

% Load the input data
load(fileData); % mat file with the input data

% Time variables
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

% Determine day or night
nighti = find(data.ze >= 90);
dayi = find(data.ze < 90);


%% Partition NEE

disp('Partitioning GPP & ER using ANNnight method...')

% Remove daytime data from wc. We will do ANN for night-time data
% and extend to daytime
wcFP = data.NEE;
wcFP(dayi) = NaN;
wcFP(wcFP < -5) = NaN;

driver_fields = {'SpCond','WTD','T2','T4','T7','Tair_f','sin','cos','ustar'};
drivers = NaN(size(data.Mdate,1),numel(driver_fields));
driver_fields_text = '';
for i = 1:numel(driver_fields)
    drivers(:,i) = getfield(data,driver_fields{i});
    driver_fields_text = [driver_fields_text ',' driver_fields{i}];
end
driver_fields_text = driver_fields_text(2:numel(driver_fields_text));

% ANN settings
arch=[];
clmax=15; % Number of environmental clusters for representative training
nmin = sum(~isnan(wcFP))/100; % Default is 1% of total non-nan target data size
clund=2;
Next=15; % number of replicate networks to run for uncertainty
Ninit=10;
f=[0.333 0.333 0.333]; % [training test validation] split
extout=1;

% Run the ANN
[~,data.er_ANNnight,data.er_ANNext,data.er_ANN_iqr(:,[1 2]),data.er_ANNc(:,1),ANNstats.er_ANNnight] = ANN_robust_full(drivers,wcFP,arch,clmax,nmin,clund,Next,Ninit,f,extout);

% Compute GPP from NEE = GPP + ER
data.gpp_ANNnight = data.NEE-data.er_ANNnight; 
data.gpp_ANNnight(data.gpp_ANNnight > 0) = NaN;

% Plot the results
if year(end)-year(1) > 0
    dfmt = 2; % mm/dd/yy
else
    dfmt = 6; % mm/dd
end
xticks = floor(data.Mdate(1)):round((data.Mdate(end)-data.Mdate(1))/6):ceil(data.Mdate(end));
zerox = [data.Mdate(1) data.Mdate(end)];
zeroy = [0 0];
    
f2 = figure('Name','GPP & ER'); clf
h = plot(zerox,zeroy,'k',data.Mdate,data.NEE,'k.',data.Mdate,data.gpp_ANNnight,'-',data.Mdate,data.er_ANNnight,'-');
set(h(end-1),'color',[0 0.6 0])
set(h(end),'color',[0.6 0 0])

axis tight
legtext = {'wc','gpp_ANNnight','er_ANNnight'};
legend(h(2:end),legtext,'location','bestoutside')
set(gca,'xtick',xticks)
datetick('x',dfmt,'keeplimits','keepticks')
ylabel({'CO_2 flux';'(\mumol m^{-2} s^{-1})'})


save(fileSave,'data','ANNstats','drivers','driver_fields')

