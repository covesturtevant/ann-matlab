% This code will retrofit old L3 flux files with the mappings for the met
% data fed into the ANN. The proper operation of this code depends on the
% relevant portions of the L3 flux code copied here. Review carefully. Or,
% just rerun the ANN with the latest ANN function that will save the
% mappings to the output.

clear
fileL3 = 'D:\BaldocchiLab\FieldSiteData\MayberryWetland\FluxOutput\MB_2013353to2015063_L3.mat'; % mat file with original L3 output. Cannot be combined file.
nameFlux = 'wm'; % Pick a flux variable

%% Load the met data and recreate the driver matrix as was originally used

% Load in the met data
load(fileL3) 

% Grab an opts structure
opts0 = opts; % Need to get to a root options structure to determine pivot year
oname = fieldnames(opts);
while strcmp(oname{end}(1:4),'file')
    opts0 = eval(['opts0.' oname{end}]);
    
    % Go until we get to the last opts structure
    oname = fieldnames(opts0);
end

% Make a master met variable list including all variables we will use to
% gap fill the fluxes
GFmetvarlist = unique([L3opts.wtGF.metvarlist{:} L3opts.wtGF.night_metvarlist' ...
    L3opts.wqGF.metvarlist' L3opts.wqGF.night_metvarlist' L3opts.wcGF.metvarlist' ...
    L3opts.wcGF.night_metvarlist' L3opts.wcFP.analyticopts.tempvar ...
    L3opts.wcFP.analyticopts.PARvar L3opts.wcFP.ANNopts.metvarlist' ...
    L3opts.wmGF.metvarlist' L3opts.wmGF.night_metvarlist']');

% Gap fill MET variables used to gap-fill NEE & partition GPP/ER
GFvar = NaN(size(Metdata.(GFmetvarlist{1}),1),length(GFmetvarlist));
il = length(GFmetvarlist);
for i = 1:il
    [~,GFvar(:,i)] = De_spike3(Metdata.(GFmetvarlist{i}),9,150,opts0.nw);
end

% Seasonal representation (sin and cos of doy/365)
seasoni = [il+1 il+2];
GFvar(:,il+1) = sin((data.DOY-1)/365*2*pi);
GFvar(:,il+2) = cos((data.DOY-1)/365*2*pi);


% Determine day or night
nighti = find(data.ze >= 90);
dayi = find(data.ze < 90);


%% Retrofit wt ANNstats with scale mappings
if L3opts.wtGF.doT && strcmp(nameFlux,'wt')
    disp(' ')
    disp('*** Retrofitting wt ANNstats***')

    % find indices of met variables we want to use in the master matrix
    [~,vi] = ismember(L3opts.wtGF.metvarlist,GFmetvarlist);

    % Pull out flux variables used to gap-fill wt
    il = length(L3opts.wtGF.fluxvarlist);
    fluxvar = NaN(size(data.wt,1),il);
    for i = 1:il
        % If chosen, gap fill the flux variables. Otherwise, record as-is
        if L3opts.wtGF.gffluxvar
            [~,fluxvar(:,i)] = De_spike3(data.(L3opts.wtGF.fluxvarlist{i}),9,150,opts0.nw);
        else
            fluxvar(:,i) = data.(L3opts.wtGF.fluxvarlist{i});
        end
    end
    
    % Including a season variable?
    if L3opts.wtGF.seasonvar
        expdata = [GFvar(:,vi) GFvar(:,seasoni) fluxvar];
        driver_fields = [GFmetvarlist(vi);'seasonSin';'seasonCos';L3opts.wtGF.fluxvarlist];
    else
        expdata = [GFvar(:,vi) fluxvar];
        driver_fields = [GFmetvarlist(vi); L3opts.wtGF.fluxvarlist];
    end

    % Separating day/night? Organize these variables separately
    if L3opts.wtGF.night_separate

        % find indices of met variables we want to use in the master matrix
        [~,vi] = ismember(L3opts.wtGF.night_metvarlist,GFmetvarlist);

        % Pull out flux variables used to gap-fill 
        il = length(L3opts.wtGF.night_fluxvarlist);
        fluxvar = NaN(size(data.wt,1),il);
        for i = 1:il
            % If chosen, gap fill the flux variables. Otherwise, record as-is
            if L3opts.wtGF.gffluxvar
                [~,fluxvar(:,i)] = De_spike3(data.(L3opts.wtGF.night_fluxvarlist{i}),9,150,opts0.nw);
            else
                fluxvar(:,i) = data.(L3opts.wtGF.night_fluxvarlist{i});
            end
        end

        % Including a season variable?
        if L3opts.wtGF.seasonvar
            expdata_night = [GFvar(:,vi) GFvar(:,seasoni) fluxvar];
            driver_fields_night = [GFmetvarlist(vi);'seasonSin';'seasonCos';L3opts.wtGF.night_fluxvarlist]; % NEW
        else
            expdata_night = [GFvar(:,vi) fluxvar];
            driver_fields_night = [GFmetvarlist(vi);L3opts.wtGF.night_fluxvarlist]; % NEW
        end
    end


    % Gap-fill, separating day/night if chosen
    if L3opts.wtGF.night_separate
        disp('Doing day-time data...')
        %[data.wt_gf(dayi,1),data.wt_ANN(dayi,1),wt_ANNext_day,data.wt_gf_iqr(dayi,[1 2]),data.wt_ANNc(dayi,1),ANNstats.wt_day] = ANNcode(expdata(dayi,:),data.wt(dayi,:),L3opts.wtGF.testarch,L3opts.wtGF.clmax,nmin,L3opts.wtGF.clund,L3opts.wtGF.Next,L3opts.wtGF.Ninit,L3opts.wtGF.f,L3opts.wtGF.extout);
        
        ANNstats.wt_day.driver_fields = driver_fields; % NEW
        % Get the scalings for the driver and prediction datasets for future
        % analysis. This has now been added to ANN_robust_full and ANN_robust_fast
        % and can be removed if/when the ANN is rerun.
        [~,ANNstats.wt_day.PSX] = mapminmax(expdata(dayi,:)'); % NEW
        [~,ANNstats.wt_day.PST] = mapminmax(data.wt(dayi,:)'); % NEW

        
        disp(' ')
        disp('Doing night-time data...')
        %[data.wt_gf(nighti,1),data.wt_ANN(nighti,1),wt_ANNext_night,data.wt_gf_iqr(nighti,[1 2]),data.wt_ANNc(nighti,1),ANNstats.wt_night] = ANNcode(expdata_night(nighti,:),data.wt(nighti,:),L3opts.wtGF.night_testarch,L3opts.wtGF.night_clmax,nmin_night,L3opts.wtGF.night_clund,L3opts.wtGF.night_Next,L3opts.wtGF.night_Ninit,L3opts.wtGF.night_f,L3opts.wtGF.night_extout);
        
        ANNstats.wt_night.driver_fields = driver_fields_night; % NEW
        % Get the scalings for the driver and prediction datasets for future
        % analysis. This has now been added to ANN_robust_full and ANN_robust_fast
        % and can be removed if/when the ANN is rerun.
        [~,ANNstats.wt_night.PSX] = mapminmax(expdata_night(nighti,:)'); % NEW
        [~,ANNstats.wt_night.PST] = mapminmax(data.wt(nighti,:)'); % NEW

    else
        disp('Doing all data (no day/night separation)...')
        %[data.wt_gf,data.wt_ANN,data.wt_ANNext,data.wt_gf_iqr,data.wt_ANNc,ANNstats.wt_all] = ANNcode(expdata,data.wt,L3opts.wtGF.testarch,L3opts.wtGF.clmax,nmin,L3opts.wtGF.clund,L3opts.wtGF.Next,L3opts.wtGF.Ninit,L3opts.wtGF.f,L3opts.wtGF.extout);

        ANNstats.wt_all.driver_fields = driver_fields; % NEW
        % Get the scalings for the driver and prediction datasets for future
        % analysis. This has now been added to ANN_robust_full and ANN_robust_fast
        % and can be removed if/when the ANN is rerun.
        [~,ANNstats.wt_all.PSX] = mapminmax(expdata'); % NEW
        [~,ANNstats.wt_all.PST] = mapminmax(data.wt'); % NEW
        
    
    end

end

%% Retrofit wq ANNstats with scale mappings
if L3opts.wqGF.doQ && strcmp(nameFlux,'wq')
    disp(' ')
    disp('*** Retrofitting wq ANNstats***')

    % find indices of met variables we want to use in the master matrix
    [~,vi] = ismember(L3opts.wqGF.metvarlist,GFmetvarlist);

    % Pull out flux variables used to gap-fill wq
    il = length(L3opts.wqGF.fluxvarlist);
    fluxvar = NaN(size(data.wq,1),il);
    for i = 1:il
        % If chosen, gap fill the flux variables. Otherwise, record as-is
        if L3opts.wqGF.gffluxvar
            [~,fluxvar(:,i)] = De_spike3(data.(L3opts.wqGF.fluxvarlist{i}),9,150,opts0.nw);
        else
            fluxvar(:,i) = data.(L3opts.wqGF.fluxvarlist{i});
        end
    end
    
    % Including a season variable?
    if L3opts.wqGF.seasonvar
        expdata = [GFvar(:,vi) GFvar(:,seasoni) fluxvar];
        driver_fields = [GFmetvarlist(vi);'seasonSin';'seasonCos';L3opts.wqGF.fluxvarlist];
    else
        expdata = [GFvar(:,vi) fluxvar];
        driver_fields = [GFmetvarlist(vi); L3opts.wqGF.fluxvarlist];
    end

    % Separating day/night? Organize these variables separately
    if L3opts.wqGF.night_separate

        % find indices of met variables we want to use in the master matrix
        [~,vi] = ismember(L3opts.wqGF.night_metvarlist,GFmetvarlist);

        % Pull out flux variables used to gap-fill 
        il = length(L3opts.wqGF.night_fluxvarlist);
        fluxvar = NaN(size(data.wq,1),il);
        for i = 1:il
            % If chosen, gap fill the flux variables. Otherwise, record as-is
            if L3opts.wqGF.gffluxvar
                [~,fluxvar(:,i)] = De_spike3(data.(L3opts.wqGF.night_fluxvarlist{i}),9,150,opts0.nw);
            else
                fluxvar(:,i) = data.(L3opts.wqGF.night_fluxvarlist{i});
            end
        end

        % Including a season variable?
        if L3opts.wqGF.seasonvar
            expdata_night = [GFvar(:,vi) GFvar(:,seasoni) fluxvar];
            driver_fields_night = [GFmetvarlist(vi);'seasonSin';'seasonCos';L3opts.wqGF.night_fluxvarlist]; % NEW
        else
            expdata_night = [GFvar(:,vi) fluxvar];
            driver_fields_night = [GFmetvarlist(vi);L3opts.wqGF.night_fluxvarlist]; % NEW
        end
    end


    % Gap-fill, separating day/night if chosen
    if L3opts.wqGF.night_separate
        disp('Doing day-time data...')
        %[data.wq_gf(dayi,1),data.wq_ANN(dayi,1),wq_ANNext_day,data.wq_gf_iqr(dayi,[1 2]),data.wq_ANNc(dayi,1),ANNstats.wq_day] = ANNcode(expdata(dayi,:),data.wq(dayi,:),L3opts.wqGF.testarch,L3opts.wqGF.clmax,nmin,L3opts.wqGF.clund,L3opts.wqGF.Next,L3opts.wqGF.Ninit,L3opts.wqGF.f,L3opts.wqGF.extout);
        
        ANNstats.wq_day.driver_fields = driver_fields; % NEW
        % Get the scalings for the driver and prediction datasets for future
        % analysis. This has now been added to ANN_robust_full and ANN_robust_fast
        % and can be removed if/when the ANN is rerun.
        [~,ANNstats.wq_day.PSX] = mapminmax(expdata(dayi,:)'); % NEW
        [~,ANNstats.wq_day.PST] = mapminmax(data.wq(dayi,:)'); % NEW

        
        disp(' ')
        disp('Doing night-time data...')
        %[data.wq_gf(nighti,1),data.wq_ANN(nighti,1),wq_ANNext_night,data.wq_gf_iqr(nighti,[1 2]),data.wq_ANNc(nighti,1),ANNstats.wq_night] = ANNcode(expdata_night(nighti,:),data.wq(nighti,:),L3opts.wqGF.night_testarch,L3opts.wqGF.night_clmax,nmin_night,L3opts.wqGF.night_clund,L3opts.wqGF.night_Next,L3opts.wqGF.night_Ninit,L3opts.wqGF.night_f,L3opts.wqGF.night_extout);
        
        ANNstats.wq_night.driver_fields = driver_fields_night; % NEW
        % Get the scalings for the driver and prediction datasets for future
        % analysis. This has now been added to ANN_robust_full and ANN_robust_fast
        % and can be removed if/when the ANN is rerun.
        [~,ANNstats.wq_night.PSX] = mapminmax(expdata_night(nighti,:)'); % NEW
        [~,ANNstats.wq_night.PST] = mapminmax(data.wq(nighti,:)'); % NEW

    else
        disp('Doing all data (no day/night separation)...')
        %[data.wq_gf,data.wq_ANN,data.wq_ANNext,data.wq_gf_iqr,data.wq_ANNc,ANNstats.wq_all] = ANNcode(expdata,data.wq,L3opts.wqGF.testarch,L3opts.wqGF.clmax,nmin,L3opts.wqGF.clund,L3opts.wqGF.Next,L3opts.wqGF.Ninit,L3opts.wqGF.f,L3opts.wqGF.extout);

        ANNstats.wq_all.driver_fields = driver_fields; % NEW
        % Get the scalings for the driver and prediction datasets for future
        % analysis. This has now been added to ANN_robust_full and ANN_robust_fast
        % and can be removed if/when the ANN is rerun.
        [~,ANNstats.wq_all.PSX] = mapminmax(expdata'); % NEW
        [~,ANNstats.wq_all.PST] = mapminmax(data.wq'); % NEW
        
    
    end

end

%% Retrofit wc ANNstats with scale mappings
if L3opts.wcGF.doC && strcmp(nameFlux,'wc')
    disp(' ')
    disp('*** Retrofitting wc ANNstats***')

    % find indices of met variables we want to use in the master matrix
    [~,vi] = ismember(L3opts.wcGF.metvarlist,GFmetvarlist);

    % Pull out flux variables used to gap-fill wc
    il = length(L3opts.wcGF.fluxvarlist);
    fluxvar = NaN(size(data.wc,1),il);
    for i = 1:il
        % If chosen, gap fill the flux variables. Otherwise, record as-is
        if L3opts.wcGF.gffluxvar
            [~,fluxvar(:,i)] = De_spike3(data.(L3opts.wcGF.fluxvarlist{i}),9,150,opts0.nw);
        else
            fluxvar(:,i) = data.(L3opts.wcGF.fluxvarlist{i});
        end
    end
    
    % Including a season variable?
    if L3opts.wcGF.seasonvar
        expdata = [GFvar(:,vi) GFvar(:,seasoni) fluxvar];
        driver_fields = [GFmetvarlist(vi);'seasonSin';'seasonCos';L3opts.wcGF.fluxvarlist];
    else
        expdata = [GFvar(:,vi) fluxvar];
        driver_fields = [GFmetvarlist(vi); L3opts.wcGF.fluxvarlist];
    end

    % Separating day/night? Organize these variables separately
    if L3opts.wcGF.night_separate

        % find indices of met variables we want to use in the master matrix
        [~,vi] = ismember(L3opts.wcGF.night_metvarlist,GFmetvarlist);

        % Pull out flux variables used to gap-fill 
        il = length(L3opts.wcGF.night_fluxvarlist);
        fluxvar = NaN(size(data.wc,1),il);
        for i = 1:il
            % If chosen, gap fill the flux variables. Otherwise, record as-is
            if L3opts.wcGF.gffluxvar
                [~,fluxvar(:,i)] = De_spike3(data.(L3opts.wcGF.night_fluxvarlist{i}),9,150,opts0.nw);
            else
                fluxvar(:,i) = data.(L3opts.wcGF.night_fluxvarlist{i});
            end
        end

        % Including a season variable?
        if L3opts.wcGF.seasonvar
            expdata_night = [GFvar(:,vi) GFvar(:,seasoni) fluxvar];
            driver_fields_night = [GFmetvarlist(vi);'seasonSin';'seasonCos';L3opts.wcGF.night_fluxvarlist]; % NEW
        else
            expdata_night = [GFvar(:,vi) fluxvar];
            driver_fields_night = [GFmetvarlist(vi);L3opts.wcGF.night_fluxvarlist]; % NEW
        end
    end


    % Gap-fill, separating day/night if chosen
    if L3opts.wcGF.night_separate
        disp('Doing day-time data...')
        %[data.wc_gf(dayi,1),data.wc_ANN(dayi,1),wc_ANNext_day,data.wc_gf_iqr(dayi,[1 2]),data.wc_ANNc(dayi,1),ANNstats.wc_day] = ANNcode(expdata(dayi,:),data.wc(dayi,:),L3opts.wcGF.testarch,L3opts.wcGF.clmax,nmin,L3opts.wcGF.clund,L3opts.wcGF.Next,L3opts.wcGF.Ninit,L3opts.wcGF.f,L3opts.wcGF.extout);
        
        ANNstats.wc_day.driver_fields = driver_fields; % NEW
        % Get the scalings for the driver and prediction datasets for future
        % analysis. This has now been added to ANN_robust_full and ANN_robust_fast
        % and can be removed if/when the ANN is rerun.
        [~,ANNstats.wc_day.PSX] = mapminmax(expdata(dayi,:)'); % NEW
        [~,ANNstats.wc_day.PST] = mapminmax(data.wc(dayi,:)'); % NEW

        
        disp(' ')
        disp('Doing night-time data...')
        %[data.wc_gf(nighti,1),data.wc_ANN(nighti,1),wc_ANNext_night,data.wc_gf_iqr(nighti,[1 2]),data.wc_ANNc(nighti,1),ANNstats.wc_night] = ANNcode(expdata_night(nighti,:),data.wc(nighti,:),L3opts.wcGF.night_testarch,L3opts.wcGF.night_clmax,nmin_night,L3opts.wcGF.night_clund,L3opts.wcGF.night_Next,L3opts.wcGF.night_Ninit,L3opts.wcGF.night_f,L3opts.wcGF.night_extout);
        
        ANNstats.wc_night.driver_fields = driver_fields_night; % NEW
        % Get the scalings for the driver and prediction datasets for future
        % analysis. This has now been added to ANN_robust_full and ANN_robust_fast
        % and can be removed if/when the ANN is rerun.
        [~,ANNstats.wc_night.PSX] = mapminmax(expdata_night(nighti,:)'); % NEW
        [~,ANNstats.wc_night.PST] = mapminmax(data.wc(nighti,:)'); % NEW

    else
        disp('Doing all data (no day/night separation)...')
        %[data.wc_gf,data.wc_ANN,data.wc_ANNext,data.wc_gf_iqr,data.wc_ANNc,ANNstats.wc_all] = ANNcode(expdata,data.wc,L3opts.wcGF.testarch,L3opts.wcGF.clmax,nmin,L3opts.wcGF.clund,L3opts.wcGF.Next,L3opts.wcGF.Ninit,L3opts.wcGF.f,L3opts.wcGF.extout);

        ANNstats.wc_all.driver_fields = driver_fields; % NEW
        % Get the scalings for the driver and prediction datasets for future
        % analysis. This has now been added to ANN_robust_full and ANN_robust_fast
        % and can be removed if/when the ANN is rerun.
        [~,ANNstats.wc_all.PSX] = mapminmax(expdata'); % NEW
        [~,ANNstats.wc_all.PST] = mapminmax(data.wc'); % NEW
        
    
    end

end

%% Retrofit wm ANNstats with mapping
if L3opts.wmGF.doM && strcmp(nameFlux,'wm')
    disp(' ')
    disp('*** Retrofitting wm ANNstats***')

    % find indices of met variables we want to use in the master matrix
    [~,vi] = ismember(L3opts.wmGF.metvarlist,GFmetvarlist);

    % Pull out flux variables used to gap-fill
    il = length(L3opts.wmGF.fluxvarlist);
    fluxvar = NaN(size(data.wm,1),il);
    for i = 1:il
        % If chosen, gap fill the flux variables. Otherwise, record as-is
        if L3opts.wmGF.gffluxvar
            [~,fluxvar(:,i)] = De_spike3(data.(L3opts.wmGF.fluxvarlist{i}),9,150,opts0.nw);
        else
            fluxvar(:,i) = data.(L3opts.wmGF.fluxvarlist{i});
        end
    end

    % Including a season variable?
    if L3opts.wmGF.seasonvar
        expdata = [GFvar(:,vi) GFvar(:,seasoni) fluxvar];
        driver_fields = [GFmetvarlist(vi);'seasonSin';'seasonCos';L3opts.wmGF.fluxvarlist];
    else
        expdata = [GFvar(:,vi) fluxvar];
        driver_fields = [GFmetvarlist(vi); L3opts.wmGF.fluxvarlist];
    end

    % Separating day/night? Organize these variables separately
    if L3opts.wmGF.night_separate

        % find indices of met variables we want to use in the master matrix
        [~,vi] = ismember(L3opts.wmGF.night_metvarlist,GFmetvarlist);

        % Pull out flux variables used to gap-fill NEE
        il = length(L3opts.wmGF.night_fluxvarlist);
        fluxvar = NaN(size(data.wm,1),il);
        for i = 1:il
            % If chosen, gap fill the flux variables. Otherwise, record as-is
            if L3opts.wmGF.gffluxvar
                [~,fluxvar(:,i)] = De_spike3(data.(L3opts.wmGF.night_fluxvarlist{i}),9,150,opts0.nw);
            else
                fluxvar(:,i) = data.(L3opts.wmGF.night_fluxvarlist{i});
            end
        end

        % Including a season variable?
        if L3opts.wmGF.seasonvar
            expdata_night = [GFvar(:,vi) GFvar(:,seasoni) fluxvar];
            driver_fields_night = [GFmetvarlist(vi);'seasonSin';'seasonCos';L3opts.wmGF.night_fluxvarlist]; % NEW
        else
            expdata_night = [GFvar(:,vi) fluxvar];
            driver_fields_night = [GFmetvarlist(vi);L3opts.wmGF.night_fluxvarlist]; % NEW
        end
    end

    % Gap-fill, separating day/night if chosen
    if L3opts.wmGF.night_separate
        disp('Doing day-time data...')
        %[data.wm_gf(dayi,1),data.wm_ANN(dayi,1),wm_ANNext_day,data.wm_gf_iqr(dayi,[1 2]),data.wm_ANNc(dayi,1),ANNstats.wm_day] = ANNcode(expdata(dayi,:),data.wm(dayi,:),L3opts.wmGF.testarch,L3opts.wmGF.clmax,nmin,L3opts.wmGF.clund,L3opts.wmGF.Next,L3opts.wmGF.Ninit,L3opts.wmGF.f,L3opts.wmGF.extout);

        ANNstats.wm_day.driver_fields = driver_fields; % NEW
        % Get the scalings for the driver and prediction datasets for future
        % analysis. This has now been added to ANN_robust_full and ANN_robust_fast
        % and can be removed if/when the ANN is rerun.
        [~,ANNstats.wm_day.PSX] = mapminmax(expdata(dayi,:)'); % NEW
        [~,ANNstats.wm_day.PST] = mapminmax(data.wm(dayi,:)'); % NEW

        
        disp(' ')
        disp('Doing night-time data...')
        %[data.wm_gf(nighti,1),data.wm_ANN(nighti,1),wm_ANNext_night,data.wm_gf_iqr(nighti,[1 2]),data.wm_ANNc(nighti,1),ANNstats.wm_night] = ANNcode(expdata_night(nighti,:),data.wm(nighti,:),L3opts.wmGF.night_testarch,L3opts.wmGF.night_clmax,nmin_night,L3opts.wmGF.night_clund,L3opts.wmGF.night_Next,L3opts.wmGF.night_Ninit,L3opts.wmGF.night_f,L3opts.wmGF.night_extout);

        ANNstats.wm_night.driver_fields = driver_fields_night; % NEW
        % Get the scalings for the driver and prediction datasets for future
        % analysis. This has now been added to ANN_robust_full and ANN_robust_fast
        % and can be removed if/when the ANN is rerun.
        [~,ANNstats.wm_night.PSX] = mapminmax(expdata_night(nighti,:)'); % NEW
        [~,ANNstats.wm_night.PST] = mapminmax(data.wm(nighti,:)'); % NEW

    else
        disp('Doing all data (no day/night separation)...')
        %[data.wm_gf,data.wm_ANN,data.wm_ANNext,data.wm_gf_iqr,data.wm_ANNc,ANNstats.wm_all] = ANNcode(expdata,data.wm,L3opts.wmGF.testarch,L3opts.wmGF.clmax,nmin,L3opts.wmGF.clund,L3opts.wmGF.Next,L3opts.wmGF.Ninit,L3opts.wmGF.f,L3opts.wmGF.extout);

        ANNstats.wm_all.driver_fields = driver_fields; % NEW
        % Get the scalings for the driver and prediction datasets for future
        % analysis. This has now been added to ANN_robust_full and ANN_robust_fast
        % and can be removed if/when the ANN is rerun.
        [~,ANNstats.wm_all.PSX] = mapminmax(expdata'); % NEW
        [~,ANNstats.wm_all.PST] = mapminmax(data.wm'); % NEW
        

    
    end
end


%% Plot dataset of chosen flux and prep variables

% NOTE: The section above creates expdata (driver dataset) for the selected flux. 
% Everything else specific to each flux variable is saved within the
% respective ANNstats structure. The expdata is not, so be sure that the
% right one is being used if this code is integrated somewhere else.

if L3opts.([nameFlux 'GF']).night_separate
    
    % Choose the day or night ANN, as desired
    if 1 % Choosing daytime 
        disp('Analyzing daytime ANN')
        ANNstat = ANNstats.([nameFlux '_day']); % Daytime ANN
        drivers = expdata(dayi,:);
        ANNpred = data.([nameFlux '_ANN'])(dayi);
        ANNext = data.([nameFlux '_ANNext'])(dayi,:);
        ANNc = data.([nameFlux '_ANNc'])(dayi);
        flux = data.(nameFlux)(dayi);
        Mdate = data.Mdate(dayi);
    else % Choosing nighttime
        disp('Analyzing night-time ANN')
        ANNstat = ANNstats.([nameFlux '_night']); % Night-time ANN
        drivers = expdata_night(nighti,:);
        ANNpred = data.([nameFlux '_ANN'])(nighti);
        ANNext = data.([nameFlux '_ANNext'])(nighti,:);
        ANNc = data.([nameFlux '_ANNc'])(nighti);
        flux = data.(nameFlux)(nighti);
        Mdate = data.Mdate(nighti);
    end
else
    ANNstat = ANNstats.([nameFlux '_all']); % No day-night splitting
    drivers = expdata;
    ANNpred = data.([nameFlux '_ANN']);
    ANNext = data.([nameFlux '_ANNext']);
    ANNc = data.([nameFlux '_ANNc']);
    flux = data.(nameFlux);
    Mdate = data.Mdate;
end


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
plot(Mdate,flux,'k.',Mdate,ANNpred,'r.')
axis tight
set(gca,'xtick',xticks)
datetick('x',dfmt,'keeplimits','keepticks')    
legend('Measured','Predicted')


% Plot ANN performance
figure(2); clf
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

%% Analyze the ANN connection weights

numNets = numel(ANNstat.bestnet); % number of networks run (typically to gather uncertainty)
RI = NaN(numNets,numel(ANNstat.driver_fields)); % initialize connection weights output
for iNet = 1:numel(ANNstat.bestnet)
    [RI(iNet,:),~] = ANN_connection_weights(ANNstat.bestnet{iNet});
end
RImean = mean(RI,1);
RIstd = std(RI,0,1);

[~,isort] = sort(RImean,'descend');

% Bar plot!!
ANNvar = figure(3); clf
bar([1:size(RI,2)],RImean(isort)); hold on
errorbar(1:size(RI,2), RImean(isort),RIstd,'k.')
legend({'mean','std'})
ax = gca;
ax.XTickLabel = ANNstat.driver_fields(isort);
xlabel('Input variable')
ylabel('Relative variable importance (%)')
title(['Analysis of ANN Connection Weights predicting ' nameFlux])

%% Get response function for a particular driver cluster

% Choose values for the next 4 lines. Then run the section.
cluster = 1; % driver cluster to use (probably in the range of 1-15
driver_fields_vary = {'WD','TA'}; % Which drivers do you want to vary (enter 2)
nval = 10; % Number of values to test across the range of the drivers for this cluster 
plot_errorbars = 1; % Set to 1 in order to plot error bars on the response plots. Set to 0 to only plot the median. % Errorbars show median absolute deviation of the response across ANN replicates.

% % Use the clustering performed on the drivers to find reasonable
% combinations of drivers, and vary the drivers we want to test. Then run 
% the ANN at these driver values
[drivers_test] = ANN_vary_drivers_by_cluster(drivers,ANNstat.driver_fields,driver_fields_vary,ANNc,cluster,nval);
[response,driver_fields_varied] = ANN_response_function(ANNstat,drivers_test);

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

figure(4); clf
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
[drivers_median] = ANN_vary_drivers_by_cluster(drivers,ANNstat.driver_fields,[],ANNc,cluster,1);
driver_median_text = {'Median driver conditions:'};
for i = 1:numel(ANNstat.driver_fields)
    driver_median_text = [driver_median_text ; {[ANNstat.driver_fields{i} ': ',num2str(drivers_median.(ANNstat.driver_fields{i}))]}];
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
% minor, their variability tends not obscure the relationship too much.
% If a consistent relationship does not show, best to show the relationship
% for a single cluster using the section above.


% Choose values for the next 2 lines. Then run the section (takes several
% seconds)
driver_fields_vary = {'WD','TA'}; % Which drivers do you want to vary (enter 2)
nval = 10; % Number of values to test across the range of the drivers for this cluster 


% Create a colormap
cmap = colormap;
nc = size(colormap,1);

% Match the full range of the varying drivers to the colormap
n_vary = length(driver_fields_vary);
driver_fields_scale = cell(1,length(driver_fields_vary));
for i = 1:n_vary
    ifield = strmatch(driver_fields_vary{i},ANNstat.driver_fields);
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
    [drivers_test] = ANN_vary_drivers_by_cluster(drivers,ANNstat.driver_fields,driver_fields_vary,ANNc,cluster,nval);
    [response,driver_fields_varied] = ANN_response_function(ANNstat,drivers_test);

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