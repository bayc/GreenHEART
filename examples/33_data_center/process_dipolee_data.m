%%
% SETUP
% ------------------------------------------------------------
tic;
close all;
clear;
clc;
format bank;
restoredefaultpath;

% supress warnings  
warning('off','all');

% Process calibrations
DataPathname_Source1 = 'C:\Users\v0y\OneDrive - Oak Ridge National Laboratory\3. Projects\4.6 HFTO - misc\4. HFTO - SI - DC\Phase 1\Task 2 – DC Load\Samples\NLR\03_whole-facility_profiles\colocation\simulated_data\';
DataFilename_Source1_1 = 'colocation_10MW_2840nodes_20u_power.xlsx';
DataFilename_Source1_2 = 'colocation_10MW_2840nodes_40u_power.xlsx';
DataFilename_Source1_3 = 'colocation_10MW_2840nodes_60u_power.xlsx';
DataFilename_Source1_4 = 'colocation_10MW_2840nodes_80u_power.xlsx';

DataPathname_Source2 = 'C:\Users\v0y\OneDrive - Oak Ridge National Laboratory\3. Projects\4.6 HFTO - misc\4. HFTO - SI - DC\Phase 1\Task 2 – DC Load\Samples\NLR\03_whole-facility_profiles\inference\simulated_data\';
DataFilename_Source2_1 = 'inference_1MW_283nodes_20u_power.xlsx';
DataFilename_Source2_2 = 'inference_1MW_283nodes_40u_power.xlsx';
DataFilename_Source2_3 = 'inference_1MW_283nodes_60u_power.xlsx';
DataFilename_Source2_4 = 'inference_1MW_283nodes_80u_power.xlsx';

DataPathname_Source3 = 'C:\Users\v0y\OneDrive - Oak Ridge National Laboratory\3. Projects\4.6 HFTO - misc\4. HFTO - SI - DC\Phase 1\Task 2 – DC Load\Samples\ORNL\';
DataFilename_Source3 = '87F42G2C+2QG3-124-131-124-131.xlsx';

datasets = {
    'Power1_1MW', 'Power1_1', 1;
    'Power1_2MW', 'Power1_2', 4;
    'Power1_3MW', 'Power1_3', 7;
    'Power1_4MW', 'Power1_4', 10;
    'Power2_1MW', 'Power2_1', 2;
    'Power2_2MW', 'Power2_2', 5;
    'Power2_3MW', 'Power2_3', 8;
    'Power2_4MW', 'Power2_4', 11;
    'Power3_1MW', 'Power3_1', 3;
    'Power3_2MW', 'Power3_2', 6;
};

disp(['1. Setup complete ...' num2str(toc)]);

%% 
% LOAD DC RAW DATA
% ------------------------------------------------------------

Data_DC.All1_1 = readtable(strcat(DataPathname_Source1,DataFilename_Source1_1));
Data_DC.Weekly.Power1_1MW = table2array(Data_DC.All1_1(:,9));
Data_DC.All1_2 = readtable(strcat(DataPathname_Source1,DataFilename_Source1_2));
Data_DC.Weekly.Power1_2MW = table2array(Data_DC.All1_2(:,9));
Data_DC.All1_3 = readtable(strcat(DataPathname_Source1,DataFilename_Source1_3));
Data_DC.Weekly.Power1_3MW = table2array(Data_DC.All1_3(:,9));
Data_DC.All1_4 = readtable(strcat(DataPathname_Source1,DataFilename_Source1_4));
Data_DC.Weekly.Power1_4MW = table2array(Data_DC.All1_4(:,9));

Data_DC.All2_1 = readtable(strcat(DataPathname_Source2,DataFilename_Source2_1));
Data_DC.Weekly.Power2_1MW = table2array(Data_DC.All2_1(:,9));
Data_DC.All2_2 = readtable(strcat(DataPathname_Source2,DataFilename_Source2_2));
Data_DC.Weekly.Power2_2MW = table2array(Data_DC.All2_2(:,9));
Data_DC.All2_3 = readtable(strcat(DataPathname_Source2,DataFilename_Source2_3));
Data_DC.Weekly.Power2_3MW = table2array(Data_DC.All2_3(:,9));
Data_DC.All2_4 = readtable(strcat(DataPathname_Source2,DataFilename_Source2_4));
Data_DC.Weekly.Power2_4MW = table2array(Data_DC.All2_4(:,9));

Data_DC.All3_1 = readtable(strcat(DataPathname_Source3,DataFilename_Source3),'Sheet','Data Set 1');
Data_DC.Weekly.Power3_1MW = table2array(Data_DC.All3_1(:,11));
Data_DC.All3_2 = readtable(strcat(DataPathname_Source3,DataFilename_Source3),'Sheet','Data Set 2');
Data_DC.Weekly.Power3_2MW = table2array(Data_DC.All3_2(:,2));

% Data_DC.Time_mins = (1 : length(Data_DC.1_1Power_MW))';

disp(['2. Data loading (Raw) complete ...' num2str(toc)]); 

%% 
% FORMAT AND TRIM DATA
% ------------------------------------------------------------

minutes_per_day = 1440;
minutes_per_week = 10080;

for k = 1:size(datasets,1)

    raw_name = datasets{k,1};
    proc_name = char(datasets{k,2});

    power = Data_DC.Weekly.(raw_name);

    % --- reshape ---
    nD = floor(length(power)/minutes_per_day);
    nW = floor(length(power)/minutes_per_week);

    p_day = reshape(power(1:nD*minutes_per_day), minutes_per_day, nD);
    p_week = reshape(power(1:nW*minutes_per_week), minutes_per_week, nW);

    % --- compute ---
    Data_DC.Daily.(proc_name)  = compute_percentiles(p_day);
    Data_DC.Weekly.(proc_name) = compute_percentiles(p_week);

end

% time axis
t_weekly = (0:minutes_per_week-1) / 1440; % days (0–7)
t_daily = (0:minutes_per_day-1) / 60; % hours (0-23)

disp(['3. Data formating and trimming complete...' num2str(toc)]); 

%% 
% PLOT WITH SHARED BANDS
% ------------------------------------------------------------
close all;

%-------------- Figure 1 ---------------
fig = figure(1); clf; fig.WindowState = 'maximized';
tl = tiledlayout(4,3,'TileSpacing','compact','Padding','compact');

for k = 1:size(datasets,1)

    proc_name = datasets{k,2};
    tile_id   = datasets{k,3};

    nexttile(tile_id); hold on; grid on;

    S = Data_DC.Daily.(proc_name);

    H = plot_percentile_panel(t_daily, S);

    if k == 1
        H_legend = H;
    end

    xlabel('Hour of Day');
    ylabel('Power (MW)');
    title(proc_name);

    xticks(0:1:24);
end

lgd = legend(H_legend, {'50%','25–75%','10–90%','0–100%'}, 'Orientation','horizontal');
lgd.Layout.Tile = 'south';

%-------------- Figure 2 ---------------
fig = figure(2); clf; fig.WindowState = 'maximized';
tl = tiledlayout(4,3,'TileSpacing','compact','Padding','compact');

for k = 1:size(datasets,1)

    proc_name = datasets{k,2};
    tile_id   = datasets{k,3};

    nexttile(tile_id); hold on; grid on;

    S = Data_DC.Weekly.(proc_name);

    H = plot_percentile_panel(t_weekly, S);

    if k == 1
        H_legend = H;
    end

    xlabel('Day of Week');
    ylabel('Power (MW)');
    title(proc_name);

    xticks(0:1:7);
    xticklabels({'Mon','Tue','Wed','Thu','Fri','Sat','Sun','Mon'});
end

lgd = legend(H_legend, {'50%','25–75%','10–90%','0–100%'}, 'Orientation','horizontal');
lgd.Layout.Tile = 'south';


disp(['4. Plots complete ...' num2str(toc)]);

%% 
% Helper function
% 
function S = compute_percentiles(X)
    S.p50  = prctile(X,50,2);
    S.p25  = prctile(X,25,2);
    S.p75  = prctile(X,75,2);
    S.p10  = prctile(X,10,2);
    S.p90  = prctile(X,90,2);
    S.p0   = prctile(X,0,2);
    S.p100 = prctile(X,100,2);
end

function H = plot_percentile_panel(t, S)

    h1 = fill([t fliplr(t)], [S.p0' fliplr(S.p100')], ...
        [0.9 0.9 0.9], 'EdgeColor','none');

    h2 = fill([t fliplr(t)], [S.p10' fliplr(S.p90')], ...
        [0.75 0.85 0.95], 'EdgeColor','none');

    h3 = fill([t fliplr(t)], [S.p25' fliplr(S.p75')], ...
        [0.5 0.7 0.9], 'EdgeColor','none');

    h4 = plot(t, S.p50, 'b', 'LineWidth', 1);

    H = [h4 h3 h2 h1]; % legend order
end