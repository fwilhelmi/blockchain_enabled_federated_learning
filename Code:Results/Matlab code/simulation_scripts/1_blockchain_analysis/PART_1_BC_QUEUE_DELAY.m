%%% *********************************************************************
%%% * Blockchain-Enabled RAN Sharing for Future 5G/6G Communications    *
%%% * Authors: Lorenza Giupponi & Francesc Wilhelmi (fwilhelmi@cttc.cat)*
%%% * Copyright (C) 2020-2025, and GNU GPLd, by Francesc Wilhelmi       *
%%% * GitHub repository: ...                                            *
%%% *********************************************************************
clear
clc

FORKS_ENABLED = 1;
lambda = [0.02 0.2 2 20];
mu = [0.01 0.05 0.1 0.2 0.3 0.5 1]; %0.1:0.1:0.5;
timeout = 10;
queue_length = 1000;
block_size = 1:50;

total_transactions = cell(1,length(mu));
transactions_dropped = cell(1,length(mu));
drop_percentage = cell(1,length(mu));
num_blocks_mined_by_timeout = cell(1,length(mu));
mean_occupancy = cell(1,length(mu));
mean_delay = cell(1,length(mu));    
p_fork_sim = cell(1,length(mu));
for i = 1 : length(mu)
    total_transactions{i} = zeros(length(lambda), length(block_size));
    transactions_dropped{i} = zeros(length(lambda), length(block_size));
    drop_percentage{i} = zeros(length(lambda), length(block_size));
    num_blocks_mined_by_timeout{i} = zeros(length(lambda), length(block_size));
    mean_occupancy{i} = zeros(length(lambda), length(block_size));
    mean_delay{i} = zeros(length(lambda), length(block_size));    
    p_fork_sim{i} = zeros(length(lambda), length(block_size));
end

%files_path = ['output_simulator_no_timer'];
files_path = ['Outputs/output_queue_simulator/output_5mbps_10miners'];
files_dir = dir([files_path '/*.txt']);
% Iterate for each file in the directory
for i = 1 : length(files_dir)
    % Get the name of the file being analyzed
    file_name = files_dir(i).name;
    % Find the parameters used (timer, lambda & block size)
    % - mu
    split1 = strsplit(file_name,'_');
    ix = 3;
    split2 = strsplit(split1{ix},'m');
    m = str2double(split2{2});
    ix_m = find(mu==m);
    % - Lambda
    split1 = strsplit(file_name,'_');
    ix = 4;
    %if FORKS_ENABLED, ix = ix+1; end
    split2 = strsplit(split1{ix},'l');
    l = str2double(split2{2});
    ix_l = find(lambda==l);
    % - Block size
    ix = 5;
    %if FORKS_ENABLED, ix = ix+1; end
    split3 = strsplit(split1{ix},'s');
    split4 = strsplit(split3{2},'.');
    s = str2double(split4{1});
    ix_s = find(block_size==s);
    % Read the file
    if isempty(ix_m) || isempty(ix_l) || isempty(ix_s)
        % Skip this file
    else
        file_data = fopen([files_path '/' file_name]);
        A = textscan(file_data,'%s','Delimiter',';');
        B = str2double(A{:});    
        % Store results to variables
        total_transactions{ix_m}(ix_l,ix_s) = B(2);
        transactions_dropped{ix_m}(ix_l,ix_s) = B(3);
        drop_percentage{ix_m}(ix_l,ix_s) = B(4);
        num_blocks_mined_by_timeout{ix_m}(ix_l,ix_s) = B(5);
        mean_occupancy{ix_m}(ix_l,ix_s) = B(6)/queue_length * 100;
        mean_delay{ix_m}(ix_l,ix_s) = B(7);     
        if FORKS_ENABLED
            p_fork_sim{ix_m}(ix_l,ix_s) = B(8);  
        end
        fclose(file_data);
    end
end

%% 
%%%%%%%%%%% 
% PLOTS
%%%%%%%%%%% 

%% Mean occupancy vs delay
for i = 1 : length(mu)
    data(i) = mean(mean_occupancy{i}(:));
    stddata(i) = std(mean_occupancy{i}(:));
    data2(i) = mean(mean_delay{i}(:));
    stddata2(i) = std(mean_delay{i}(:));
    data3(i) = mean(p_fork_sim{i}(:));
end
figure
plot(1:length(data2), data2, '--o', 'linewidth', 2.0, 'markersize', 10)
ylabel('Delay (s)')
yyaxis right
plot(1:length(data), data, '--x', 'linewidth', 2.0, 'markersize', 10)
hold on
plot(1:length(data3), data3.*100, '-.s', 'linewidth', 2.0, 'markersize', 10)
ylabel('Occupancy (%) / Fork prob. (%)')
xticks([1:length(mu)])
xticklabels(mu)
grid on
grid minor
set(gca,'fontsize',16)
xlabel('Block generation ratio, \lambda (Hz)')
legend({'Delay', 'Occupancy', 'p_{fork}'})

%% Boxplots delay - Block size
ixes = [1 4 7];
for i = 1 : length(ixes)
    f = figure;
    f.Position = [10 10 450 300]; 
    boxplot(mean_delay{ixes(i)})
    ylabel('Queue delay (s)')
    yyaxis right
    plot(mean(p_fork_sim{ixes(i)}), 'r--x', 'linewidth', 2.0)
    ylabel('Fork probability (%)')
    grid on
    grid minor
    xticks(1:2:50)
    xticklabels([1:2:50])
    set(gca,'fontsize',12)    
    xlabel('Block size')
    title({'\lambda = ' num2str(mu(ixes(i)))})
end

%% Bar plot delay vs block size (lambda = 0.01)
f = figure;
%subplot(2,1,1)
b = bar(mean_delay{2}(:,1:20)','stacked');
%axis([0 21 0 4000])
b(1).FaceColor = [0.36,0.59,0.05];
b(2).FaceColor = [1.00,0.37,0.37];
b(3).FaceColor = [1.00,0.70,0.00];
b(4).FaceColor = [0.07,0.60,0.94];
ylabel('Queue delay, \delta_q (s)')
yyaxis right
plot(drop_percentage{2}(1,1:20)', '-.x', 'color', [0.36,0.59,0.05], 'linewidth', 1.0, 'markersize', 12)
hold on
plot(drop_percentage{2}(2,1:20)', '-.s', 'color', [1.00,0.37,0.37], 'linewidth', 1.0, 'markersize', 12)
plot(drop_percentage{2}(3,1:20)', '-.o', 'color', [1.00,0.70,0.00], 'linewidth', 1.0, 'markersize', 10)
plot(drop_percentage{2}(4,1:20)', '-.d', 'color', [0.07,0.60,0.94], 'linewidth', 1.0, 'markersize', 10)
ylabel('Packet drop rate')
grid on
grid minor
set(gca,'fontsize',16)
xlabel('Block size, S_B (# of transactions)')
% legend({'Delay (\nu=0.02)','Delay (\nu=0.2)','Delay (\nu=2)','Delay (\nu=20)',...
%     'Drop rate (\nu=0.02)','Drop rate (\nu=0.2)','Drop rate (\nu=2)','Drop rate (\nu=20)'})


%% Bar plot delay vs block size (lambda = 0.2)
f = figure;
%subplot(2,1,2)
b = bar(mean_delay{4}(:,1:20)','stacked');
%axis([0 21 0 4000])
b(1).FaceColor = [0.36,0.59,0.05];
b(2).FaceColor = [1.00,0.37,0.37];
b(3).FaceColor = [1.00,0.70,0.00];
b(4).FaceColor = [0.04,0.48,0.78];
ylabel('Queue delay, \delta_q')
yyaxis right
plot(drop_percentage{4}(1,1:20)', '-.x', 'color', [0.36,0.59,0.05], 'linewidth', 1.0, 'markersize', 12)
hold on
plot(drop_percentage{4}(2,1:20)', '-.s', 'color', [1.00,0.37,0.37], 'linewidth', 1.0, 'markersize', 12)
plot(drop_percentage{4}(3,1:20)', '-.o', 'color', [1.00,0.70,0.00], 'linewidth', 1.0, 'markersize', 10)
plot(drop_percentage{4}(4,1:20)', '-.d', 'color', [0.04,0.48,0.78], 'linewidth', 1.0, 'markersize', 10)
ylabel('Packet drop rate')
grid on
grid minor
set(gca,'fontsize',16)
xlabel('Block size, S_B (# of transactions)')
legend({'Delay (\nu=0.02)','Delay (\nu=0.2)','Delay (\nu=2)','Delay (\nu=20)',...
    'Drop rate (\nu=0.02)','Drop rate (\nu=0.2)','Drop rate (\nu=2)','Drop rate (\nu=20)'})

%% Bar plot delay vs block size (lambda = 1)
f = figure;
b = bar(mean_delay{7}(:,1:20)','stacked');
b(1).FaceColor = [0.36,0.59,0.05];
b(2).FaceColor = [1.00,0.37,0.37];
b(3).FaceColor = [1.00,0.70,0.00];
b(4).FaceColor = [0.07,0.60,0.94];
ylabel('Delay (s)')
yyaxis right
plot(drop_percentage{7}(1,1:20)', '-.x', 'color', [0.36,0.59,0.05], 'linewidth', 1.0, 'markersize', 12)
hold on
plot(drop_percentage{7}(2,1:20)', '-.s', 'color', [1.00,0.37,0.37], 'linewidth', 1.0, 'markersize', 12)
plot(drop_percentage{7}(3,1:20)', '-.o', 'color', [1.00,0.70,0.00], 'linewidth', 1.0, 'markersize', 10)
plot(drop_percentage{7}(4,1:20)', '-.d', 'color', [0.07,0.60,0.94], 'linewidth', 1.0, 'markersize', 10)
ylabel('Packet drop rate')
grid on
grid minor
set(gca,'fontsize',16)
xlabel('Block size, S_B (# of transactions)')
legend({'Delay (\nu=0.02)','Delay (\nu=0.2)','Delay (\nu=2)','Delay (\nu=20)',...
    'Drop rate (\nu=0.02)','Drop rate (\nu=0.2)','Drop rate (\nu=2)','Drop rate (\nu=20)'})

%% Bar plot delay (nu = 0.02)
figure
d1 = mean_delay{1}(1,:);
d2 = mean_delay{4}(1,:);
d3 = mean_delay{7}(1,:);
bar([d1' d2' d3'], 'stacked')
xlabel('Block size')
ylabel('Delay (s)')
title('\nu = 0.02')
legend({'\lambda=0.1','\lambda=0.2','\lambda=0.3','\lambda=0.4'})
grid on
grid minor
set(gca,'fontsize',14)

%% Bar plot delay (nu = 0.2)
figure
d1 = mean_delay{1}(2,:);
d2 = mean_delay{4}(2,:);
d3 = mean_delay{7}(2,:);
bar([d1' d2' d3'], 'stacked')
xlabel('Block size')
ylabel('Delay (s)')
title('\nu = 0.2')
legend({'\lambda=0.1','\lambda=0.2','\lambda=0.3','\lambda=0.4'})
grid on
grid minor
set(gca,'fontsize',14)

%% Bar plot delay (nu = 2)
figure
d1 = mean_delay{1}(3,:);
d2 = mean_delay{4}(3,:);
d3 = mean_delay{7}(3,:);
bar([d1' d2' d3'], 'stacked')
xlabel('Block size')
ylabel('Delay (s)')
title('\nu = 2')
legend({'\lambda=0.1','\lambda=0.2','\lambda=0.3','\lambda=0.4'})
grid on
grid minor
set(gca,'fontsize',14)

%% Bar plot delay (nu = 20)
figure
d1 = mean_delay{1}(4,:);
d2 = mean_delay{4}(4,:);
d3 = mean_delay{7}(4,:);
bar([d1' d2' d3'], 'stacked')
xlabel('Block size')
ylabel('Delay (s)')
title('\nu = 20')
legend({'\lambda=0.1','\lambda=0.2','\lambda=0.3','\lambda=0.4'})
grid on
grid minor
set(gca,'fontsize',14)

%% Plot S_B vs Queue delay
figure
plot(mean_delay{2}(1,1:50)', '-.', 'linewidth', 2.0)
hold on
plot(mean_delay{4}(1,1:50)', '-.', 'linewidth', 2.0)
plot(mean_delay{7}(1,1:50)', '-.', 'linewidth', 2.0)
plot(mean_delay{2}(4,1:50)', '--', 'linewidth', 2.0)
plot(mean_delay{4}(4,1:50)', '--', 'linewidth', 2.0)
plot(mean_delay{7}(4,1:50)', '--', 'linewidth', 2.0)
grid on
grid minor
set(gca,'fontsize',16)
axis([1 50 0 2000])
ylabel('Queue delay, \delta_q')
xlabel('Block size, S_B (# of transactions)')

legend({'\nu = 0.2 tps, \lambda = 0.05 Hz','\nu = 0.2 tps, \lambda = 0.2 Hz',...
    '\nu = 0.2 tps, \lambda = 1 Hz', ...
    '\nu = 20 tps, \lambda = 0.05 Hz','\nu = 20 tps, \lambda = 0.2 Hz',...
    '\nu = 20 tps, \lambda = 1 Hz'})

%% Save workspace
save('tmp/simulator_output')