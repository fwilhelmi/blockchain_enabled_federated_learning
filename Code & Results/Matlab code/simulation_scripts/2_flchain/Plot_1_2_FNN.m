close all
clear all
clc

users = [10 50 100 200];
partitions = [.1 .25 .5 .75 1];

num_rounds = 200;
upper_bound_accuray = 0.9594;

root_folder_iid = 'Outputs/output_tensorflow/results_emnist/num_classes_10_fnn';
root_folder_noniid = 'Outputs/output_tensorflow/results_emnist/num_classes_3_fnn';

%% Get the results from the simulation outputs
results_bar_iid = cell(1, length(users));
results_bar_noniid = cell(1, length(users));
for k = 1 : length(users)      
    results_iid = [];
    results_noniid = [];
    for p = 1 : length(partitions)
        % IID
        file_path = [root_folder_iid '/eval_accuracy_K' ...
            num2str(users(k)) '_' num2str(partitions(p)) '.txt'];
        lgd{k} = strcat('K=',num2str(users(k))) ;
        fid = fopen(file_path);
        tline = fgetl(fid);
        B = [];
        results_bar_iid{k} = [];
        while ischar(tline)
            A = textscan(tline, '%s', 'Delimiter', ' ');
            B = [B str2double(A{:})];    
            tline = fgetl(fid);
        end
        results_iid = [results_iid B];
        results_bar_iid{k} = [results_bar_iid{k} results_iid];
        fclose(fid);        
        % Non-IID
        file_path = [root_folder_noniid '/eval_accuracy_K' ...
            num2str(users(k)) '_' num2str(partitions(p)) '.txt'];
        lgd{k} = strcat('K=',num2str(users(k))) ;
        fid = fopen(file_path);
        tline = fgetl(fid);
        B = [];
        results_bar_noniid{k} = [];
        while ischar(tline)
            A = textscan(tline, '%s', 'Delimiter', ' ');
            B = [B str2double(A{:})];    
            tline = fgetl(fid);
        end
        results_noniid = [results_noniid B];
        results_bar_noniid{k} = [results_bar_noniid{k} results_noniid];
        fclose(fid);          
    end
end

%% Plot the mean accuracy achieved in the last 50 iterations
f11 = figure;
mean_bar_data = [mean(results_bar_iid{1}(150:200,:)); mean(results_bar_iid{2}(150:200,:)); ...
    mean(results_bar_iid{3}(150:200,:)); mean(results_bar_iid{4}(150:200,:))];
% std_bar_data = [std(results_bar{1}(150:200,:)); std(results_bar{2}(150:200,:)); ...
%     std(results_bar{3}(150:200,:)); std(results_bar{4}(150:200,:))];
bar(mean_bar_data,'LineStyle',':','LineWidth',1.5,'FaceAlpha',0.3);
hold on
mean_bar_data_noniid = [mean(results_bar_noniid{1}(150:200,:)); mean(results_bar_noniid{2}(150:200,:)); ...
    mean(results_bar_noniid{3}(150:200,:)); mean(results_bar_noniid{4}(150:200,:))];
std_bar_data_noniid = [std(results_bar_noniid{1}(150:200,:)); std(results_bar_noniid{2}(150:200,:)); ...
    std(results_bar_noniid{3}(150:200,:)); std(results_bar_noniid{4}(150:200,:))];
b1 = bar(mean_bar_data_noniid);
ngroups = size(mean_bar_data, 1);
nbars = size(mean_bar_data, 2);
% Calculating the width for each bar group
% groupwidth = min(0.8, nbars/(nbars + 1.5));
% for i = 1:nbars
%     x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
%     errorbar(x, mean_bar_data_noniid(:,i), std_bar_data_noniid(:,i), 'r.', 'linewidth', 1);
% end
grid on
grid minor
c = plot(0:5, upper_bound_accuray*ones(1,6), 'k--','linewidth',2.0);
ylabel('Mean eval. accuracy')
xlabel('Num. clients, K')
legend([b1 c], {'a-FLchain (\Upsilon=10%)','a-FLchain (\Upsilon=25%)',...
    'a-FLchain (\Upsilon=50%)','a-FLchain (\Upsilon=5%)',...
    's-FLchain','Centr.'})
set(gca, 'fontsize', 16)
% save_figure( f11, 'mean_eval_accuracy', 'figures/' )
