close all
clear all
clc

users = [10 50 100 200];
partitions = [.1 .25 .5 .75 1];
mu = 2.5;
timeout = 10000;
queue_length = 500;
average_local_dataset_length = 100;
S_h = 200e3;
S_t = 8*0.407e6;
sigma = 0.1e-4;
R_node = 1; % Mbps
R_p2p = 5;  % Mbps
M = 10;

num_rounds = 200;
upper_bound_accuray = 0.9594;
lower_bound_loss = 0.1352;

block_size = ceil(users.*partitions');
for k = 1 : length(users)
    for i = 1 : length(block_size)
        lambda(i,k) = sqrt( users(k) ./ ( ((S_h+S_t*block_size(i))/(R_node*1e6)) + ...
            (average_local_dataset_length*sigma) + (S_t/(R_node*1e6))) );
    end
end
lambda = round(lambda,1);

%% PART 1: DELAYS

% 1.1 s-FLchain
% Compute times involved in the transaction confirmation latency
T_c = average_local_dataset_length * sigma;
T_up = (S_h + S_t) / (R_node*1e6);
T_bp = (S_h + (S_t.*users)) / (R_p2p*1e6); 
p_fork_analytical = 1 - exp(-mu*(M-1).*T_bp);  
% Compute the iteration time in s-FLchain
T_iter_sync = T_c + T_up + (((1/mu) + T_bp) ./ (1-p_fork_analytical));

% 1.2 a-FLchain
T_iter_async = zeros(length(partitions), length(users));
for i = 1 : length(users)
    for j = 1 : length(partitions)
        bs = block_size(j,i);
        l = lambda(j,i);
        T_bp = (S_h + (S_t*bs)) / (R_p2p*1e6); 
        files_path = ['Outputs/output_queue_simulator/output_part2_25mbps_emnist/script_output_m' ...
            num2str(mu) '_l' num2str(l) '_s' num2str(bs) '.txt'];   
        % Iterate for each file in the directory
        file_data = fopen(files_path);
        A = textscan(file_data,'%s','Delimiter',';');
        B = str2double(A{:});    
        % Store results to variables
        queue_delay(j,i) = B(7);     
        p_fork_sim(j,i) = B(8);  
        T_iter_async(j,i) = (queue_delay(j,i) + T_bp) ...
            * (1/(1-p_fork_sim(j,i)));
        fclose(file_data);    
    end
end
% T = [T_iter_async(1:4,:); T_iter_sync];
T = [T_iter_async(1:5,:)];
bar(T')
legend({'a-FLchain (10%)','a-FLchain (25%)','a-FLchain (50%)',...
    'a-FLchain (75%)','s-FLchain'})
xlabel('Num. users, K')
ylabel('FL round time (s)')
xticks(1:4)
xticklabels([10, 50, 100, 200])
set(gca, 'fontsize', 14)
grid on
grid minor
    
%%
iteration_time_per_approach = T';%[T_iter_async  T_iter_sync'];
root_folder_iid = 'results_emnist/num_classes_10_cnn';
root_folder_noniid = 'results_emnist/num_classes_3_cnn';

% EVAL. ACCURACY
results_iid = [];
results_non_iid = [];
for k = 1 : length(users)
    for p = 1 : length(partitions)
        file_path = ['Outputs/output_tensorflow/' root_folder_iid '/CNN_eval_accuracy_K' ...
            num2str(users(k)) '_' num2str(partitions(p)) '.txt'];
        lgd{k} = strcat('K=',num2str(users(k))) ;
        fid = fopen(file_path);
        tline = fgetl(fid);
        B = [];
        while ischar(tline)
            A = textscan(tline, '%s', 'Delimiter', ' ');
            B = [B str2double(A{:})];    
            tline = fgetl(fid);
        end
        results_iid = [results_iid B(1:200)];
        fclose(fid);   
        file_path = ['Outputs/output_tensorflow/' root_folder_noniid '/CNN_eval_accuracy_K' ...
            num2str(users(k)) '_' num2str(partitions(p)) '.txt'];
        lgd{k} = strcat('K=',num2str(users(k))) ;
        fid = fopen(file_path);
        tline = fgetl(fid);
        C = [];
        while ischar(tline)
            A = textscan(tline, '%s', 'Delimiter', ' ');
            C = [C str2double(A{:})];    
            tline = fgetl(fid);
        end
        results_non_iid = [results_non_iid C(1:200)];
        fclose(fid);   
    end
end

for k = 1 : length(users) 
    disp([' * Num. users = ' num2str(users(k))])
    for k2 = 1 : length(partitions)
        disp(['     - \Upsilon = ' num2str(partitions(k2))])
        acc_per_time_iid = mean(results_iid(1:200,k2)/iteration_time_per_approach(k,k2));
        acc_per_time_noniid = mean(results_non_iid(1:200,k2)/iteration_time_per_approach(k,k2));
        disp(['        -> accuracy/s = ' num2str(round(acc_per_time_iid,3)) '/'...
            num2str(round(acc_per_time_noniid,3))])
    end
end