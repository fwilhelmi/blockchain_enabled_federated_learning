close all
clear all
clc

users = [10 50 100 200];
partitions = [.1 .25 .5 .75 1];
%lambda = 10;
mu = 2.5;
timeout = 10000;
queue_length = 500;
average_local_dataset_length = 100;
S_h = 200e3;
S_t = 5e3;
sigma = 0.1e-4;
R_node = 1; % Mbps
R_p2p = 5;  % Mbps
M = 10;

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
        files_path = ['Outputs/output_queue_simulator/output_part2_25mbps/script_output_m' ...
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
T = [T_iter_async(1:4,:); T_iter_sync];
bar(T')
legend({'Async (10%)','Async (25%)','Async (50%)','Async (75%)','Sync','Centr.'})
xlabel('Num. users, K')
ylabel('FL round time (s)')
xticks(1:4)
xticklabels([10, 50, 100, 200])
set(gca, 'fontsize', 16)
grid on
grid minor
    
%%
iteration_time_per_approach = T';%[T_iter_async  T_iter_sync'];

num_rounds = 200;
upper_bound_accuray = 0.9594;
lower_bound_loss = 0.1352;

root_folder = 'sync_vs_async/num_classes_10';

% EVAL. ACCURACY
f = figure;
f.Position = [300 300 780 200];
for k = 1 : length(users)  
    results = [];
    for p = 1 : length(partitions)
        file_path = ['Outputs/output_tensorflow/' root_folder '/eval_accuracy_K' ...
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
        results = [results B];
        fclose(fid);    
    end
    subplot(1,length(users),k)
    for k2 = 1 : length(partitions)
        plot((1:num_rounds).*iteration_time_per_approach(k,k2),results(:,k2),'linewidth',2.0)
        hold on
    end    
    grid on
    grid minor
    plot((1:num_rounds) .* max(iteration_time_per_approach(k,:)), ...
        upper_bound_accuray*ones(1,num_rounds), 'k--','linewidth',2.0)
    title(['K = ' num2str(users(k))])
    ylabel('Eval accuracy')
    xlabel('Time (s)')
    set(gca, 'fontsize', 16)
end
legend({'Async (10%)','Async (25%)','Async (50%)','Async (75%)','Sync','Centr.'})
save_figure( f, 'eval_accuracy', 'figures/' )

% EVAL. LOSS
f2 = figure;
f2.Position = [300 300 780 200];
for k = 1 : length(users)  
    results = [];
    for p = 1 : length(partitions)
        file_path = ['Outputs/output_tensorflow/' root_folder '/eval_loss_K' ...
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
        results = [results B];
        fclose(fid);    
    end
    subplot(1,length(users),k)
    for k2 = 1 : length(partitions)
        plot((1:num_rounds).*iteration_time_per_approach(k,k2),results(:,k2),'linewidth',2.0)
        hold on
    end    
    grid on
    grid minor
    plot((1:num_rounds).* max(iteration_time_per_approach(k,:)),...
        lower_bound_loss*ones(1,num_rounds), 'k--','linewidth',2.0)
    title(['K = ' num2str(users(k))])
    ylabel('Eval loss')
    xlabel('Time (s)')
    set(gca, 'fontsize', 16)
    axis([0 200*max(iteration_time_per_approach(k,:)) 0 4])
end
legend({'Async (10%)','Async (25%)','Async (50%)','Async (75%)','Sync','Centr.'})
save_figure( f2, 'eval_loss', 'figures/' )

%% OTHER PLOTS THAT HAVE NOT BEEN INCLUDED (COMMENTED)
% %% TRAIN ACCURACY
% f3 = figure;
% f3.Position = [300 300 780 200];
% for k = 1 : length(users)  
%     results = [];
%     for p = 1 : length(partitions)
%         file_path = ['output_tensorflow/' root_folder '/train_accuracy_K' ...
%             num2str(users(k)) '_' num2str(partitions(p)) '.txt'];
%         lgd{k} = strcat('K=',num2str(users(k))) ;
%         fid = fopen(file_path);
%         tline = fgetl(fid);
%         B = [];
%         while ischar(tline)
%             A = textscan(tline, '%s', 'Delimiter', ' ');
%             B = [B str2double(A{:})];    
%             tline = fgetl(fid);
%         end
%         results = [results B];
%         fclose(fid);    
%     end
%     subplot(1,length(users),k)
%     plot(results,'linewidth',2.0)
%     hold on
%     grid on
%     grid minor
%     plot(1:num_rounds, upper_bound_accuray*ones(1,num_rounds), 'k--','linewidth',2.0)
%     title(['K = ' num2str(users(k))])
%     ylabel('Train accuracy')
%     xlabel('Num. rounds')
%     set(gca, 'fontsize', 16)
% end
% legend({'Async (10%)','Async (25%)','Async (50%)','Async (75%)','Sync','Centr.'})
% save_figure( f3, 'train_accuracy', 'figures/' )
% 
% %% TRAIN LOSS
% f4 = figure;
% f4.Position = [300 300 780 200];
% for k = 1 : length(users)  
%     results = [];
%     for p = 1 : length(partitions)
%         file_path = ['output_tensorflow/' root_folder '/train_loss_K' ...
%             num2str(users(k)) '_' num2str(partitions(p)) '.txt'];
%         lgd{k} = strcat('K=',num2str(users(k))) ;
%         fid = fopen(file_path);
%         tline = fgetl(fid);
%         B = [];
%         while ischar(tline)
%             A = textscan(tline, '%s', 'Delimiter', ' ');
%             B = [B str2double(A{:})];    
%             tline = fgetl(fid);
%         end
%         results = [results B];
%         fclose(fid);    
%     end
%     subplot(1,length(users),k)
%     plot(results,'linewidth',2.0)
%     hold on
%     grid on
%     grid minor
%     plot(1:num_rounds, lower_bound_loss*ones(1,num_rounds), 'k--','linewidth',2.0)
%     title(['K = ' num2str(users(k))])
%     ylabel('Train loss')
%     xlabel('Num. rounds')
%     set(gca, 'fontsize', 16)
% end
% legend({'Async (10%)','Async (25%)','Async (50%)','Async (75%)','Sync','Centr.'})
% save_figure( f4, 'train_loss', 'figures/tensorflow/' )
% 
% %% TEST ACCURACY
% f5 = figure;
% f5.Position = [300 300 780 200];
% for k = 1 : length(users)  
%     results = [];
%     for p = 1 : length(partitions)
%         file_path = ['output_tensorflow/' root_folder '/test_accuracy_K' ...
%             num2str(users(k)) '_' num2str(partitions(p)) '.txt'];
%         lgd{k} = strcat('K=',num2str(users(k))) ;
%         fid = fopen(file_path);
%         tline = fgetl(fid);
%         B = [];
%         while ischar(tline)
%             A = textscan(tline, '%s', 'Delimiter', ' ');
%             B = [B str2double(A{:})];    
%             tline = fgetl(fid);
%         end
%         results = [results B];
%         fclose(fid);    
%     end
%     subplot(1,length(users),k)
%     plot(results,'linewidth',2.0)
%     hold on
%     grid on
%     grid minor
%     plot(1:num_rounds, upper_bound_accuray*ones(1,num_rounds), 'k--','linewidth',2.0)
%     title(['K = ' num2str(users(k))])
%     ylabel('Test accuracy')
%     xlabel('Num. rounds')
%     set(gca, 'fontsize', 16)
% end
% legend({'Async (10%)','Async (25%)','Async (50%)','Async (75%)','Sync','Centr.'})
% save_figure( f5, 'test_accuracy', 'figures/tensorflow/' )
% 
% %% TEST LOSS
% f6 = figure;
% f6.Position = [300 300 780 200];
% for k = 1 : length(users)  
%     results = [];
%     for p = 1 : length(partitions)
%         file_path = ['output_tensorflow/' root_folder '/test_loss_K' ...
%             num2str(users(k)) '_' num2str(partitions(p)) '.txt'];
%         lgd{k} = strcat('K=',num2str(users(k))) ;
%         fid = fopen(file_path);
%         tline = fgetl(fid);
%         B = [];
%         while ischar(tline)
%             %disp(tline)
%             A = textscan(tline, '%s', 'Delimiter', ' ');
%             B = [B str2double(A{:})];    
%             tline = fgetl(fid);
%         end
%         results = [results B];
%         fclose(fid);    
%     end
%     subplot(1,length(users),k)
%     plot(results,'linewidth',2.0)
%     hold on
%     grid on
%     grid minor
%     plot(1:num_rounds, lower_bound_loss*ones(1,num_rounds), 'k--','linewidth',2.0)
% %     lgd{length(users)+1} = 'Centralized';
%     title(['K = ' num2str(users(k))])
%     %legend(lgd)
%     ylabel('Test loss')
%     xlabel('Num. rounds')
%     set(gca, 'fontsize', 16)
% end
% legend({'Async (10%)','Async (25%)','Async (50%)','Async (75%)','Sync','Centr.'})
% save_figure( f6, 'test_loss', 'figures/tensorflow/' )
% 
% close all