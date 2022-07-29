close all
clear all
clc

%%
users = 1:200;
block_size = ceil(users);
mu = 2.5;
timeout = 10000;
queue_length = 500;
average_local_dataset_length = 100;
S_h = 200e3;
sigma = 0.1e-4;
R_node = 1; % Mbps
R_p2p = 5;  % Mbps
M = 10;

num_rounds = 200;

% Size of the different models in bits [FNN, CNN, Resnet50, VGG19]
%  -> 0.407 MB (FNN) / 78.63 MB (VGG19) 8*78.63e6; / 47.585 (RESNET) 47.585
models = {'FNN', 'CNN','Resnet50', 'VGG19'};
size_models = [8*0.407e6, 8*4.749e6, 8*47.585e6, 8*78.63e6]; 

% Lambda (for FNN)
% 1.2000    2.7000    3.9000    5.5000
% 0.9000    1.9000    2.7000    3.9000
% 0.7000    1.6000    2.3000    3.2000
% 0.6000    1.3000    1.8000    2.6000
% 0.5000    1.2000    1.7000    2.4000

% Lambda (for CNN)
% 0.3600    0.8100    1.1500    1.6200
% 0.2600    0.5700    0.8100    1.1500
% 0.2100    0.4700    0.6600    0.9400
% 0.1700    0.3800    0.5400    0.7600

% Lambda (for VGG19)
% 0.0900    0.2000    0.2800    0.4000
% 0.0600    0.1400    0.2000    0.2800
% 0.0500    0.1200    0.1600    0.2300
% 0.0400    0.0900    0.1300    0.1900
% 0.0400    0.0900    0.1200    0.1700

% Lambda (for ResNet50)
% 0.1100    0.2600    0.3600    0.5100
% 0.0800    0.1800    0.2600    0.3600
% 0.0700    0.1500    0.2100    0.3000
% 0.0500    0.1200    0.1700    0.2400

% Compute the iteration time for each model and configuration
for i = 1 : length(models)
    S_t = size_models(i); 
    for k = 1 : length(users)
        sqrt( users(k) ./ ( ((S_h+S_t*k)/(R_node*1e6)) + ...
            (average_local_dataset_length*sigma) + (S_t/(R_node*1e6))) );
        lambda{i}(k) = sqrt( users(k) ./ ( ((S_h+S_t*k)/(R_node*1e6)) + ...
            (average_local_dataset_length*sigma) + (S_t/(R_node*1e6))) );
    end
    % Round the result to 2 decimals
    lambda{i} = round(lambda{i}, 2); 
    % Compute the iteration time for each possible configuration
    T_iter{i} = zeros(1, length(users));
    for n = 1 : length(users)
        bs = n;
        l = lambda{i}(n);
        % Get the block propagation delay
        T_bp = (S_h + (S_t*bs)) / (R_p2p*1e6); 
        % Load the delay computed through simulations
        files_path = ['Outputs/output_queue_simulator/output_all/script_output_m' ...
            num2str(mu) '_l' num2str(l) '_s' num2str(bs) '.txt'];   
        % Iterate for each file in the directory
        file_data = fopen(files_path);
        A = textscan(file_data,'%s','Delimiter',';');
        B = str2double(A{:});    
        % Get the queue delay and the fork probability from simulations
        queue_delay(n) = B(7);     
        p_fork_sim(n) = B(8);  
        % Save the results and close the file
        T_iter{i}(n) = (queue_delay(n) + T_bp) * (1/(1-p_fork_sim(n)));
        fclose(file_data);    
    end
end

% S_t_fnn = 8*0.407e6; % 0.407 MB (FNN) / 78.63 MB (VGG19) 8*78.63e6; / 47.585 (RESNET) 47.585
% for k = 1 : length(users)
%     sqrt( users(k) ./ ( ((S_h+S_t_fnn*k)/(R_node*1e6)) + ...
%         (average_local_dataset_length*sigma) + (S_t_fnn/(R_node*1e6))) );
%     lambda_fnn(k) = sqrt( users(k) ./ ( ((S_h+S_t_fnn*k)/(R_node*1e6)) + ...
%         (average_local_dataset_length*sigma) + (S_t_fnn/(R_node*1e6))) );
% end
% lambda_fnn = round(lambda_fnn,1);
% 
% S_t_cnn = 8*4.749e6; % 0.407 MB (FNN) / 78.63 MB (VGG19) 8*78.63e6; / 47.585 (RESNET) 47.585
% for k = 1 : length(users)
%     sqrt( users(k) ./ ( ((S_h+S_t_cnn*k)/(R_node*1e6)) + ...
%         (average_local_dataset_length*sigma) + (S_t_cnn/(R_node*1e6))) );
%     lambda_cnn(k) = sqrt( users(k) ./ ( ((S_h+S_t_cnn*k)/(R_node*1e6)) + ...
%         (average_local_dataset_length*sigma) + (S_t_cnn/(R_node*1e6))) );
% end
% lambda_cnn = round(lambda_cnn,2);
% 
% S_t_resnet = 8*47.585e6; % 0.407 MB (FNN) / 78.63 MB (VGG19) 8*78.63e6; / 47.585 (RESNET) 47.585
% for k = 1 : length(users)
%     sqrt( users(k) ./ ( ((S_h+S_t_resnet*k)/(R_node*1e6)) + ...
%         (average_local_dataset_length*sigma) + (S_t_resnet/(R_node*1e6))) );
%     lambda_resnet(k) = sqrt( users(k) ./ ( ((S_h+S_t_resnet*k)/(R_node*1e6)) + ...
%         (average_local_dataset_length*sigma) + (S_t_resnet/(R_node*1e6))) );
% end
% lambda_resnet = round(lambda_resnet,2);
% 
% S_t_vgg = 8*78.63e6; % 0.407 MB (FNN) / 78.63 MB (VGG19) 8*78.63e6; / 47.585 (RESNET) 47.585
% for k = 1 : length(users)
%     sqrt( users(k) ./ ( ((S_h+S_t_vgg*k)/(R_node*1e6)) + ...
%         (average_local_dataset_length*sigma) + (S_t_vgg/(R_node*1e6))) );
%     lambda_vgg(k) = sqrt( users(k) ./ ( ((S_h+S_t_vgg*k)/(R_node*1e6)) + ...
%         (average_local_dataset_length*sigma) + (S_t_vgg/(R_node*1e6))) );
% end
% lambda_vgg = round(lambda_vgg,2);

%% PART 1: DELAYS

% 1.0 s-FLchain
% Compute times involved in the transaction confirmation latency
% T_c = average_local_dataset_length * sigma;
% T_up_resnet = (S_h + S_t_resnet) / (R_node*1e6);
% T_up_vgg = (S_h + S_t_vgg) / (R_node*1e6);
% T_bp = (S_h + (S_t.*users)) / (R_p2p*1e6); 
% p_fork_analytical = 1 - exp(-mu*(M-1).*T_bp);  

% % 1.1  FNN
% T_iter_fnn = zeros(length(partitions), length(users));
% for i = 1 : length(users)
%     bs = i;
%     l = lambda_fnn(i);
%     T_bp = (S_h + (S_t_fnn*bs)) / (R_p2p*1e6); 
%     files_path = ['Outputs/output_queue_simulator/output_all/script_output_m' ...
%         num2str(mu) '_l' num2str(l) '_s' num2str(bs) '.txt'];   
%     % Iterate for each file in the directory
%     file_data = fopen(files_path);
%     A = textscan(file_data,'%s','Delimiter',';');
%     B = str2double(A{:});    
%     % Store results to variables
%     queue_delay(i) = B(7);     
%     p_fork_sim(i) = B(8);  
%     T_iter_fnn(i) = (queue_delay(i) + T_bp) ...
%         * (1/(1-p_fork_sim(i)));
%     fclose(file_data);    
% end
% T_fnn = [T_iter_fnn];
% 
% % 1.2  CNN
% T_iter_cnn = zeros(length(partitions), length(users));
% for i = 1 : length(users)
%     bs = i;
%     l = lambda_cnn(i);
%     T_bp = (S_h + (S_t_cnn*bs)) / (R_p2p*1e6); 
%     files_path = ['Outputs/output_queue_simulator/output_all/script_output_m' ...
%         num2str(mu) '_l' num2str(l) '_s' num2str(bs) '.txt'];   
%     % Iterate for each file in the directory
%     file_data = fopen(files_path);
%     A = textscan(file_data,'%s','Delimiter',';');
%     B = str2double(A{:});    
%     % Store results to variables
%     queue_delay(i) = B(7);     
%     p_fork_sim(i) = B(8);  
%     T_iter_cnn(i) = (queue_delay(i) + T_bp) ...
%         * (1/(1-p_fork_sim(i)));
%     fclose(file_data);    
% end
% T_cnn = [T_iter_cnn];
% 
% % 1.3  RESNET
% T_iter_resnet = zeros(length(partitions), length(users));
% for i = 1 : length(users)
%     bs = i;
%     l = lambda_resnet(i);
%     T_bp = (S_h + (S_t_resnet*bs)) / (R_p2p*1e6); 
%     files_path = ['Outputs/output_queue_simulator/output_all/script_output_m' ...
%         num2str(mu) '_l' num2str(l) '_s' num2str(bs) '.txt'];   
%     % Iterate for each file in the directory
%     file_data = fopen(files_path);
%     A = textscan(file_data,'%s','Delimiter',';');
%     B = str2double(A{:});    
%     % Store results to variables
%     queue_delay(i) = B(7);     
%     p_fork_sim(i) = B(8);  
%     T_iter_resnet(i) = (queue_delay(i) + T_bp) ...
%         * (1/(1-p_fork_sim(i)));
%     fclose(file_data);    
% end
% T_resnet = [T_iter_resnet];
% % 1.4 VGG
% T_iter_vgg = zeros(length(partitions), length(users));
% for i = 1 : length(users)
%     bs = i;
%     l = lambda_vgg(i);
%     T_bp = (S_h + (S_t_vgg*bs)) / (R_p2p*1e6); 
%     files_path = ['Outputs/output_queue_simulator/output_all/script_output_m' ...
%         num2str(mu) '_l' num2str(l) '_s' num2str(bs) '.txt'];   
%     % Iterate for each file in the directory
%     file_data = fopen(files_path);
%     A = textscan(file_data,'%s','Delimiter',';');
%     B = str2double(A{:});    
%     % Store results to variables
%     queue_delay(i) = B(7);     
%     p_fork_sim(i) = B(8);  
%     T_iter_vgg(i) = (queue_delay(i) + T_bp) ...
%         * (1/(1-p_fork_sim(i)));
%     fclose(file_data);    
% end
% % T = [T_iter_async(1:4,:); T_iter_sync];
% T_vgg = [T_iter_vgg];

%% Plot the results
hold on
for i = 1 : length(models)
    plot(log(T_iter{i}), 'linewidth', 1.5)    
end
legend(models)
xlabel('Num. of users, K')
ylabel('Iteration time, log(T)')
% cdfplot(T_cnn(:))
% cdfplot(T_resnet(:))
% cdfplot(T_vgg(:))
% bar(T_resnet')
% hold on
% bar(T_vgg', 'facealpha', 0.1)
% legend({'a-FLchain (10%)','a-FLchain (25%)','a-FLchain (50%)',...
%     'a-FLchain (75%)','s-FLchain'})

% xlabel('Num. users, K')
% ylabel('FL round time (s)')
%xticks(1:4)
%xticklabels([10, 50, 100, 200])
set(gca, 'fontsize', 14)
grid on
grid minor