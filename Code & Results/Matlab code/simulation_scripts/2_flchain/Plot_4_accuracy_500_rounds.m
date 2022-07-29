close all
clear all
clc

users = [200];
partitions = [1];

num_rounds = 500;
upper_bound_accuray = 0.9594;
upper_bound_accuray_cnn = 0.9893;
%lower_bound_loss = 0.1352;

% CNN
% Epoch 1/5
% 7500/7500 [==============================] - 74s 10ms/step - loss: 0.2421 - sparse_categorical_accuracy: 0.9262
% Epoch 2/5
% 7500/7500 [==============================] - 74s 10ms/step - loss: 0.1211 - sparse_categorical_accuracy: 0.9642
% Epoch 3/5
% 7500/7500 [==============================] - 73s 10ms/step - loss: 0.0929 - sparse_categorical_accuracy: 0.9723
% Epoch 4/5
% 7500/7500 [==============================] - 70s 9ms/step - loss: 0.0749 - sparse_categorical_accuracy: 0.9777
% Epoch 5/5
% 7500/7500 [==============================] - 74s 10ms/step - loss: 0.0625 - sparse_categorical_accuracy: 0.9816
% 1250/1250 [==============================] - 7s 5ms/step - loss: 0.0344 - sparse_categorical_accuracy: 0.9893
% test metrics=[0.034437425434589386, 0.989300012588501]

root_folder_1 = 'Outputs/output_tensorflow/results_emnist/num_classes_10_fnn';
root_folder_2 = 'Outputs/output_tensorflow/results_emnist/num_classes_10_cnn';

%% EVAL. ACCURACY
f = figure;
f.Position = [300 300 780 200];
results_bar = cell(1, length(users));
results_bar_noniid = cell(1, length(users));
pwd
for k = 1 : length(users)      
    results = [];
    results2= [];
    for p = 1 : length(partitions)
        % IID FNN
        file_path = [root_folder_1 '/eval_accuracy_K' ...
            num2str(users(k)) '_' num2str(partitions(p)) '.txt'];
        lgd{k} = strcat('K=',num2str(users(k))) ;
        fid = fopen(file_path);
        tline = fgetl(fid);
        B = [];
        results_bar{k} = [];
        while ischar(tline)
            A = textscan(tline, '%s', 'Delimiter', ' ');
            B = [B str2double(A{:})];    
            tline = fgetl(fid);
        end
        results = [results B];
        results_bar{k} = [results_bar{k} results];
        fclose(fid);   
        % IID CNN
        file_path = [root_folder_2 '/CNN_eval_accuracy_K' ...
            num2str(users(k)) '_' num2str(partitions(p)) '.txt'];
        lgd{k} = strcat('K=',num2str(users(k))) ;
        fid = fopen(file_path);
        tline = fgetl(fid);
        B = [];
        results_bar2{k} = [];
        while ischar(tline)
            A = textscan(tline, '%s', 'Delimiter', ' ');
            B = [B str2double(A{:})];    
            tline = fgetl(fid);
        end
        results2 = [results2 B];
        results_bar2{k} = [results_bar2{k} results2];
        fclose(fid);  
    end
    subplot(1,length(users),k)
    plot(results,'linewidth',2.0)
    hold on
    plot(1:num_rounds, upper_bound_accuray*ones(1,num_rounds), 'k--','linewidth',2.0)
    plot(results2,'linewidth',2.0)
    plot(1:num_rounds, upper_bound_accuray_cnn*ones(1,num_rounds), 'k:','linewidth',2.0)
        grid on
    grid minor
    title(['K = ' num2str(users(k))])
    ylabel('Eval accuracy')
    xlabel('Num. rounds')
    set(gca, 'fontsize', 16)
end
legend({'FNN (federated)', 'FNN (Centr.)','CNN (federated)', 'CNN (Centr.)'})
%save_figure( f, 'eval_accuracy', 'figures/' )

% %% Last 50 iterations
% f11 = figure;
% mean_bar_data = [mean(results_bar{1}(150:200,:)); mean(results_bar{2}(150:200,:)); ...
%     mean(results_bar{3}(150:200,:)); mean(results_bar{4}(150:200,:))];
% % std_bar_data = [std(results_bar{1}(150:200,:)); std(results_bar{2}(150:200,:)); ...
% %     std(results_bar{3}(150:200,:)); std(results_bar{4}(150:200,:))];
% bar(mean_bar_data,'LineStyle',':','LineWidth',1.5,'FaceAlpha',0.3)
% hold on
% mean_bar_data_noniid = [mean(results_bar_noniid{1}(150:200,:)); mean(results_bar_noniid{2}(150:200,:)); ...
%     mean(results_bar_noniid{3}(150:200,:)); mean(results_bar_noniid{4}(150:200,:))];
% std_bar_data_noniid = [std(results_bar_noniid{1}(150:200,:)); std(results_bar_noniid{2}(150:200,:)); ...
%     std(results_bar_noniid{3}(150:200,:)); std(results_bar_noniid{4}(150:200,:))];
% bar(mean_bar_data_noniid)
% hold on
% ngroups = size(mean_bar_data, 1);
% nbars = size(mean_bar_data, 2);
% % Calculating the width for each bar group
% groupwidth = min(0.8, nbars/(nbars + 1.5));
% for i = 1:nbars
%     x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
%     errorbar(x, mean_bar_data_noniid(:,i), std_bar_data_noniid(:,i), 'r.', 'linewidth', 1);
% end
% grid on
% grid minor
% load('mean_bar_data.mat')
% plot(0:5, upper_bound_accuray*ones(1,6), 'k--','linewidth',2.0)
% ylabel('Mean eval. accuracy')
% xlabel('Num. clients, K')
% legend({'','','','','','Async (10%)','Async (25%)','Async (50%)','Async (75%)',...
%     'Sync','','','','','','Centr.'})
% set(gca, 'fontsize', 16)
% save_figure( f11, 'mean_eval_accuracy', 'figures/' )

%% OTHER PLOTS THAT HAVE NOT BEEN INCLUDED (COMMENTED)
% %% All 200 iterations
% figure;
% mean_bar_data = [mean(results_bar{1}); mean(results_bar{2}); ...
%     mean(results_bar{3}); mean(results_bar{4})];
% std_bar_data = [std(results_bar{1}); std(results_bar{2}); ...
%     std(results_bar{3}); std(results_bar{4})];
% bar(mean_bar_data)
% hold on
% ngroups = size(mean_bar_data, 1);
% nbars = size(mean_bar_data, 2);
% % Calculating the width for each bar group
% groupwidth = min(0.8, nbars/(nbars + 1.5));
% for i = 1:nbars
%     x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
%     errorbar(x, mean_bar_data(:,i), std_bar_data(:,i), 'r.', 'linewidth', 1);
% end
% grid on
% grid minor
% plot(0:5, upper_bound_accuray*ones(1,6), 'k--','linewidth',2.0)
% ylabel('Mean eval. accuracy')
% xlabel('Num. clients, K')
% legend({'Async (10%)','Async (25%)','Async (50%)','Async (75%)',...
%     'Sync','','','','','','Centr.'})
% set(gca, 'fontsize', 16)

% %% EVAL. LOSS
% f2 = figure;
% f2.Position = [300 300 780 200];
% results_bar = cell(1, length(users));
% for k = 1 : length(users)  
%     results = [];
%     for p = 1 : length(partitions)
%         file_path = ['output_tensorflow/' root_folder '/eval_loss_K' ...
%             num2str(users(k)) '_' num2str(partitions(p)) '.txt'];
%         lgd{k} = strcat('K=',num2str(users(k))) ;
%         fid = fopen(file_path);
%         tline = fgetl(fid);
%         B = [];
%         results_bar{k} = [];
%         while ischar(tline)
%             A = textscan(tline, '%s', 'Delimiter', ' ');
%             B = [B str2double(A{:})];    
%             tline = fgetl(fid);
%         end
%         results = [results B];
%         results_bar{k} = [results_bar{k} results];
%         fclose(fid);    
%     end
%     subplot(1,length(users),k)
%     plot(results,'linewidth',2.0)
%     hold on
%     grid on
%     grid minor
%     plot(1:num_rounds, lower_bound_loss*ones(1,num_rounds), 'k--','linewidth',2.0)
%     title(['K = ' num2str(users(k))])
%     ylabel('Eval loss (%)')
%     xlabel('Num. rounds')
%     set(gca, 'fontsize', 16)
%     axis([0 200 0 4])
% end
% legend({'Async (10%)','Async (25%)','Async (50%)','Async (75%)','Sync','Centr.'})
% save_figure( f2, 'eval_loss', 'figures/' )
% 
% %% Last 50 iterations
% f21 = figure;
% mean_bar_data = [mean(results_bar{1}(150:200,:)); mean(results_bar{2}(150:200,:)); ...
%     mean(results_bar{3}(150:200,:)); mean(results_bar{4}(150:200,:))];
% std_bar_data = [std(results_bar{1}(150:200,:)); std(results_bar{2}(150:200,:)); ...
%     std(results_bar{3}(150:200,:)); std(results_bar{4}(150:200,:))];
% bar(mean_bar_data)
% hold on
% ngroups = size(mean_bar_data, 1);
% nbars = size(mean_bar_data, 2);
% % Calculating the width for each bar group
% groupwidth = min(0.8, nbars/(nbars + 1.5));
% for i = 1:nbars
%     x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
%     errorbar(x, mean_bar_data(:,i), std_bar_data(:,i), 'r.', 'linewidth', 1);
% end
% grid on
% grid minor
% plot(0:5, lower_bound_loss*ones(1,6), 'k--','linewidth',2.0)
% ylabel('Mean eval. loss')
% xlabel('Num. clients, K')
% legend({'Async (10%)','Async (25%)','Async (50%)','Async (75%)',...
%     'Sync','','','','','','Centr.'})
% set(gca, 'fontsize', 16)
% save_figure( f21, 'mean_eval_loss', 'figures/' )

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
%     ylabel('Train accuracy (%)')
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
%     ylabel('Train loss (%)')
%     xlabel('Num. rounds')
%     set(gca, 'fontsize', 16)
% end
% legend({'Async (10%)','Async (25%)','Async (50%)','Async (75%)','Sync','Centr.'})
% save_figure( f4, 'train_loss', 'figures/' )
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
%     ylabel('Test accuracy (%)')
%     xlabel('Num. rounds')
%     set(gca, 'fontsize', 16)
% end
% legend({'Async (10%)','Async (25%)','Async (50%)','Async (75%)','Sync','Centr.'})
% save_figure( f5, 'test_accuracy', 'figures/' )
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
%     ylabel('Test loss (%)')
%     xlabel('Num. rounds')
%     set(gca, 'fontsize', 16)
% end
% legend({'Async (10%)','Async (25%)','Async (50%)','Async (75%)','Sync','Centr.'})
% save_figure( f6, 'test_loss', 'figures/' )

%close all