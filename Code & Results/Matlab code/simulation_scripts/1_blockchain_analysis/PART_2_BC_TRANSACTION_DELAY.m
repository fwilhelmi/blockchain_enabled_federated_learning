%%% *********************************************************************
%%% * Blockchain-Enabled RAN Sharing for Future 5G/6G Communications    *
%%% * Authors: Lorenza Giupponi & Francesc Wilhelmi (fwilhelmi@cttc.cat)*
%%% * Copyright (C) 2020-2025, and GNU GPLd, by Francesc Wilhelmi       *
%%% * GitHub repository: ...                                            *
%%% *********************************************************************
clear
clc

lambda = [0.02 0.2 2 20];
mu = [0.01 0.05 0.1 0.2 0.3 0.5 1];
timeout = 10;
queue_length = 1000;
block_size = 1:50;

mean_data_points = 500;
S_h = 200e3;
S_t = 5e3;
sigma = 0.1e-4;
R_node = 1e6;
R_p2p = 5; % Mbps
M = 10;
FORKS_ENABLED = 1;

for r = 1 : length(R_p2p)

    % Compute times involved in the transaction confirmation latency
    T_c = mean_data_points * sigma;
    T_up = (S_h + S_t) / R_node;
    %T_bp = log2(M) * (S_h + (S_t.*block_size)) / R_p2p; 
    T_bp = (S_h + (S_t.*block_size)) / (R_p2p(r)*1e6); 
    for i = 1 : length(mu)
        p_fork_analytical(i,:) = 1 - exp(-mu(i)*(M-1).*T_bp);  
    end

    % Process queue's simulator output
    total_transactions = cell(1,length(mu));
    transactions_dropped = cell(1,length(mu));
    drop_percentage = cell(1,length(mu));
    num_blocks_mined_by_timeout = cell(1,length(mu));
    queue_occupancy = cell(1,length(mu));
    queue_delay = cell(1,length(mu));    
    p_fork_sim = cell(1,length(mu));
    for i = 1 : length(mu)
        total_transactions{i} = zeros(length(lambda), length(block_size));
        transactions_dropped{i} = zeros(length(lambda), length(block_size));
        drop_percentage{i} = zeros(length(lambda), length(block_size));
        num_blocks_mined_by_timeout{i} = zeros(length(lambda), length(block_size));
        queue_occupancy{i} = zeros(length(lambda), length(block_size));
        queue_delay{i} = zeros(length(lambda), length(block_size));    
        p_fork_sim{i} = zeros(length(lambda), length(block_size));
    end

    %files_path = ['output_simulator_no_timer'];
    files_path = ['Outputs/output_queue_simulator/output_' num2str(R_p2p(r)) 'mbps_10miners'];
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
            queue_occupancy{ix_m}(ix_l,ix_s) = B(6)/queue_length * 100;
            queue_delay{ix_m}(ix_l,ix_s) = B(7);     
            if FORKS_ENABLED
                p_fork_sim{ix_m}(ix_l,ix_s) = B(8);  
            end
            fclose(file_data);
        end
    end

    % Compute the transaction confirmation delay
    for i = 1 : length(mu)
        for j = 1 : length(lambda)        
            for k = 1 : length(block_size)
                T_total{i}(j,k) = (queue_delay{i}(j,k) + T_bp(k)) ...
                    * (1/(1-p_fork_analytical(i,k)));
            end
        end
    end
end

%%
figure
subplot(2,2,1)
surf(T_total{2})
title(['\lambda = ' num2str(mu(2))])
xlabel('Block size (S^B)')
ylabel('\nu')
zlabel('T_{BC} (s)')
axis([1 50 1 4 0 2300])
set(gca,'fontsize',14)
subplot(2,2,2)
surf(T_total{4})
title(['\lambda = ' num2str(mu(4))])
xlabel('Block size (S^B)')
ylabel('\nu')
zlabel('T_{BC} (s)')
axis([1 50 1 4 0 2300])
set(gca,'fontsize',14)
subplot(2,2,3)
surf(T_total{7})
title(['\lambda = ' num2str(mu(7))])
xlabel('Block size (S^B)')
ylabel('\nu')
zlabel('T_{BC} (s)')
axis([1 50 1 4 0 2300])
set(gca,'fontsize',14)