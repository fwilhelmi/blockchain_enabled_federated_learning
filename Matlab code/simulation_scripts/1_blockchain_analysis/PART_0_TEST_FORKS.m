%%% *********************************************************************
%%% * Blockchain-Enabled RAN Sharing for Future 5G/6G Communications    *
%%% * Authors: Lorenza Giupponi & Francesc Wilhelmi (fwilhelmi@cttc.cat)*
%%% * Copyright (C) 2020-2025, and GNU GPLd, by Francesc Wilhelmi       *
%%% * GitHub repository: ...                                            *
%%% *********************************************************************
clear
clc

mu = [0.01 0.1];
block_size = 1:10000;

S_h = 200e3;
S_t = 5e3;
R_p2p = [1e6 5e6 20e6];
M = [10 100 1000];

figure
for m = 1 : length(M)   
    leg = {};
    counter = 1;
    for c = 1 : length(R_p2p)
        for i = 1 : length(mu)     
            for k = 1 : length(block_size)
                T_bp = (S_h + (S_t.*block_size(k))) ./ R_p2p(c); 
                p_fork{c}(i,k) = 1 - exp(-mu(i)*(M(m)-1)*T_bp);
            end
            leg{counter} = ['\lambda = ' num2str(mu(i)) ' - C_{P2P} = ' num2str(R_p2p(c)/1e6) ' Mbps'];
            counter = counter + 1;
        end
    end
    subplot(1,length(M),m)
    for c = 1 : length(R_p2p)
        for i = 1 : length(mu)
            plot(p_fork{c}(i,:), 'linewidth', 2)
            hold on
        end
    end
    set(gca, 'fontsize', 16)
    xlabel('# of users')
    ylabel('Fork probability, p_{fork}')
    grid on
    grid minor
    if m == 1
        legend(leg)
    end
    title(['M = ' num2str(M(m))])
end