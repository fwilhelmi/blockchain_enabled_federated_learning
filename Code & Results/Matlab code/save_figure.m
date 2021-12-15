function [] = save_figure( fig, fig_name, path_figure )
% save_figure saves a figure to the indicated path under the indicated name
%   INPUT: 
%       * fig - figure to be saved
%       * fig_name - desired name for the figure
%       * path_figure - desired path to save the figure

    savefig([path_figure fig_name '.fig'])
    %saveas(gcf, [path_figure fig_name], 'epsc')
    saveas(gcf, [path_figure fig_name], 'png')
    
end

