function NMF_SOURCES_SEPARATION(file_prefix, nsrc, data_dir, result_dir)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                        
    % Based on the Flexible Audio Source Separatqion Toolbox (FASST), Version 1.0 
    %
    %  A. Ozerov, E. Vincent and F. Bimbot                                                              
    % "A General Flexible Framework for the Handling of Prior Information in Audio Source Separation," 
    % IEEE Transactions on Audio, Speech and Signal Processing 20(4), pp. 1118-1133 (2012).                                                   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    
    % Note: > file_prefix: musical piece name (without its format).
    %       > data_dir: directory in which the musical piece is located.
    %       > result_dir: directory in which sources are to be saved.
    %       > nsrc: number of sources to be retrieved.

    % Defaults parameters.
    if nargin==3
        result_dir = 'sources/';
    elseif nargin==2
        data_dir = 'data/';
        result_dir = 'sources/';
    elseif nargin==1
        nsrc = 3;
        data_dir = 'data/';
        result_dir = 'sources/';
    elseif nargin==0
        file_prefix = 'S_Hurley_Sunrise';
        nsrc = 3;
        data_dir = 'data/';
        result_dir = 'sources/';
    end

    % Type of spectral transform to be pursued.
    transform_type = 'stft';

    % Length of the window.
    window_length = 1024;

    % Number of NMF components.
    n_components_NMF = 12;

    % Number of iterations.
    n_iterations = 200;

    % Compute time-frequency representation.
    fprintf('Input time-frequency representation\n');
    [x, fs] = audioread([data_dir file_prefix '.wav']);
    x = x.';
    mix_nsamp = size(x,2);
    Cx = comp_transf_Cx(x, transform_type, window_length, fs);

    % Fill in mixture structure.
    mix_str = init_mix_struct_Mult_NMF_inst(Cx, nsrc, n_components_NMF, ...
                                        transform_type, fs, window_length);

    % Reinitialize mixing parameters.
    A = [sin((1:nsrc).*pi/8); cos((1:nsrc).*pi/8) ];
    for j = 1:nsrc
        mix_str.spat_comps{j}.params = A(:,j);
    end

    % Run parameters estimation (with simulated annealing).
    mix_str = estim_param_a_post_model(mix_str, n_iterations, 'ann');

    % Sources separation.
    ie_EM = separate_spat_comps(x, mix_str);

    % Computation of the spatial source images.
    fprintf('Computation of the spatial source images\n');
    for j=1:nsrc,
        audiowrite([result_dir file_prefix '_source_' int2str(j) '.wav'], ...
                    reshape(ie_EM(j,:,:),mix_nsamp,2),fs);
    end
    fprintf('Estimation of the sources done.\n  Audio files created. END.\n');
end