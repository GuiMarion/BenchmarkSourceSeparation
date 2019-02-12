%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
% ATIAM - MASTER PROGRAM - PROJECT AND MUSICAL APPLICATIONS
% MUSICAL SOURCES SEPARATION
% Separation using SOBI algorithm (Blind Source Separation).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   


% Reference: > A. Ozerov, E. Vincent and F. Bimbot                                                              
%             "A General Flexible Framework for the Handling of Prior Information in Audio Source Separation," 
%             IEEE Transactions on Audio, Speech and Signal Processing 20(4), pp. 1118-1133 (2012).    


% Reset.
clear all,
close all, 
clc

% Load the files to be used.
fprintf('Load audio files.');
[sound_cello, FS] = audioread('Cello_13.wav');
[sound_clrnt, ~]  = audioread('Clarinette_12.wav');
[sound_guitr, ~]  = audioread('Gtr_15.wav');

% Pursue SOBI algorithm with 3 observations to determine 3 sources.
X = [sound_cello(:,1)' + sound_cello(:,2)' ;
     sound_clrnt(:,1)' + sound_clrnt(:,2)' ;
     sound_guitr(:,1)' + sound_guitr(:,2)'];

% Number of sources to be retrieved.
n = size(X,1);

% Number of correlation matrices to be calculated by SOBI algorithm.
p = 5;

% Apply SOBI algorithm and estimate the sources.
fprintf('Applying SOBI algorithm.\n');
[H,S]=SOBI(X,n,p);

% Multiplication factor in order to obtain sufficiently audible sources.
power=250;

% Save the audio files associated with the sources.
file_name = 'sources/SOBI_source_';
for s=1:1:size(S,1)
    filename = strcat(file_name, int2str(s));
    filename = strcat(filename, '.wav');
    audiowrite(filename, power*S(s,:), FS);
end
fprintf('Estimation of the sources done.\n  Audio files created. END.\n');
