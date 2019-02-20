% ps2
pkg load image;  % Octave only


%% 1-a
% Read images
L = im2double(imread(fullfile('input', 'pair0-L.png')));
R = im2double(imread(fullfile('input', 'pair0-R.png')));

% Compute disparity
D_L = disparity_ssd(L, R);
D_R = disparity_ssd(R, L);

% TODO: Save output images (D_L as output/ps2-1-a-2a-1.png and D_R as output/ps2-1-a-2.png)
% Note: They may need to be scaled/shifted before saving to show results properly
imwrite(mat2gray(abs(D_L)), fullfile('output', 'ps2-1-a-1.png'));
imwrite(mat2gray(abs(D_R)), fullfile('output', 'ps2-1-a-2.png'));


%% 2-a
L = im2double(rgb2gray(imread(fullfile('input', 'tsukuba-L.png'))));
R = im2double(rgb2gray(imread(fullfile('input', 'tsukuba-R.png'))));

% tic;
% D_L = disparity_ssd(L, R, 'MaxDisparity', 20, 'WindowSize', 13);
% toc;
% imwrite(mat2gray(abs(D_L)), fullfile('output', 'ps2-2-a-1.png'));

%% 3-a
% L_noised = imnoise(L, "gaussian", 0, 0.03);
% imwrite(L_noised, fullfile('input', 'tsukuba-L-noised.png'));

% tic;
% D_L = disparity_ssd(L_noised, R, 'MaxDisparity', 20, 'WindowSize', 13);
% toc;
% imwrite(mat2gray(abs(D_L)), fullfile('output', 'ps2-3-a-1.png'));

%% 4-a

tic;
D_L = disparity_ncorr(L, R, 'MaxDisparity', 20, 'WindowSize', 13);
toc;
imwrite(mat2gray(abs(D_L)), fullfile('output', 'ps2-4-a-1.png'));

%% 4-b
% tic;
% D_L = disparity_ncorr(L_noised, R, 'MaxDisparity', 20, 'WindowSize', 13);
% toc;
% imwrite(mat2gray(abs(D_L)), fullfile('output', 'ps2-4-b-1.png'));
