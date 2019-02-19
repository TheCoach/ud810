% ps2
pkg load image;  % Octave only

%{
%% 1-a
% Read images
L = im2double(imread(fullfile('input', 'pair0-L.png')));
R = im2double(imread(fullfile('input', 'pair0-R.png')));

% Compute disparity
D_L = disparity_ssd(L, R);
D_R = disparity_ssd(R, L);

% TODO: Save output images (D_L as output/ps2-1-a-2a-1.png and D_R as output/ps2-1-a-2.png)
% Note: They may need to be scaled/shifted before saving to show results properly
imwrite(rescale(abs(D_L), 0, 1), fullfile('output', 'ps2-1-a-1.png'));
imwrite(rescale(abs(D_R), 0, 1), fullfile('output', 'ps2-1-a-2.png'));
%}

%% 2-a
L = im2double(rgb2gray(imread(fullfile('input', 'pair1-L.png'))));
R = im2double(rgb2gray(imread(fullfile('input', 'pair1-R.png'))));

D_L = disparity_ssd(L, R, 'MaxDisparity', 128, 'WindowSize', 21);
imwrite(rescale(abs(D_L), 0, 1), fullfile('output', 'ps2-2-a-1.png'));

D_R = disparity_ssd(R, L, 'MaxDisparity', 128, 'WindowSize', 21);
imwrite(rescale(abs(D_R), 0, 1), fullfile('output', 'ps2-2-a-2.png'));

%% 4-a
L = im2double(rgb2gray(imread(fullfile('input', 'pair1-L.png'))));
R = im2double(rgb2gray(imread(fullfile('input', 'pair1-R.png'))));

D_L = disparity_ncorr(L, R, 'MaxDisparity', 128, 'WindowSize', 15);
imwrite(rescale(abs(D_L), 0, 1), fullfile('output', 'ps2-4-a-1.png'));

D_R = disparity_ncorr(R, L, 'MaxDisparity', 128, 'WindowSize', 15);
imwrite(rescale(abs(D_R), 0, 1), fullfile('output', 'ps2-4-a-2.png'));

