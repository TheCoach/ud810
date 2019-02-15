% ps1
pkg load image;  % Octave only

%% 1-a
img = imread(fullfile('input', 'ps1-input0.png'));  % already grayscale
%% TODO: Compute edge image img_edges
img_edges = edge(img, 'Canny');
imwrite(img_edges, fullfile('output', 'ps1-1-a-1.png'));  % save as output/ps1-1-a-1.png

%% 2-a
[H, theta, rho] = hough_lines_acc(img_edges);  % defined in hough_lines_acc.m
%% Plot/show accumulator array H, save as output/ps1-2-a-1.png
H_display = imadjust(rescale(H));
imwrite(H_display, fullfile('output', 'ps1-2-a-1.png'));

%% 2-b
peaks = hough_peaks(H, 10);  % defined in hough_peaks.m
%% Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png
f1 = figure('Name', 'Hough Peaks, ps1-2-b');
imshow(H_display, 'XData', theta, 'YData', rho, 'InitialMagnification', 'fit');
axis on, axis normal, hold on;
plot(theta(peaks(:,2)), rho(peaks(:,1)),'s','color','red');
hold off;
saveas(f1, fullfile('output', 'ps1-2-b-1.png'));

%% 2-c
hough_lines_draw(img, fullfile('output', 'ps1-2-c-1.png'), peaks, rho, theta); 

%% 3-a
img_input0_original = imread(fullfile('input', 'ps1-input0-noise.png'));
%% Smooth with sigma == 5 gives all lines without having to adjust hough parameters
img_input0_smoothed = imsmooth(img_input0_original, 'gaussian', 5);
%% Smooth with sigma == 3 requires to adjust the 'Threshold' of hough_peaks to get all lines
% img_input0_smoothed = imsmooth(img_input0_original, 'gaussian', 3);
imwrite(img_input0_smoothed, fullfile('output', 'ps1-3-a-1.png'));

%% 3-b
img_input0_noise_edges = edge(img_input0_original, 'Canny');
img_input0_smoothed_edges = edge(img_input0_smoothed, 'Canny');
imwrite(img_input0_noise_edges, fullfile('output', 'ps1-3-b-1.png'));
imwrite(img_input0_smoothed_edges, fullfile('output', 'ps1-3-b-2.png'));

%% 3-c
[H, theta, rho] = hough_lines_acc(img_input0_smoothed_edges);
%% Plot/show accumulator array H, save as output/ps1-2-a-1.png
H_display = imadjust(rescale(H));
peaks = hough_peaks(H, 10);
%% For sigma == 3
% peaks = hough_peaks(H, 10, 'Threshold', 0.4 * max(H(:)));
f1 = figure('Name', 'Hough Peaks, ps1-3-c');
imshow(H_display, 'XData', theta, 'YData', rho, 'InitialMagnification', 'fit');
axis on, axis normal, hold on;
plot(theta(peaks(:,2)), rho(peaks(:,1)),'s','color','red');
hold off;
saveas(f1, fullfile('output', 'ps1-3-c-1.png'));
hough_lines_draw(img_input0_original, fullfile('output', 'ps1-3-c-2.png'), peaks, rho, theta);

%% 4-a & 4-b
img_input1 = rgb2gray(imread(fullfile('input', 'ps1-input1.png')));
img_input1_smoothed = imsmooth(img_input1, 'gaussian', 5);
img_input1_edges = uint8(edge(img_input1_smoothed, 'Canny'));
f1 = figure('name', 'ps1-4-a-and-b');
montage(cat(4, img_input1_smoothed, imadjust(uint8(img_input1_edges))));
saveas(f1, fullfile('output', 'ps1-4-a_and_b.png'));

%% 4-c
[H, theta, rho] = hough_lines_acc(img_input1_edges);
H_display = imadjust(rescale(H));
peaks = hough_peaks(H, 10, 'NHoodSize', floor(size(H) / 90.0) * 2 + 1);
f1 = figure('Name', 'Hough Peaks, ps-1-4-c');
imshow(H_display, 'XData', theta, 'YData', rho, 'InitialMagnification', 'fit');
axis on, axis normal, hold on;
plot(theta(peaks(:,2)), rho(peaks(:,1)),'s','color','red');
hold off;
saveas(f1, fullfile('output', 'ps1-4-c-1.png'));
hough_lines_draw(img_input1, fullfile('output', 'ps1-4-c-2.png'), peaks, rho, theta);
