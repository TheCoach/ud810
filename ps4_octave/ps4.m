pkg load image;

img_transA = imread(fullfile('input', 'transA.jpg'));
img_transB = imread(fullfile('input', 'transB.jpg'));
img_simA = imread(fullfile('input', 'simA.jpg'));
img_simB = imread(fullfile('input', 'simB.jpg'));
img_check = uint8(imread(fullfile('input', 'check.bmp')));
img_check_rot = uint8(imread(fullfile('input', 'check_rot.bmp')));

sigma = 2;
f_g = fspecial('gaussian', [2*ceil(2*sigma)+1, 2*ceil(2*sigma)+1], sigma);
[dgx, dgy] = imgradientxy(f_g);

gx_transA = imfilter(img_transA, dgx, 'conv');
gy_transA = imfilter(img_transA, dgy, 'conv');

gx_simA = imfilter(img_simA, dgx, 'conv');
gy_simA = imfilter(img_simA, dgy, 'conv');

f1a1 = figure('visible', 'off');
hold on
montage(cat(4, gx_transA, gy_transA), 'size', [1 2]);
hold off;

f1a2 = figure('visible', 'off');
hold on
montage(cat(4, gx_simA, gy_simA), 'size', [1 2]);
hold off;

saveas(f1a1, fullfile('output', 'ps4-1-a-1.png'));
saveas(f1a2, fullfile('output', 'ps4-1-a-2.png'));

%{
gx_check_rot = imfilter(img_check_rot, dgx, 'conv');
gy_check_rot = imfilter(img_check_rot, dgy, 'conv');

figure;
hold on
montage(cat(4, gx_check_rot, gy_check_rot), 'size', [1 2]);
hold off;

harris_check_rot = compute_harris_value(gx_check_rot, gy_check_rot, 5);

figure();
imshow(mat2gray(harris_check_rot));
colormap(gca, 'jet');
colorbar();

[corners, R] = find_harris_corners(harris_check_rot, 0.7, 3);
figure(), imshow(mat2gray(R));
colormap(gca, 'jet');
colorbar();

figure(), imshow(img_check_rot), hold on;
plot(corners(:, 2), corners(:, 1), 'r+');
hold off;
%}

harris_transA = compute_harris_value(gx_transA, gy_transA);
% show harris values
f1b1 = figure('visible', 'off');
imshow(mat2gray(harris_transA));
colormap(gca, 'jet');
colorbar();
saveas(f1b1, fullfile('output', 'ps4-1-b-1.png'));
% plot corners on original image
[corners, ~] = find_harris_corners(harris_transA, 0.5, 15);
f1c1 = figure('visible', 'off');
imshow(img_transA)
hold on;
plot(corners(:, 2), corners(:, 1), 'r+');
hold off;
saveas(f1c1, fullfile('output', 'ps4-1-c-1.png'));

gx_transB = imfilter(img_transB, dgx, 'conv');
gy_transB = imfilter(img_transB, dgy, 'conv');
harris_transB = compute_harris_value(gx_transB, gy_transB);
% show harris values
f1b2 = figure('visible', 'off');
imshow(mat2gray(harris_transB));
colormap(gca, 'jet');
colorbar();
saveas(f1b2, fullfile('output', 'ps4-1-b-2.png'));
[corners, ~] = find_harris_corners(harris_transB, 0.6, 15);
f1c2 = figure('visible', 'off');
imshow(img_transB);
hold on;
plot(corners(:, 2), corners(:, 1), 'r+');
hold off;
saveas(f1c2, fullfile('output', 'ps4-1-c-2.png'));

harris_simA = compute_harris_value(gx_simA, gy_simA);
f1b3 = figure('visible', 'off');
imshow(mat2gray(harris_simA));
colormap(gca, 'jet');
colorbar();
saveas(f1b3, fullfile('output', 'ps4-1-b-3.png'));
[corners, ~] = find_harris_corners(harris_simA, 0.6, 15);
f1c3 = figure('visible', 'off');
imshow(img_simA)
hold on;
plot(corners(:, 2), corners(:, 1), 'r+');
hold off;
saveas(f1c3, fullfile('output', 'ps4-1-c-3.png'));

gx_simB = imfilter(img_simB, dgx);
gy_simB = imfilter(img_simB, dgy);
harris_simB = compute_harris_value(gx_simB, gy_simB);
f1b4 = figure('visible', 'off');
imshow(mat2gray(harris_simB));
colormap(gca, 'jet');
colorbar();
saveas(f1b4, fullfile('output', 'ps4-1-b-4.png'));
[corners, ~] = find_harris_corners(harris_simB, 0.6, 15);
f1c4 = figure('visible', 'off');
imshow(img_simB);
hold on;
plot(corners(:, 2), corners(:, 1), 'r+');
hold off;
saveas(f1c4, fullfile('output', 'ps4-1-c-4.png'));