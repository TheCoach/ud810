% part 1
img1 = imread("input/4.1.04.tiff");
img2 = imread("input/4.2.07.tiff");
imwrite(img1, "output/ps0-1-a-1.png");
imwrite(img2, "output/ps0-1-a-2.png");

% part 2
img1_r = img1(:,:,1);
img1_g = img1(:,:,2);
img1_b = img1(:,:,3);

img1_p2_a = img1;
img1_p2_a(:,:,1) = img1_b;
img1_p2_a(:,:,3) = img1_r;

imwrite(img1_p2_a, "output/ps0-2-a-1.png");
imwrite(img1_g, "output/ps0-2-b-1.png");
imwrite(img1_r, "output/ps0-2-c-1.png");

% part 3
img2_g = img2(:,:,2);
[img1_nr, img1_nc] = size(img1_g);
[img2_nr, img2_nc] = size(img2_g);

img2_center_r = round((img2_nr - 100) / 2);
img2_center_c = round((img2_nc - 100) / 2);

img1_center_r = round((img1_nr - 100) / 2);
img1_center_c = round((img1_nc - 100) / 2);

img2_g_p3 = img2_g;
img2_g_p3([img2_center_r : img2_center_r + 99], [img2_center_c : img2_center_c + 99]) = ...
    img1_g([img1_center_r : img1_center_r + 99], [img1_center_c : img1_center_c + 99]);

imwrite(img2_g_p3, "output/ps0-3-a-1.png");

% part 4
img1_g_vector = reshape(img1_g, 1, []);
img1_g_min = min(img1_g_vector);
img1_g_max = max(img1_g_vector);
img1_g_mean = mean(img1_g_vector);
img1_g_sd = std(img1_g_vector);

fprintf("(min, max, standard deviation) in image1_green: (%d, %d, %f)\n", img1_g_min, img1_g_max, img1_g_sd);

img1_g_p4_b = ((img1_g .- img1_g_mean) ./ img1_g_sd) .* 10 .+ img1_g_mean;
imwrite(img1_g_p4_b, "output/ps0-4-b-1.png");

img1_g_p4_c = circshift(img1_g, 2, -2);
imwrite(img1_g_p4_c, "output/ps0-4-c-1.png");

img1_g_p4_d = abs(img1_g .- img1_g_p4_c);
imwrite(img1_g_p4_d, "output/ps0-4-d-1.png");

% part 5
sigma = 16;
noise = randn(size(img1_g)).*sigma;
img1_p5_a = img1(:,:,2) + noise;
imwrite(img1_p5_a, "output/ps0-5-a-1.png");

img1_p5_b = img1(:,:,3) + noise;
imwrite(img1_p5_b, "output/ps0-5-b-1.png");
