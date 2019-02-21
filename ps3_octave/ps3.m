pkg load image

%{
%% 1-a
pts2d = dlmread(fullfile('input/pts2d-norm-pic_a.txt'));
pts3d = dlmread(fullfile('input/pts3d-norm.txt'));

M = compute_projection_matrix(pts2d, pts3d);

%% Now comparing the projected points vs expected points
pts2d_projected = project_points(pts3d, M);

plot_residuals(pts2d_projected, pts2d, true);
%}

%{
%% 1-b
pts2d = dlmread(fullfile('input/pts2d-pic_b.txt'));
pts3d = dlmread(fullfile('input/pts3d.txt'));

M = compute_projection_matrix(pts2d, pts3d);
pts2d_projected = project_points(pts3d, M);
residual = compute_average_residual(pts2d_projected, pts2d, false);

least_residual = realmax();
best_M = [];
for k = [8, 12, 16]
    for i = 1 : 10
        points_index = sort(randperm(length(pts2d), k));
        pts2d_compute = pts2d(points_index, :);
        pts3d_compute = pts3d(points_index, :);

        pts2d_test = pts2d;
        pts3d_test = pts3d;
        pts2d_test(points_index, :) = [];
        pts3d_test(points_index, :) = [];

        M = compute_projection_matrix(pts2d_compute, pts3d_compute);
        pts2d_projected = project_points(pts3d_test, M);
        residual = compute_average_residual(pts2d_projected, pts2d_test, false);

        if residual < least_residual
            least_residual = residual;
            best_M = M;
        end
    end
end

disp('least_residual:'), disp(least_residual);
disp('M:'), disp(best_M);

%% 1-c
camera_center = compute_camera_center(best_M);
disp('Camera center (World coordinates):');
disp(camera_center);
%}

%% 2-a
pts_a = dlmread(fullfile('input/pts2d-pic_a.txt'));
pts_b = dlmread(fullfile('input/pts2d-pic_b.txt'));
img_a = imread(fullfile('input/pic_a.jpg'));
img_b = imread(fullfile('input/pic_b.jpg'));

F = compute_fundamental_matrix(pts_b, pts_a);

epi_lines_a = compute_epipolor_lines(pts_b, 1, F);
epi_lines_b = compute_epipolor_lines(pts_a, 2, F);

f1 = drawEpiLines(epi_lines_a, img_a);
figure(f1);
hold on;
for i = 1 : length(pts_a);
    plot(pts_a(i, 1), pts_a(i, 2), 'r+');
end
hold off;
saveas(f1, fullfile('output', 'ps3-2-c-1.png'));

f2 = drawEpiLines(epi_lines_b, img_b);
figure(f2);
hold on;
for i = 1 : length(pts_b);
    plot(pts_b(i, 1), pts_b(i, 2), 'r+');
end
hold off;
saveas(f2, fullfile('output', 'ps3-2-c-2.png'));
