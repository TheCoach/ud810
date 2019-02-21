function pts2d_projected = project_points(pts3d, M)
    if size(pts3d, 2) ~= 3
        error('Dimension of pts3d must be Mx3');
    end
    if ~isequal(size(M), [3 4])
        error('Dimension of M must be 3x4');
    end

    % homogeneous coordinates, also note the transpose
    pts3d_H = padarray(pts3d, [0 1], 1, 'post')';
    pts2d_projected_H = M * pts3d_H;

    pts2d_projected = pts2d_projected_H;
    for c = 1 : size(pts2d_projected_H, 2)
        pts2d_projected(:, c) = pts2d_projected(:, c) ./ pts2d_projected(3, c);
    end

    pts2d_projected = pts2d_projected(1:2 , :)';
end
