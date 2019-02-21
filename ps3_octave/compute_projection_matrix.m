function M = compute_projection_matrix(pts2d, pts3d);
    if size(pts2d, 2) ~= 2 || size(pts2d, 1) < 6
        error('Dimension of pts2d must be Mx2, M >= 6');
    end
    if size(pts3d, 2) ~= 3 || size(pts3d, 1) < 6
        error('Dimension of pts3d must be Mx3, M >= 6');
    end
    if size(pts2d, 1) ~= size(pts3d, 1)
        error('pts2d and pts3d must have same number of points');
    end

    numPoints = size(pts2d, 1);

    % solving for M using Direct Linear Calibration,
    % i.e, A * m = 0, find the eigenvector of A' * A with smallest eigenvalue, that is m.
    
    A = [];
    for i = 1 : numPoints
        x_i = pts3d(i, 1);
        y_i = pts3d(i, 2);
        z_i = pts3d(i, 3);
        u_i = pts2d(i, 1);
        v_i = pts2d(i, 2);
        A = [A; ...
             x_i, y_i, z_i, 1, 0  , 0  , 0  , 0, -u_i * x_i, -u_i * y_i, -u_i * z_i, -u_i; ...
             0  , 0  , 0  , 0, x_i, y_i, z_i, 1, -v_i * x_i, -v_i * y_i, -v_i * z_i, -v_i];
    end

    [v, d] = eig(A' * A);
    [~, i_min_eig] = min(diag(d));
    M = v(:, i_min_eig);
    M = reshape(M, 4, 3)';
end
