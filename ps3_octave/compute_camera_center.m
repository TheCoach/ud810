function C = compute_camera_center(M)
    if ~isequal(size(M), [3 4])
        error('Dimension of M must be 3x4');
    end
    Q = M(:, 1:3);
    m4 = M(:,4);
    C = -inverse(Q) * m4;
end