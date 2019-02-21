function epi_lines = compute_epipolor_lines(points, imageIndex, F)
    if size(points, 2) ~= 2
        error('Dimension of pts3d must be Mx2');
    end
    if ~isequal(size(F), [3 3])
        error('Dimension of M must be 3x4');
    end

    if imageIndex ~= 1 && imageIndex ~= 2
        error('imageIndex must be either 1 or 2, indicating which image the point belong to.');
    end

    % homogeneous coordinates, also note the transpose
    points_H = padarray(points', [1 0], 1, 'post');

    %{
        p_transpose * F * p_prime = 0,
        then,
        l = F * p_prime is the epipolar line in the p image associated with p_prime
        l_prime = F_transpose * p is the epipolar line in the p_prime image associated with p
    %}

    if (imageIndex == 2)
        epi_lines = F * points_H;
    else
        epi_lines = F' * points_H;
    end
end
