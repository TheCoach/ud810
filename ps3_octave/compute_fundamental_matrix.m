function F = compute_fundamental_matrix(pts1, pts2);
    if size(pts1, 2) ~= 2 || size(pts2, 2) ~= 2 
        error('Dimension of points must be Mx2, M >= 8');
    end

    if ~isequal(size(pts1), size(pts2))
        error('Dimensions of points must be the same');
    end

    f = solve_for_f(pts1, pts2);
    [u,s,v] = svd(f);
    [r, c] = ind2sub(size(s), find(s == min(diag(s))));
    s(r, c) = 0;
    F = u * s * v';
endfunction

function F = solve_for_f(pts1, pts2);
    if size(pts1, 2) ~= 2 || size(pts2, 2) ~= 2
        error('Dimension of points must be Mx2, M >= 8');
    end

    numPoints = size(pts1, 1);

    P = [];
    for i = 1 : numPoints
        u_1_i = pts1(i, 1);
        v_1_i = pts1(i, 2);
        u_2_i = pts2(i ,1);
        v_2_i = pts2(i, 2);
        P = [P; ...
             u_1_i*u_2_i, u_1_i*v_2_i, u_1_i, v_1_i*u_2_i, v_1_i*v_2_i, v_1_i, u_2_i, v_2_i, 1];
    end

    [v, d] = eig(P' * P);
    [~, i_min_eig] = min(diag(d));
    F = v(:, i_min_eig);
    F = reshape(F, 3, 3)';
endfunction
