function R = compute_harris_value(gx, gy, varargin)
    p = inputParser;
    p.addRequired('gx', @(x) size(x, 3) == 1);
    p.addRequired('gy', @(x) size(x, 3) == 1);
    p.addOptional('windowSize', 5, @(x) floor(x) == x && mod(x, 2) == 1);
    p.addOptional('alpha', 0.04, @(x) isnumeric(x) && x > 0);
    
    p.parse(gx, gy, varargin{:});
    alpha = p.Results.alpha;
    wSz = p.Results.windowSize;

    if ~isequal(size(gx), size(gy))
        error('gx and gy must be of the same size.');
    end
    halfWsz = (wSz - 1) / 2;

    gx = padarray(gx, [halfWsz halfWsz], 'replicate');
    gy = padarray(gy, [halfWsz halfWsz], 'replicate');

    R = zeros(size(gx));

    for r = halfWsz + 1 : size(gx, 1) - halfWsz
        printf('working on row %d ...\n', r);
        for c = halfWsz + 1 : size(gx, 2) - halfWsz
            w_Gx = gx(r - halfWsz : r + halfWsz, c - halfWsz : c + halfWsz); 
            w_Gy = gy(r - halfWsz : r + halfWsz, c - halfWsz : c + halfWsz);
            M = compute_M(w_Gx, w_Gy, wSz);
            R(r, c) = det(M) - alpha * (trace(M) ^ 2);
            %{
                R < 0, edge
                large R > 0, corner
                magnitude(R) small, flat region
            %}
        end
    end

    R = R(halfWsz + 1 : size(R, 1) - halfWsz, halfWsz + 1 : size(R, 2) - halfWsz);
    R(find(R < 0)) = 0;
end

function M = compute_M(w_Gx, w_Gy)
    M = zeros(2);
    w = fspecial('gaussian', size(w_Gx));
    for r = 1 : size(w_Gx, 1)
        for c = 1 : size(w_Gx, 2)
            M += w(r, c) .* ...
                  [w_Gx(r, c) ^ 2, w_Gx(r, c) * w_Gy(r, c); ...
                  w_Gx(r, c) * w_Gy(r, c), w_Gy(r, c) ^ 2];
        end
    end
endfunction