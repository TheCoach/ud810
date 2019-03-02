function R = compute_harris_value(imgGray, varargin)
    p = inputParser;
    p.addRequired('imgGray', @(x) size(x, 3) == 1);
    p.addOptional('windowSize', 5, @(x) floor(x) == x && mod(x, 2) == 1);
    p.addOptional('alpha', 0.04, @(x) isnumeric(x) && x > 0);
    
    p.parse(imgGray, varargin{:});
    alpha = p.Results.alpha;
    wSz = p.Results.windowSize;

    halfWsz = (wSz - 1) / 2;
    imgGray = double(imgGray);
    
    f_g = fspecial('gaussian');
    [dgx, dgy] = imgradientxy(f_g);

    gx = imfilter(imgGray, dgx, 'conv');
    gy = imfilter(imgGray, dgy, 'conv');

    gxx = gx .^ 2;
    gxy = gx .* gy;
    gyy = gy .^ 2;

    w = fspecial('gaussian', wSz);

    R = zeros(size(gx));

    for r = halfWsz + 1 : size(gx, 1) - halfWsz
        if mod(r, 20) == 0
            printf('Processing row %d/%d\n', r, size(imgGray, 1));
        end
        for c = halfWsz + 1 : size(gx, 2) - halfWsz
            wGxx = gxx(r - halfWsz : r + halfWsz, c - halfWsz : c + halfWsz); 
            wGxy = gxy(r - halfWsz : r + halfWsz, c - halfWsz : c + halfWsz);
            wGyx = gxy(r - halfWsz : r + halfWsz, c - halfWsz : c + halfWsz);
            wGyy = gyy(r - halfWsz : r + halfWsz, c - halfWsz : c + halfWsz);

            Mxx = sum((w .* wGxx)(:));
            Mxy = sum((w .* wGxy)(:));
            Myx = sum((w .* wGyx)(:));
            Myy = sum((w .* wGyy)(:));

            M = [Mxx, Mxy; Myx, Myy];

            R(r, c) = det(M) - alpha * (trace(M) ^ 2);
        end
    end

    R = R(halfWsz + 1 : size(R, 1) - halfWsz, halfWsz + 1 : size(R, 2) - halfWsz);
    R(find(R < 0)) = 0;
end
