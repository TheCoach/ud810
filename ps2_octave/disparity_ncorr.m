function D = disparity_ncorr(L, R, varargin)
    % Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    %
    % L: Grayscale left image
    % R: Grayscale right image, same size as L
    % D: Output disparity map, same size as L, R

    % TODO: Your code here
    p = inputParser;
    p.addParameter('WindowSize', 15, @isnumeric);
    p.addParameter('MaxDisparity', 16, @isnumeric);

    p.parse(varargin{:});

    w = p.Results.WindowSize;
    maxd = p.Results.MaxDisparity;

    if (size(L) ~= size(R))
        error('The sizes of L and R are not the same');
        return
    end
    if (mod(w, 2)) == 0
	   error('Window size is not odd');
    end

    half_w  = (w - 1) / 2;

    L = padarray(L, [half_w half_w], 'symmetric');
    R = padarray(R, [half_w half_w], 'symmetric');

    D = zeros(size(L));

    for r = 1 + half_w : size(L, 1) - half_w
        if (mod(r, 10) == 0)
            fprintf('%d/%d ...\n', r, size(L, 1));
        end
        for c =  1 + half_w : size(L, 2) - half_w
            tpl_r_begin = r - half_w;
            tpl_r_end = r + half_w;
            tpl_c_begin = c - half_w;
            tpl_c_end = c + half_w;

            tpl = L(tpl_r_begin : tpl_r_end, tpl_c_begin : tpl_c_end);

            strip_c_begin = c - maxd;
            strip_c_end = c + maxd;

            if (strip_c_begin < 1) % ouf of boundary
                strip_c_begin = 1;
            end
            if (strip_c_end > size(R, 2))
                strip_c_end = size(R, 2);
            end

            ncorr = normxcorr2(tpl, R(tpl_r_begin : tpl_r_end, strip_c_begin : strip_c_end));
            [y_peak, x_peak] = find(ncorr == max(ncorr(:)));
            if (length(x_peak) > 1)
                x_peak = min(abs(x_peak - c));
            end
            c_r = (x_peak - half_w) + (strip_c_begin - 1); % (y_peak, x_peak) is the right-bottom corner of the patch
            D(r, c) = c_r - c;
        end
    end

    D = D(half_w + 1 : size(D, 1) - half_w, half_w + 1 : size(D, 2) - half_w);
end
