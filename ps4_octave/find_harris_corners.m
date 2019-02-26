function [corner_points, R] = find_harris_corners(harris_R, varargin)
    p = inputParser;
    p.addRequired('harris_R', @(x) size(x, 3) == 1);
    p.addOptional('threshold', 0.5, @(x) isnumeric(x) && x < 1 && x > 0);
    p.addOptional('NHoodSize', 5, @(x) floor(x) == x && mod(x, 2) == 1);
    
    p.parse(harris_R, varargin{:});
    threshold = p.Results.threshold;
    NHoodSize = p.Results.NHoodSize;

    R = harris_R;
    R(find(R < threshold * max(R(:)))) = 0;
 
    internal_R = R;
    corner_points = [];
    while true
        max_r = max(internal_R(:));
        if (max_r == 0)
            break;
        end
        [mr, mc] = ind2sub(size(internal_R), find(internal_R == max_r, 1));
        corner_points = [corner_points; mr, mc];

        nhood_r_begin = mr - floor(NHoodSize / 2);
        nhood_r_end = mr + floor(NHoodSize / 2);
        nhood_c_begin = mc - floor(NHoodSize / 2);
        nhood_c_end = mc + floor(NHoodSize / 2);
        if (nhood_r_begin <= 0)
            nhood_r_begin = 1;
        end
        if (nhood_r_end > size(internal_R)(1))
            nhood_r_end = size(internal_R)(1);
        end

        if (nhood_c_begin <= 0)
            nhood_c_begin = 1;
        end
        if (nhood_c_end > size(internal_R)(2))
            nhood_c_end = size(internal_R)(2);
        end

        internal_R(nhood_r_begin : nhood_r_end, nhood_c_begin : nhood_c_end) = zeros(nhood_r_end - nhood_r_begin + 1, nhood_c_end - nhood_c_begin + 1);
    end
end