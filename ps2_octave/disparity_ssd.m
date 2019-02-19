function D = disparity_ssd(L, R, varargin)
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

    half_w = (w - 1) / 2;

	L = padarray(L, [half_w half_w], 'symmetric');
	R = padarray(R, [half_w half_w], 'symmetric');

    D = zeros(size(L));

    for r = 1 + half_w : size(L, 1) - half_w
    	if (mod(r, 10) == 0)
    		fprintf('%d/%d ...\n', r, size(L, 1));
    	end
    	for c =  1 + half_w : size(L, 2) - half_w
    		r_begin = r - half_w;
	        r_end = r + half_w;
	        c_begin = c - half_w;
	        c_end = c + half_w;

    		tpl = L(r_begin : r_end, c_begin : c_end);
    		[ssd, d] = find_best_match_in_strip(tpl, R(r_begin : r_end, :), c, maxd);
    		D(r, c) = d;
    	end
    end

    D = D(half_w + 1 : size(D, 1) - half_w, half_w + 1 : size(D, 2) - half_w);

end

function [min_ssd, marked_d] = find_best_match_in_strip(tpl, strip, c, maxd)
	if size(tpl, 1) ~= size(strip, 1)
		error('Template size is not correct');
	end
	disparity_range = [-maxd maxd];
	half_tplSize = (size(tpl, 1) - 1) / 2;
	min_ssd = realmax();
	marked_d = 0;
	ssds = [];
	for d = disparity_range(1) : disparity_range(2)
		if (c + d < 1 + half_tplSize) % ouf of boundary
			continue;
		end
		if (c + d > size(strip, 2) - half_tplSize)
			break;
		end
		s = sum_squared_differences(tpl, strip(:, c + d - half_tplSize : c + d + half_tplSize));
		ssds = [ssds, s];
		if (s <= min_ssd)
			min_ssd = s;
			marked_d = d;
		end
	end
endfunction

function ssd = sum_squared_differences(tpl, x)
	ssd = sum((tpl(:) - x(:)).^2);
endfunction
