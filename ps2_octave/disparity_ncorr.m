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
    		r_begin = r - half_w;
	        r_end = r + half_w;
	        c_begin = c - half_w;
	        c_end = c + half_w;

    		tpl = L(r_begin : r_end, c_begin : c_end);
    		ncorr = normxcorr2(tpl, R(r_begin : r_end, :));
    		% match_found = false;
    		% while ~match_found
				[y_peak, x_peak] = find(ncorr == max(ncorr(:)));
				c_r = x_peak - half_w; % (y_peak, x_peak) is the right-bottom corner of the patch
				% if abs(c - c_r) > maxd
					% ncorr(y_peak, x_peak) = 0 - ncorr(y_peak, x_peak);
					% continue;
				% else
					% match_found = true;
					% break;
				% end
    		% end
    		D(r, c) = c_r - c;
    	end
    end

    D = D(half_w + 1 : size(D, 1) - half_w, half_w + 1 : size(D, 2) - half_w);
end

