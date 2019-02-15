function peaks = hough_peaks(H, varargin)
    % Find peaks in a Hough accumulator array.
    %
    % Threshold (optional): Threshold at which values of H are considered to be peaks
    % NHoodSize (optional): Size of the suppression neighborhood, [M N]
    %
    % Please see the Matlab documentation for houghpeaks():
    % http://www.mathworks.com/help/images/ref/houghpeaks.html
    % Your code should imitate the matlab implementation.

    %% Parse input arguments
    p = inputParser;
    p.addOptional('numpeaks', 1, @isnumeric);
    p.addParameter('Threshold', 0.5 * max(H(:)));
    p.addParameter('NHoodSize', floor(size(H) / 100.0) * 2 + 1);  % odd values >= size(H)/50
    p.parse(varargin{:});

    numpeaks = p.Results.numpeaks;
    threshold = p.Results.Threshold;
    nHoodSize = p.Results.NHoodSize;

    % TODO: Your code here
    peaks = [];
    done = false;
    H_internal = H;
    while not(done)
        [m, m_idx] = max(H_internal(:));
        if (m < threshold) 
            break;
        end
        [max_r, max_c] = ind2sub(size(H_internal), m_idx);
        peaks = [peaks; max_r, max_c];


        r_begin = max_r - (nHoodSize(1) - 1) / 2;
        r_end = max_r + (nHoodSize(1) - 1) / 2;
        if (r_begin <= 0)
            r_begin = 1;
        end
        if (r_end > size(H)(1))
            r_end = size(H)(1);
        end

        c_begin = max_c - (nHoodSize(2) - 1) / 2;
        c_end = max_c + (nHoodSize(2) - 1) / 2;
        if (c_begin <= 0)
            c_begin = 1;
        end
        if (c_end > size(H)(2))
            c_end = size(H)(2);
        end

        H_internal(r_begin : r_end, c_begin : c_end) = zeros(r_end - r_begin + 1, c_end - c_begin + 1);
        if length(peaks) >= numpeaks
            break;
        end
    end
end