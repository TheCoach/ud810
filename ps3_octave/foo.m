function foo(varargin)
    p = inputParser;
    p.addOptional('optArg1', 5, @(x) isnumeric(x) && mod(x, 2) ~= 0);
    p.addOptional('optArg2', 5, @(x) isnumeric(x) && mod(x, 2) == 0);

    p.parse(varargin{:});
    optArg1 = p.Results.optArg1;
    optArg2 = p.Results.optArg2;

    disp(optArg1);
    disp(optArg2);
end