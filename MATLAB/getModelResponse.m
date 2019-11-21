function modelResponse = getModelResponse(theta, xdata, extra)
    % The extra cell is used to pass extra parameters.
    % Here the first cell of extra is used to control model mode,
    % e.g. there could be a mode where only "a" is estimated and "b" is fixed,
    % instead of "a" and "b" being estimated at the same time.
    if extra{1}==0
        a = theta(1);
        b = theta(2);
    else
        a = theta(1);
        b = 0;
    end
    
    % The extra cell can also pass the values of parameters that are never
    % estimated in any scenario.
    modelResponse = a * xdata + b;
end