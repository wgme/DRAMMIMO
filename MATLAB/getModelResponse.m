function modelResponse = getModelResponse(theta,xdata,extra)
    % 'extra' can be used to pass extra parameter values.
    % These parameters are needed by the model, but not being estimated.
    if extra{1}==0
        a = theta(1);
        b = theta(2);
    else
        a = theta(1);
        b = theta(2);
    end
    
    modelResponse = a*xdata+b;
end