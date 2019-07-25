function modelResponseError = getModelResponseError(theta,xdata,ydata,extra)
    % 'extra' can be used to pass extra parameter values.
    % These parameters are needed by the model, but not being estimated.
    modelResponse = getModelResponse(theta,xdata,extra);
    modelResponseError = modelResponse-ydata;
end