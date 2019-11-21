function modelResponseError = getModelResponseError(theta, xdata, ydata, extra)
    % Get the model response based on the parameter values in theta.
    modelResponse = getModelResponse(theta,xdata,extra);
    % Get the error between model prediction and data.
    modelResponseError = modelResponse-ydata;
end