%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% By: Wei Gao (wg14@my.fsu.edu)
% Last Modified: 07/24/2019
% Desciption:
% 1. Based on the code from Dr. Marko Laine 
%    (http://helios.fmi.fi/~lainema/mcmc/).
% 2. Also based on the math from Dr. Ralph C. Smith 
%    (Uncertainty Quantification: Theory, Implementation, and Applications).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [credLims,predLims] = getDRAMMIMOIntervals(data,model,modelParams,chain_q,chain_cov_err,nSample)
    % Initialize parameters. Using 95% intervals.
    lims = [0.025, 0.5, 0.975];
    m = size(chain_q,1);
    N = length(data.xdata);
    n = size(data.xdata{1},1);
    
    % Set the number of points to be pulled out from the estimation chain.
    if nargin<5 || isempty(nSample)
        nSample = m;
    end
    
    % Get the indices of points to be pulled out of the estimation chain.
    if nSample == m
        isample = 1:1:m;
    else
        isample = ceil(rand(nSample,1)*m);
    end
    
    % Sample the estimation chain for the credible region as ysave and the prediction region as osave.
    ysave = zeros(nSample,n,N);
    osave = zeros(nSample,n,N);
    for iisample = 1:1:nSample
        qi = chain_q(isample(iisample),:)';
        randError = mvnrnd(zeros(N,1),chain_cov_err(:,:,isample(iisample)));
        for i = 1:1:N
            y = model.fun{i}(qi,data.xdata{i},modelParams.extra{i});
            ysave(iisample,:,i) = y';
            osave(iisample,:,i) = y'+randError(i);
        end
    end
    
    % Interpolate the credible and prediction intervals.
    credLims = zeros(length(lims),n,N);
    predLims = zeros(length(lims),n,N);
    for i = 1:1:N
        credLims(:,:,i) = interp1(sort(ysave(:,:,i)),(size(ysave(:,:,i),1)-1)*lims+1);
        predLims(:,:,i) = interp1(sort(osave(:,:,i)),(size(osave(:,:,i),1)-1)*lims+1);
    end
end