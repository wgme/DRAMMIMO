%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% By: Wei Gao (wg14@my.fsu.edu)
% Last Modified: 11/20/2019
% Desciption:
% 1. Based on the code from Dr. Marko Laine 
%    (http://helios.fmi.fi/~lainema/mcmc/).
% 2. Also based on the math from Dr. Ralph C. Smith 
%    (Uncertainty Quantification: Theory, Implementation, and Applications).
% V01: Null.
% V02a: Null.
% V02b: The code was used for dissertation, named propErrDRAMMIMO(). 
%       The code was a modified version of the mcmcpred() function in 
%       Dr. Marko Laine's MCMC toolbox for MATLAB.
% V03: The name was changed from propErrDRAMMIMO() to getDRAMMIMOIntervals().
% V04: The code for saving results was added.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data: cell array, size = 1 * N. 
%       Each cell is a vector, size = n * 1.
%       N = number of data sets (N=1 for Bayesian, N>1 for Max Entropy).
%       n = number of observation points.
% model: struct.
%        .fun, cell array, size = 1 * N. Each cell is a function handle.
%        .errFun, cell array, size = 1 * N. Each cell is a function handle.
% modelParams: struct.
%              .names, cell array, size = 1 * p. Each cell is a string.
%              .values, cell array, size = 1 * p. Each cell is a scalar.
%              .lowerLimits, cell array, size = 1 * p. Each cell is a scalar.
%              .upperLimits, cell array, size = 1 * p. Each cell is a scalar.
%              .extra, cell array, size = 1 * p. Each cell is a cell array.
%              p = number of parameters to be estimated.
% chain_q: matrix, m * p.
%          m = number of iterations selected.
% chain_cov_err: matrix, m * N * N.
% nSample: scalar, rule of thumb value is 500.
% credLims: matrix, 3 * p * N.
% predLims: matrix, 3 * p * N.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [credLims,predLims] = getDRAMMIMOIntervals(data,model,modelParams,chain_q,chain_cov_err,nSample)
    disp('--------------------------------------------------');
    disp('Initializing credible and prediction intervals...');
    
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
    
    disp('Generating credible and prediction intervals...');
    
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
    
    % Save the results.
    intervals.credLims = credLims;
    intervals.predLims = predLims;
    save('intervals.mat','intervals');
    
    disp('Credible and prediction intervals generated.');
    disp('--------------------------------------------------');
end