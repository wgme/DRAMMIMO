close all; clear; clc; format short g;
%% Below is an example of using the DRAMMIMO package.
% Two simple linear models y1 = a * x1 + b and y2 = a * x2 + b are considered.
% These two models share parameters a and b, such that the Maximum Entropy
% method can be taken advantage of.
% Two sets of ficticious data are generated using the models with Gaussian 
% noise, i.e. y1 = a1 * x1 + b + epsilon1 and y2 = a2 * x2 + b + epsilon2.
% A discrepancy is introduced between a1 and a2, with different magnitudes 
% of noise, epsilon1 and epsilon2.

%% Load the data.

disp('Loading data...');

inputData1 = linspace(0, 1, 101)';
inputData2 = linspace(0, 1, 101)';
% outputData1 = 0.8 * inputData1 .* (1 + 0.05 * randn(101, 1));
% outputData2 = 1.2 * inputData2 .* (1 + 0.10 * randn(101, 1));
outputData1 = 0.8 * inputData1 + 0.05 * randn(101, 1);
outputData2 = 1.2 * inputData2 + 0.10 * randn(101, 1);

%% Set up the DRAMMIMO.

disp('Setting DRAMMIMO...');

% This example has two data sets.
% Set the data struct.
data.xdata = {inputData1, inputData2};
data.ydata = {outputData1, outputData2};
% Set the model struct.
model.fun = {@getModelResponse, @getModelResponse};
model.errFun = {@getModelResponseError, @getModelResponseError};
% Set the modelParams struct.
modelParams.names = {'a', 'b'};
modelParams.values = {1, 0};
modelParams.lowerLimits = {-inf, -inf};
modelParams.upperLimits = {inf, inf};
modelParams.extra = {{0}, {0}};
% Set the DRAMParams struct.
% Number of iterations already done.
DRAMParams.numIterationsDone = 1;
% Number of iterations when done.
DRAMParams.numIterationsExpected = 5000;
% Every X number of iterations, display current estimation.
DRAMParams.numIterationsDisplay = 200;
% Every X number of iterations, save the estimation chains. 
DRAMParams.numIterationsSave = 1000;
% For initial run, the previousResults struct is empty.
DRAMParams.previousResults.prior.psi_s = [];
DRAMParams.previousResults.prior.nu_s = [];
DRAMParams.previousResults.chain_q = [];
DRAMParams.previousResults.last_cov_q = [];
DRAMParams.previousResults.chain_cov_err = [];

%% Run the DRAMMIMO.

% The uncertainty quantification results consist of three parts:
% 1. Estimation chains.
% 2. Posterior densities.
% 3. Credible and prediction intervals.

disp('Running DRAMMIMO...');

% Get the estimation chains.
% The estimation chains can be obtained in multiple runs.
% 1st run.
[prior, chain_q, last_cov_q, chain_cov_err] = ...
    getDRAMMIMOChains(data, model, modelParams, DRAMParams);
% 2nd run.
% Need to set the DRAMParams struct for continuous runs.
DRAMParams.numIterationsDone = 5000;
DRAMParams.numIterationsExpected = 10000;
DRAMParams.numIterationsDisplay = 200;
DRAMParams.numIterationsSave = 1000;
DRAMParams.previousResults.prior.psi_s = prior.psi_s;
DRAMParams.previousResults.prior.nu_s = prior.nu_s;
DRAMParams.previousResults.chain_q = chain_q;
DRAMParams.previousResults.last_cov_q = last_cov_q;
DRAMParams.previousResults.chain_cov_err = chain_cov_err;
[prior, chain_q, last_cov_q, chain_cov_err] = ...
    getDRAMMIMOChains(data, model, modelParams, DRAMParams);

% Get the posterior densities.
% Assuming the second half of the chains are in steady-state.
num = round(size(chain_q,1)/2)+1;
[vals,probs] = getDRAMMIMODensities(chain_q(num:end, :));

% Get the credible and prediction intervals.
% 500 is the rule of thumb number.
nSample = 500;
[credLims,predLims] = ...
    getDRAMMIMOIntervals(data, model, modelParams, ...
                         chain_q(num:end,:),chain_cov_err(:,:,num:end),...
                         nSample);

%% Display the results.

disp('Presenting results...');

% Mean parameter estimation.
disp('Mean Parameter Estimation = ');
disp(mean(chain_q(num:end,:)));

% Plot the results.
figNum = 0;

% Raw data.
figNum = figNum+1;
fh = figure(figNum);
set(fh,'outerposition',96*[2,2,7,6]);
hold on;
h(1) = plot(inputData1,outputData1,'bo');
h(2) = plot(inputData2,outputData2,'ro');
hold off;
box on;
set(gca,'fontsize',24,'xlim',[0,1],'ylim',[-0.1,1.3]);
xlabel('x');
ylabel('y');
legend(h,'Data I','Data II','location','nw');
legend boxoff;

% Estimation chains.
figNum = figNum+1;
fh = figure(figNum);
set(fh,'outerposition',96*[2,2,7,6]);
subplot(2,1,1);
hold on;
plot(1:1:size(chain_q,1),chain_q(:,1),'b.');
hold off;
set(gca,'fontsize',24,'xtick',[],...
    'xlim',[0,DRAMParams.numIterationsExpected],'ylim',[min(chain_q(:,1)),max(chain_q(:,1))]);
box on;
ylabel('a');
subplot(2,1,2);
hold on;
plot(1:1:size(chain_q,1),chain_q(:,2),'b.');
hold off;
set(gca,'fontsize',24,'xtick',[],...
    'xlim',[0,DRAMParams.numIterationsExpected],'ylim',[min(chain_q(:,2)),max(chain_q(:,2))]);
box on;
ylabel('b');
xlabel('Iterations');

% Posterior densities.
figNum = figNum+1;
fh = figure(figNum);
set(fh,'outerposition',96*[2,2,7,6]);
subplot(1,2,1);
hold on;
plot(vals(:,1),probs(:,1),'k','linewidth',3);
hold off;
set(gca,'fontsize',24,'xlim',[min(vals(:,1)),max(vals(:,1))],'ytick',[]);
box on;
xlabel('a');
ylabel('Posterior Density');
subplot(1,2,2);
hold on;
plot(vals(:,2),probs(:,2),'k','linewidth',3);
hold off;
set(gca,'fontsize',24,'xlim',[min(vals(:,2)),max(vals(:,2))],'ytick',[]);
box on;
xlabel('b');

% Credible and prediction interval for Data I.
figNum = figNum+1;
fh = figure(figNum);
set(fh,'outerposition',96*[2,2,7,6]);
set(gca,'fontsize',24,'xlim',[0,1],'ylim',[-0.7,2]);
hold on;
h(1) = patch([data.xdata{1}',fliplr(data.xdata{1}')],...
      [predLims(1,:,1),fliplr(predLims(3,:,1))],...
      [1,0.75,0.5],'linestyle','none');
h(2) = patch([data.xdata{1}',fliplr(data.xdata{1}')],...
      [credLims(1,:,1),fliplr(credLims(3,:,1))],...
      [0.75,1,0.5],'linestyle','none');
h(3) = plot(inputData1,credLims(2,:,1),'k');
h(4) = plot(inputData1,outputData1,'bo');
hold off;
box on;
lh = legend(h,'95% Pred Interval','95% Cred Interval','Model','Data I','location','nw');
lh.FontSize = 18;
legend boxoff;
xlabel('x');
ylabel('y_1');

% Credible and prediction interval for Data II.
figNum = figNum+1;
fh = figure(figNum);
set(fh,'outerposition',96*[2,2,7,6]);
set(gca,'fontsize',24,'xlim',[0,1],'ylim',[-0.7,2]);
hold on;
h(1) = patch([data.xdata{2}',fliplr(data.xdata{2}')],...
      [predLims(1,:,2),fliplr(predLims(3,:,2))],...
      [1,0.75,0.5],'linestyle','none');
h(2) = patch([data.xdata{2}',fliplr(data.xdata{2}')],...
      [credLims(1,:,2),fliplr(credLims(3,:,2))],...
      [0.75,1,0.5],'linestyle','none');
h(3) = plot(inputData1,credLims(2,:,2),'k');
h(4) = plot(inputData2,outputData2,'ro');
hold off;
box on;
lh = legend(h,'95% Pred Interval','95% Cred Interval','Model','Data II','location','nw');
lh.FontSize =18;
legend boxoff;
xlabel('x');
ylabel('y_2');