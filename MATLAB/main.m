close all;clear;clc;

%% Below is an example of using the DRAMMIMO package.

% Load the data.
% Fictitious data are generated here for a linear model y = a*x+b.
% Two data sets are available.
n = 101;
p = 2;
N = 2;
inputData1 = linspace(0,1,n)';
inputData2 = linspace(0,1,n)';
outputData1 = 0.8*inputData1+0.05*randn(n,1);
outputData2 = 1.2*inputData2+0.15*randn(n,1);

% Set up DRAMMIMO.
% Scenarios with one or two data sets are differentiated by mode.
% mode = 1: using one data set (Bayesian method).
% mode = 2: using two data sets (Maximum Entropy method) 
mode = 2;
if mode == 1
    data.xdata = {inputData1};
    data.ydata = {outputData1};
    model.fun = {@getModelResponse};
    model.errFun = {@getModelResponseError};
    modelParams.names = {'a','b'};
    modelParams.values = {1,0};
    modelParams.lowerLimits = {-inf,-inf};
    modelParams.upperLimits = {inf,inf};
    modelParams.extra = {{0}};
elseif mode == 2
    data.xdata = {inputData1,inputData2};
    data.ydata = {outputData1,outputData2};
    model.fun = {@getModelResponse,@getModelResponse};
    model.errFun = {@getModelResponseError,@getModelResponseError};
    modelParams.names = {'a','b'};
    modelParams.values = {1,0};
    modelParams.lowerLimits = {-inf,-inf};
    modelParams.upperLimits = {inf,inf};
    modelParams.extra = {{0},{0}};
end

% Get estimation chains.
% The estimation chains can be obtained in multiple consecutive runs.
% 1st round.
numDone = 1;
numTotal = 5000;
DRAMParams.numDRAMIterationsDone = numDone;
DRAMParams.numDRAMIterations = numTotal;
DRAMParams.previousResults.chain_q = [];
DRAMParams.previousResults.last_cov_q = [];
DRAMParams.previousResults.chain_cov_err = [];
[chain_q,last_cov_q,chain_cov_err] = getDRAMMIMOChains(data,model,modelParams,DRAMParams);
% 2nd round.
numDone = numTotal;
numTotal = 10000;
DRAMParams.numDRAMIterationsDone = numDone;
DRAMParams.numDRAMIterations = numTotal;
DRAMParams.previousResults.chain_q = chain_q;
DRAMParams.previousResults.last_cov_q = last_cov_q;
DRAMParams.previousResults.chain_cov_err = chain_cov_err;
[chain_q,last_cov_q,chain_cov_err] = getDRAMMIMOChains(data,model,modelParams,DRAMParams);

% Get posterior densities.
num = round(size(chain_q,1)/2)+1;
[vals,probs] = getDRAMMIMODensities(chain_q(num:end,:));

% Get credible and prediction intervals.
nSample = 500;
[credLims,predLims] = getDRAMMIMOIntervals(data,model,modelParams,chain_q(num:end,:),chain_cov_err(:,:,num:end),nSample);

%% Plot the results.

figNum = 0;

% Data.
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
set(gca,'fontsize',24,'xtick',[],'xlim',[0,numTotal],'ylim',[min(chain_q(:,1)),max(chain_q(:,1))]);
box on;
ylabel('a');
subplot(2,1,2);
hold on;
plot(1:1:size(chain_q,1),chain_q(:,2),'b.');
hold off;
set(gca,'fontsize',24,'xtick',[],'xlim',[0,numTotal],'ylim',[min(chain_q(:,2)),max(chain_q(:,2))]);
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

% Credible and prediction intervals for data set I.
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

% Credible and prediction intervals for data set II.
if mode == 2
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
end