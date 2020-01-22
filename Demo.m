clear all
clc
close all

%% Loading Autompg data
% load autompg_train;
% load autompg_test;
% TRAIN=autompg_train;
% TEST=autompg_test;
% Y0 = TRAIN(:,7);
% data0 = TRAIN(:,1:6);
% Y1 = TEST(:,7);
% data1 = TEST(:,1:6);
%% Loading Delta Ailerons data
% load deltaailerons_train;
% load deltaailerons_test;
% TRAIN=deltaailerons_train;
% TEST=deltaailerons_test;
% Y0 = TRAIN(:,6);
% data0 = TRAIN(:,1:5);
% Y1 = TEST(:,6);
% data1 = TEST(:,1:5);
%% Loading Triazines data
% load triazines_train;
% load triazines_test;
% TRAIN=triazines_train;
% TEST=triazines_test;
% Y0 = TRAIN(:,61);
% data0 = TRAIN(:,1:60);
% Y1 = TEST(:,61);
% data1 = TEST(:,1:60);
%% Loading Autos data
load autos_train;
load autos_test;
TRAIN=autos_train;
TEST=autos_test;
Y0= TRAIN(:,16);
data0 = TRAIN(:,1:15);
Y1 = TEST(:,16);
data1 = TEST(:,1:15);
%% Loading Calhousing data
% load calhousing_train;
% load calhousing_test;
% TRAIN=calhousing_train;
% TEST=calhousing_test;
% Y0 = TRAIN(:,9);
% data0 = TRAIN(:,1:8);
% Y1 = TEST(:,9);
% data1 = TEST(:,1:8);


Input.data0=data0; % training input
Input.data1=data1; % testing input
Input.Y0=Y0; % training output
Input.Y1=Y1; % testing output

Input.MaxIt=200; % maximum iteration number for PSO algorithm
Input.nPop=100; % population number for PSO algorithm
Input.w=0.7298; % inertia weight for PSO algorithm
Input.cp=1.49618; % c1 for PSO algorithm
Input.cs=1.49618; % c2 for PSO algorithm
Input.wdamp=1; % damping coefficient for PSO algorithm

[Output]=PSOALFS(Input);
Prediction=Output.PreSeq; % predicted output 
Output.RMSE % RMSE of the prediction
