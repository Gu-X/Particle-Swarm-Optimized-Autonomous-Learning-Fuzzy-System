clear all
clc
close all

%%
% load autompg_train;
% load autompg_test;
% TRAIN=autompg_train;
% TEST=autompg_test;
% Y0 = TRAIN(:,7);
% data0 = TRAIN(:,1:6);
% Y1 = TEST(:,7);
% data1 = TEST(:,1:6);
%%
% load deltaailerons_train;
% load deltaailerons_test;
% TRAIN=deltaailerons_train;
% TEST=deltaailerons_test;
% Y0 = TRAIN(:,6);
% data0 = TRAIN(:,1:5);
% Y1 = TEST(:,6);
% data1 = TEST(:,1:5);
%%
% load triazines_train;
% load triazines_test;
% TRAIN=triazines_train;
% TEST=triazines_test;
% Y0 = TRAIN(:,61);
% data0 = TRAIN(:,1:60);
% Y1 = TEST(:,61);
% data1 = TEST(:,1:60);
%%
load autos_train;
load autos_test;
TRAIN=autos_train;
TEST=autos_test;
Y0= TRAIN(:,16);
data0 = TRAIN(:,1:15);
Y1 = TEST(:,16);
data1 = TEST(:,1:15);
%%
% load calhousing_train;
% load calhousing_test;
% TRAIN=calhousing_train;
% TEST=calhousing_test;
% Y0 = TRAIN(:,9);
% data0 = TRAIN(:,1:8);
% Y1 = TEST(:,9);
% data1 = TEST(:,1:8);


Input.MaxIt=200;
Input.nPop=100;
Input.w=0.7298;
Input.cs=1.49618;
Input.cp=1.49618;
Input.c=1.49618;
Input.wdamp=1;
Input.data0=data0;
Input.data1=data1;
Input.Y0=Y0;
Input.Y1=Y1;
[Output]=PSOALFS(Input);
Prediction=Output.PreSeq;
Output.RMSE