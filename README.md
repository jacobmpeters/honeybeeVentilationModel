## honeybeeVentilationModel
This function contains a user friendly simulation of ventilation dynamics in honeybee nests. 
It is intended to accompany a manuscript submitted to Journal of the Royal Society Interface. 
The function allows for optional input so that the user can easily play with the parameters in 
the model and visualize the resulting dynamics. 
```matlab
% file: simulateVentilationRobinBC.m
%
% written by:   Jacob Peters
% contributed:  Orit Peleg
% model by:     Jacob Peters and L. Mahadevan
% date written: May 11, 2015
% last updated: Nov 5, 2018
%
% FORM: outputStructure = simulateVentilationRobinBC(plotMode,vargin)
%
% Optional input:
%
%'plotMode' is categorical variable controlling how data is visualized
%   0 -> no visualization
%   1 -> visualize rho, T, and V in real time & plot k_on and k_off
%   2 -> visualize using imagesc plots & plot k_on and k_off
%   Defalt: 0
%
%'T_amb' is the ambient temperature (degrees Celsius)
%   Default: 32
%
%'L_x' is the nest entrance size (m)
%   Default: 0.38
%
%'saveInterval' is number of frames between saved frames. Only needed
%   if plotMode is 2 or 3 (i.e., data is being saved).
%   Default: 1
%
%'Dv' is the effective diffusion coefficient for velocity
%   Default: 1E-3

%
%'Dt' is the diffusion coefficient for temperature
%   Default: 5E-5
%
%'m' is parameter that sets slope of behavioral switch function.
%   Default: 0.1
%
%'c' is constant in cooling/heating equation
%   Defalut: 0.05
%
%'nSteps' is the number of time steps in simulation
%   Default: 50000
%
%'alpha' is a coefficient used in the implementation of diffusion of T at 
% the boundaries of the nest entrance (i.e., hive wall).
%   0 -> boundary is perfect conductor (i.e., gradient is T - T_amb
%   1 -> " perfect insulator (i.e., gradient vanishes at boundary)
%   0<alpha<1 -> " somewhere in between
%   Default: 0.9
%
%'seed' is an integer used to seed the random number generator in prior to
%   the simulation. Set seed to 0 to generate random seed and make
%   simulations unique. If a positive integer is used (e.g., 1,2,3,..)
%   simulations will be reproducible.
%   0 <- unique simulation 
%   seed>0 <- reproducible simulation
%   Default: 1
%
% Outputs:
%
%*Note: All outputs are stored in struct called outputStructure. Output
% variables listed are fields within outputStructure.
%
%rho: local fanner density. rho is an mxn matrix where m is the number of
%   time steps and n is the number of x positions.
%
%V: local velocity. V is an mxn matrix where m is the number of time
%   steps and n is the number of x positions.
%
%T: local temperature. T is an mxn matrix where m is the number of time
%   steps and n is the number of x positions.
%
%timeArray: array with data for time axis. Needed because data for every
%   time step may not be output depending on value of plotMode.
%
%optionalInput: a cell array holding values of manually supplied inputs
%
%parameters: a structure holding values for all model parameters
%
%seed: data provided by matlab about the random number generator using rng
%   function. Can be used to replicate simulation.
```
