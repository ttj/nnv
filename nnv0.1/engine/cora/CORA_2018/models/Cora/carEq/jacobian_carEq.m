function [A,B]=jacobian_carEq(x,u)

A=[0,1;...
0,- x(2)/5 - 1/2];

B=[0;...
1];

