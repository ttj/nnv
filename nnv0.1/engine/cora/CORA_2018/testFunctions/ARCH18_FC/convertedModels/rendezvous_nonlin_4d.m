function [dx]=rendezvous_nonlin_4d(t,x,u)

	dx(1,1) = x(3);
	dx(2,1) = x(4);
	dx(3,1) = 0.000019143193144076605051541923661988*x(1) + 0.0087505869846717380200443741272328*x(4) - (1434960000000000000.0*(x(1) + 42164000))/((x(1) + 42164000)^2 - x(2)^2)^(3/2) + 807.15359572684597539321366928407;
	dx(4,1) = 0.000019143193144076605051541923661988*x(2) - 0.0087505869846717380200443741272328*x(3) - (1434960000000000000.0*x(2))/((x(1) + 42164000)^2 - x(2)^2)^(3/2);
