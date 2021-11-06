% code for assignment 2 Q1
%% fuzzy control system
% load pre-defined fuzzy interface system (FIS)
fis = readfis('truck_parking_control.fis');
% plotfis(fis)
% initialization
L = 0.1;
T = 2.5;
v = 0.5;
theta = -10;
y_position = 10;
x_position = 50;
deign_output = 0;
y = [];
x = [];
angle = [];
u_value = [];
iteration_count = 1;
% iteration
while (abs(y_position - deign_output)>1e-3) ||...
        (abs(theta - deign_output)>1e-3)
    u = evalfis(fis, [y_position - deign_output, theta - deign_output]);
    theta = theta + v * T * tand(u) / L;
    y_position = y_position + v * T * sind(theta);
    x_position = x_position + v * T * cosd(theta);
    y(iteration_count) = y_position;
    x(iteration_count) = x_position;
    angle(iteration_count) = theta;
    u_value(iteration_count) = u;
    iteration_count = iteration_count + 1;
end
% plot results
time = 1:iteration_count-1;
figure(1)
subplot(221)
plot(time,u_value,'LineWidth',1.5);
ylabel('бу','Fontname','Times New Roman')
ylim([-30 30])
xlabel('Iteration/t(s)','Fontname','Times New Roman')
title('u(t)','Fontname','Times New Roman')
subplot(222)
plot(time,y,'LineWidth',1.5);
ylabel('m','Fontname','Times New Roman')
ylim([-100 100])
xlabel('Iteration/t(s)','Fontname','Times New Roman')
title('y(t)','Fontname','Times New Roman')
subplot(223)
plot(time,angle,'LineWidth',1.5);
ylabel('бу','Fontname','Times New Roman')
ylim([-180 180])
xlabel('Iteration/t(s)','Fontname','Times New Roman')
title('theta(t)','Fontname','Times New Roman')
subplot(224)
plot(time,x,'LineWidth',1.5);
ylabel('m','Fontname','Times New Roman')
xlabel('Iteration/t(s)','Fontname','Times New Roman')
title('x(t)','Fontname','Times New Roman')
%% non-fuzzy control system
% propotional gain of y and theta
ky = -2.8;
ktheta = -0.1;
% initialization
L = 0.1;
T = 2.5;
v = 0.5;
theta = -10;
y_position = 10;
x_position = 50;
deign_output = 0;
y = [];
x = [];
angle = [];
u_value = [];
iteration_count = 1;
% iteration
while (abs(y_position - deign_output)>1e-3) ||...
        (abs(theta - deign_output)>1e-3)
    % proportional controller
    u = (y_position - deign_output) * ky + (theta - deign_output) * ktheta;
    % make sure that u is valid
    if u > 30
        u = u - 60;
    elseif u < -30
        u = u + 60;
    end
    theta = theta + v * T * tand(u) / L;
    % make sure that theta is valid
    if theta > 180
        theta = theta - 360;
    elseif theta < -180
        theta = theta + 360;
    end
    y_position = y_position + v * T * sind(theta);
    % make sure that y is valid
    if y_position > 100
        y_position = y_position - 200;
    elseif y_position < -100
        y_position = y_position + 200;
    end
    x_position = x_position + v * T * cosd(theta);
    y(iteration_count) = y_position;
    x(iteration_count) = x_position;
    angle(iteration_count) = theta;
    u_value(iteration_count) = u;
    iteration_count = iteration_count + 1;
end
% plot results
time = 1:iteration_count-1;
figure(1)
subplot(221)
plot(time,u_value,'LineWidth',1.5);
ylabel('бу','Fontname','Times New Roman')
% ylim([-30 30])
xlabel('Iteration/t(s)','Fontname','Times New Roman')
title('u(t)','Fontname','Times New Roman')
subplot(222)
plot(time,y,'LineWidth',1.5);
ylabel('m','Fontname','Times New Roman')
% ylim([-100 100])
xlabel('Iteration/t(s)','Fontname','Times New Roman')
title('y(t)','Fontname','Times New Roman')
subplot(223)
plot(time,angle,'LineWidth',1.5);
ylabel('бу','Fontname','Times New Roman')
% ylim([-180 180])
xlabel('Iteration/t(s)','Fontname','Times New Roman')
title('theta(t)','Fontname','Times New Roman')
subplot(224)
plot(time,x,'LineWidth',1.5);
ylabel('m','Fontname','Times New Roman')
xlabel('Iteration/t(s)','Fontname','Times New Roman')
title('x(t)','Fontname','Times New Roman')

