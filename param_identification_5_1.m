clc; clear all;

%% Load files
base = '.\Group6_Data\4-1a';

in  = load(fullfile(base,'Input-1_5p-1hz.mat'));
vel = load(fullfile(base,'velocity-1_5p_1hz.mat'));

Input = in.Input;      % [t  u]
v     = vel.v;         % [t  v_measured]

% feed Simulink as time-based data
idealInput.time               = Input(:,1);
idealInput.signals.values     = Input(:,2);
idealInput.signals.dimensions = 1;

%% RUN SIM AND PLOT -------------------------------------------------------

model = 'plant_5_1';   % <-- change if your model has a different name

% Load model (without opening UI)
load_system(model);

% Stop time based on input end time
stopT = Input(end,1);

% Run sim
simOut = sim(model, 'StopTime', num2str(stopT));

% Retrieve timeseries from To Workspace block
ideal_v_sim = simOut.get('ideal_v_sim');   % This must match the To Workspace variable name

t_sim = ideal_v_sim.Time;
y_sim = ideal_v_sim.Data;

xwin = [4.4 5.1];
ywin = [-600 600];

figure;

subplot(2,1,1);
plot(Input(:,1), Input(:,2), 'b-');
xlim(xwin); grid on;
title('Input (voltage)');

subplot(2,1,2);
hold on;
plot(v(:,1), v(:,2), 'r-', 'DisplayName','Measured v');
plot(t_sim, y_sim, 'g--', 'LineWidth',1.5, 'DisplayName','Ideal v (Sim)');  % << green line
hold off; grid on;
xlim(xwin); ylim(ywin);
xlabel('Time [s]');
ylabel('Velocity');
%legend('Location','best');
title('Velocity: measured');
