% #### Frequency Response of 2nd Order Sys #### %
clear; close all; clc;

% --== Load data ==-- %

% Input data from 5-1a:
K_p = 1; % Assume 1 from experimental design
K_v = 4000;
tau = 1.2;

topdir_path = '.\Group6_Data\4-3a';
folders = dir(fullfile(topdir_path, 'f_*'));
folders = folders([folders.isdir]);

data = containers.Map();

for i = 1:length(folders)
    folder_name = folders(i).name;
    
    file_name_data = ['x_' folder_name(3:end) '.mat'];
    file_path_data = fullfile(topdir_path, folder_name, file_name_data);

    file_name_target = ['target_' folder_name(3:end) '.mat'];
    file_path_target = fullfile(topdir_path, folder_name, file_name_target);

    entry = struct();

    if isfile(file_path_data)
        fprintf('Loading: %s...\n', file_path_data);
        entry.x = load(file_path_data).x;
    else 
        fprintf('File not found: %s\n', file_path_data);
    end

    if isfile(file_path_target)
        fprintf('Loading: %s...\n', file_path_target);
        entry.target = load(file_path_target).Target;
    else 
        fprintf('File not found: %s\n', file_path_target);
    end

    data(folder_name) = entry;
end

% --== Compute FFT on data ==-- %
Fs = 1000;   % Sampling frequency for fft [Hz]
results = table([], [], [], [], [], [], ...
    'VariableNames', {'test_freq', 'C_mag', 'X_mag', 'time_lag', 'gain', 'phase_delay_deg'});

for i = 1:length(folders)
    folder_name = folders(i).name;

    x = data(folder_name).x(:,2);
    c = data(folder_name).target(:,2);

    % extract last 80% of signal
    N_total = length(x);
    start_indx = round(0.2 * N_total) + 1;
    x = x(start_indx:end);
    c = c(start_indx:end);

    % compute FFT of data
    N = length(x);

    % apply hanning window
    w = hann(N);
    G = mean(w);
    xw = x .* w;
    cw = c .* w;

    X = fft(xw);
    C = fft(cw);

    f = (0:N-1)*(Fs/N);
    f_half = f(1:round(N/2)+1);
    % compute amplitude of fft
    X_mag = (2/N)*abs(X(1:round(N/2)+1))/G;
    C_mag = (2/N)*abs(C(1:round(N/2)+1))/G;
    
    % find dominant frequency bin
    [~, idx_peak] =  max(C_mag(2:end)); %max(X_mag(2:end));
    idx_peak = idx_peak + 1;
    f0 = f_half(idx_peak);

    gain = X_mag(idx_peak)/C_mag(idx_peak);

    % compute phase delay
    phi_x = angle(X(idx_peak));
    phi_c = angle(C(idx_peak));

    phase_diff = phi_x - phi_c;

    phase_diff = mod(phase_diff + pi, 2*pi) - pi; 
    phase_diff_deg = rad2deg(phase_diff);

    time_lag = abs(phase_diff / (2*pi*f0));

    % format data for easy reporting
    test_freq = str2double(strrep(extractAfter(folder_name, 'f_'), '_', '.'));
    
    new_row = {test_freq, C_mag(idx_peak), X_mag(idx_peak), time_lag, gain, phase_diff_deg};
    results = [results ; new_row];
end
results = sortrows(results, 'test_freq')

% --== Compute theoretical response ==-- %
wn = sqrt(K_p*K_v/tau);
zeta = 1/(2*wn*tau);

sys = tf(wn^2, [1 2*zeta*wn wn^2]);

f = linspace(0.5, 50, 1000); % consider setting to 0.5Hz
w = 2*pi.*f;

[H, ~] = freqresp(sys, w);
H = squeeze(H);
mag = abs(H);
phase = angle(H);
phase_deg = rad2deg(phase);

% --== Generate plots ==-- %
figure;
t = tiledlayout(2,1,'TileSpacing', 'compact');

% gain plot
nexttile;
loglog(2*pi.*f, mag, 'LineWidth', 1.5, 'Color', 'r');
hold on;
loglog(2*pi.*results.test_freq, results.gain, 'x', 'LineWidth', 1.5, 'Color', 'b');
ax1 = gca;      
ax1.XTickLabel = [];
ax1.YTickLabel = [];
axis padded;
set(gca, 'YScale', 'log');
title('Experimental vs Theoretical Bode Plots');
ylabel('Gain [mm/mm]');
%grid on; grid minor;

% Bode phase plot
nexttile;
semilogx(2*pi.*f, phase_deg, 'LineWidth', 1.5, 'Color', 'r')
hold on;
semilogx(2*pi.*results.test_freq, results.phase_delay_deg, 'x', 'LineWidth', 1.5, 'Color', 'b');
ax2 = gca;      
ax2.XTickLabel = [];
ax2.YTickLabel = [];
axis padded;
xlabel('Frequency [rad/s]');
ylabel('Phase [deg]');
%grid on; grid minor;