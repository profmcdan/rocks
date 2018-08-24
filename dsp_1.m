%% Create the signal from codes 
close all
f = [1000, 2500]; % Hz
fs = 40 * f(1); % Smapling Freq

t = 0: 1/fs : 2/f(1);

A = 1.0;

x1 = A * sin(2*pi*f(1)*t);
x2 = A * sin(2*pi*f(2)*t);

x = x1 + x2;

len = length(x);

noise = randn(1, len);

x_noisy = x + noise;
plot(t,x_noisy)


%% Calculate the PSD
N = 1024;
X = fft(x_noisy, N);

freq = fs * (0:N-1)/N;
len = length(freq);
Power = X.*conj(X)/N;
figure;
plot(freq(1:len/2), Power(1:len/2))



%% Another method is to use System Objects
sigObj1 = dsp.SineWave('Amplitude', A, 'Frequency', f(1), 'SampleRate', fs);
sigObj1.SamplesPerFrame = 1024;
sig1 = sigObj();

sigObj2 = dsp.SineWave('Amplitude', A, 'Frequency', f(2), 'SampleRate', fs);
sigObj2.SamplesPerFrame = 1024;
sig2 = sigObj();

figure; plot(sig)

specScope = dsp.SpectrumAnalyzer;
specScope.SampleRate = fs;
specScope.SpectralAverages = 1;
specScope.PlotAsTwoSidedSpectrum = false;
specScope.RBWSource = 'Auto';
specScope.PowerUnits = 'dBW';


for i = 1;100
    specScope(sig1 + sig2);
end
%% Simulink

