function [signal,FPT_MultiChannel] = ReadFileAndAnnotate(fileName)
%read wfdb ecg file, filter in multiple steps and annotate automatically
%file needs to be found in current directory!
% read wfdb ecg file, 12 leads in signal, frequency in Fs
% currentPath = pwd; % only for tryout
[signal,Fs,~]=rdsamp(char(fileName));

%% filtering of the 12-lead ECG
% Remove baseline wander
% usage: [filtered_signal,baseline]=ECG_Baseline_Removal(signal,samplerate,window_length,overlap)
[signal,~] = ECG_Baseline_Removal(signal,Fs,1,0.5);

% filter noise frequencies
% frequencies are already optimized for ECG signals (literature values):
% Lowpass: 120 Hz, Highpass: 0.3 Hz, Bandstop (49-51 Hz)
[signal] = ECG_High_Low_Filter(signal,Fs,1,40);
signal=Notch_Filter(signal,Fs,50,1);

% isoline correction
% usage: [filteredsignal,offset,frequency_matrix,bins_matrix]=Isoline_Correction(signal,varargin)

[signal,~,~,~]=Isoline_Correction(signal);

%% Feature calculation
% produce FPT Table
% usage: [FPT_MultiChannel,FPT_Cell]=Process_ECG_Multi(signal,samplerate,varargin)
[FPT_MultiChannel,~]=Annotate_ECG_Multi(signal,Fs);
end