function [A] = CreateOutputArray(signal,FPT_MultiChannel)
%read ecg signal data Nx12 and FPT_MultiChannel from EcgDeli Annotate_ECG_Multi
% initialise output array and write median of inout in first column

%disp("The size of the raw ecg signal is ")
%disp(size(signal))

% 12 leads and 6 features
A = zeros(size(signal,1),size(signal,2) + 6);

% A(:,1) = median(signal,2); no longer calculate median, use raw leads
% instead
A(:, 1:size(signal,2)) = signal;

% iterate through signal and create array marking area and peak of features
% with 1 in output array

for i=1:size(FPT_MultiChannel,1)

    % create output for P-wave
    A(FPT_MultiChannel(i,1):FPT_MultiChannel(i,3),size(signal,2)+1) = 1;
    A(FPT_MultiChannel(i,2),size(signal,2)+2) = 1;

    % create output for QRS-complex
    A(FPT_MultiChannel(i,4):FPT_MultiChannel(i,8),size(signal,2)+3) = 1;
    A(FPT_MultiChannel(i,6),size(signal,2)+4) = 1;

    % create output for T-wave
    A(FPT_MultiChannel(i,10):FPT_MultiChannel(i,12),size(signal,2)+5) = 1;
    A(FPT_MultiChannel(i,11),size(signal,2)+6) = 1;

end
end