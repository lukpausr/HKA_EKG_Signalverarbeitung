function [A] = CreateOutputArray(signal,FPT_MultiChannel)
%read ecg signal data Nx12 and FPT_MultiChannel from EcgDeli Annotate_ECG_Multi
% initialise output array and write median of inout in first column
A = zeros(size(signal,1),7);
A(:,1) = median(signal,2);

% iterate through signal and create array marking area and peak of features
% with 1 in output array
for i=1:size(FPT_MultiChannel,1)
    % create output for P-wave
    A(FPT_MultiChannel(i,1):FPT_MultiChannel(i,3),2) = 1;
    A(FPT_MultiChannel(i,2),3) = 1;

    % create output for QRS-complex
    A(FPT_MultiChannel(i,4):FPT_MultiChannel(i,8),4) = 1;
    A(FPT_MultiChannel(i,6),5) = 1;

    % create output for T-wave
    A(FPT_MultiChannel(i,10):FPT_MultiChannel(i,12),6) = 1;
    A(FPT_MultiChannel(i,11),7) = 1;
end
end