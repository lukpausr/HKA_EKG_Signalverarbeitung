% clear
close

% ptbList = physionetdb('ptbdb'); % list all files in ptb data base
% wfdb2mat('ptbdb/patient001/s0014lre') % safe signal files as *.mat and *.hea files
path = 'ptbdb\patient001\s0014lre';

ecgpuwave(path,'test'); % annotate file (qrs, p wave and t wave)
[signal,Fs,tm]=rdsamp(path);

t_waves=rdann(path,'test',[],[],[],'t');
% p_waves=rdann('s0014lre','test',[],[],[],'p');
% q_peaks=rdann('s0014lre','test',[],[],[],'q')

plot(tm,signal(:,1));hold on;grid on
plot(tm(t_waves),signal(t_waves),'or')
% plot(tm(p_waves),signal(p_waves),'+r')
% plot(tm(q_peaks),signal(q_peaks),'xr')