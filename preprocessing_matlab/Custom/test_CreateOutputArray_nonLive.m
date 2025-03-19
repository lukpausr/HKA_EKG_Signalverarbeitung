% safe current directory

basePath = 'F:\Users\Mika\Documents\Studium_HKA\Semester1\ProjektarbeitEkg\Daten\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\records500';
targetPath = 'F:\Users\Mika\Documents\Studium_HKA\Semester1\ProjektarbeitEkg\Daten\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\CSVs\records500\';

initialDirectory = pwd;
cd(basePath);

failedFiles = ["failed"];


folderList = struct2cell(dir(pwd));
folderList = string(folderList(1,3:end));
% folderList = folderList(2:4); % only for tryout
for i_folder = folderList
    cd(i_folder);
    fileList = struct2cell(dir('*.dat'));
    fileList = string(fileList(1,3:end));
    % fileList = fileList(2:4); % only for tryout
    for i_file = fileList
        [~,fileName,~] = fileparts(i_file);
        if ~isfile(strcat(targetPath,fileName,'.csv'))
            % try
                [signal,FPT_MultiChannel] = ReadFileAndAnnotate(fileName);
                A = CreateOutputArray(signal,FPT_MultiChannel);
                T = array2table(A,'VariableNames',["raw_data","P-wave","P-peak","QRS-comples","R-peak","T-wave","T-peak"]);
                writetable(T,strcat(targetPath,fileName,'.csv'));
            % catch ME
            %     failedFiles = [failedFiles,fileName];
            % end
        end
    end
    cd('..\');
end

if (size(failedFiles,2) > 1)
    warning(strcat(string(size(failedFiles,2)-1), " files failed"));
    % failedFiles
end

cd(initialDirectory);