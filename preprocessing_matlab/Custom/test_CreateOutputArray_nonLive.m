% safe current directory

basePath = 'C:\Users\lukas\Documents\HKA_DEV\HKA_EKG_Signalverarbeitung_Data\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\';
originPath = [basePath 'records500\'];
targetPath = [basePath 'preprocessing_output\'];

disp(originPath)
disp(targetPath)

initialDirectory = pwd;
cd(originPath);

failedFiles = ["failed"];


folderList = struct2cell(dir(originPath));
folderList = string(folderList(1,3:end));

%folderList = string(folderList(2:4)) % only for tryout

for i_folder = folderList
    % Check if the folder name is numeric and greater than or equal to 12000
    if str2double(i_folder) >= 12000
        disp(i_folder);
        cd(i_folder);
        fileList = struct2cell(dir('*.dat'));

        fileList = string(fileList(1,3:end));

        for i_file = fileList
            [~,fileName,~] = fileparts(i_file);

            if ~isfile(strcat(targetPath, fileName, '.csv'))
                try
                    [signal, FPT_MultiChannel] = ReadFileAndAnnotate(fileName);
                    A = CreateOutputArray(signal, FPT_MultiChannel);
            
                    T = array2table(A, 'VariableNames', ...
                        [ ...
                        "I", ...
                        "II", ...
                        "III", ...
                        "AVR", ...
                        "AVL", ...
                        "AVF", ...
                        "V1", ...
                        "V2", ...
                        "V3", ...
                        "V4", ...
                        "V5", ...
                        "V6", ...
                        "P-wave", ...
                        "P-peak", ...
                        "QRS-complex", ...
                        "R-peak", ...
                        "T-wave", ...
                        "T-peak" ...
                        ]);
            
                    writetable(T, strcat(targetPath, fileName, '.csv'));
                catch ME
                    % Optionally log the error message or handle it as needed
                    disp(['Error processing file: ', fileName, ' - ', ME.message]);
                    % Continue to the next file without interruption
                end

            end

        end
        
        cd('..\');
    end
end


if (size(failedFiles,2) > 1)
    warning(strcat(string(size(failedFiles,2)-1), " files failed"));
    % failedFiles
end

cd(initialDirectory);