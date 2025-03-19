% Datenbank einlesen und einzigartige IDs bestimmen
ptbxldatabase = readtable("ptbxl_database.csv");
uniquePatientIds = unique(ptbxldatabase.patient_id);
% zu jedem Patient die Anzahl der Datensätze bestimmen
% max, mean & median berechnen
DatasetsPerPatient = histcounts(ptbxldatabase.patient_id,uniquePatientIds);
[maxDatasetsPerPatient,maxPatientIdx] = max(DatasetsPerPatient);
medianDatasetsPerPatient = median(DatasetsPerPatient);
averageDatasetsPerPatient = mean(DatasetsPerPatient);
% Verteilung Anzahl der Patienten mit 1,2...10 EKGs
numPatientsByNumDatasets = [unique(DatasetsPerPatient);histc(DatasetsPerPatient, unique(DatasetsPerPatient))];
% Anzahl der Standorte
numberOfSites = size(unique(ptbxldatabase.site),1);
yearSpanDatabase = years(ptbxldatabase.recording_date(size(ptbxldatabase,1)) - ptbxldatabase.recording_date(1));
% zeige Daten der Aufzeichnungen der 10 EKGs eines Patienten
maxPatientId = uniquePatientIds(maxPatientIdx);
Dates = ptbxldatabase.recording_date(find(ptbxldatabase.patient_id==maxPatientId));