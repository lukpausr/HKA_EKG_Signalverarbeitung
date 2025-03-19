T = readtable("ptbxl_database.csv");
T1 = T.patient_id;
T2 = unique(T.patient_id);