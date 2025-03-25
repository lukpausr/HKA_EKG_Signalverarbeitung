% CSV-Datei einlesen
T = readtable('867_0.csv');

% Zeitreihe extrahieren
zeitreihe = T{:, 1};

% Daten extrahieren (Spalten 2 bis 7)
daten = T{:, 2:7};

% Farben für die Bereiche
farben = ['g', 'b', 'r', 'b', "#D95319", "b"];
marker = ['none', "o", 'none', "x",'none' "+"];
% Diagramm erstellen
figure;
hold on;

% Gesamte Zeitreihe blau zeichnen (als Hintergrund)
h = plot(1:length(zeitreihe), zeitreihe, 'b', 'LineWidth', 2);

% Legendenbeschriftungen und Handles erstellen
legenden_beschriftungen = {"Ausgangssignal"};
legenden_handles = [h];

% Schleife durch die Daten-Spalten
for spalte = 1:size(daten, 2)
    % Bereichs-Indices finden
    bereich_indices = find(daten(:, spalte) == 1);

    % Schleife durch die Bereiche
    for i = 1:length(bereich_indices)
        % Bereichsstart und -ende bestimmen
        start_index = bereich_indices(i);
        end_index = start_index;

        % Nachfolgende 1er-Werte finden, um den Bereich zu erweitern
        while end_index < length(zeitreihe) && daten(end_index + 1, spalte) == 1
            end_index = end_index + 1;
        end

        % Bereich farbig und dicker zeichnen (vor der blauen Linie)
        h = plot(start_index:end_index, zeitreihe(start_index:end_index), 'Color', farben(spalte), 'Marker', marker(spalte), 'LineWidth', 3);

        % Legendenbeschriftung und Handle hinzufügen (nur einmal pro Spalte)
        if isempty(find(strcmp(legenden_beschriftungen, T.Properties.VariableNames{spalte+1}), 1))
            legenden_beschriftungen{end+1} = T.Properties.VariableNames{spalte+1};
            legenden_handles(end+1) = h;
        end
    end
end

% Achsenbeschriftungen und Titel hinzufügen
xlabel('Sample');
ylabel('normierter Wert');
title('EKG (Features highlighted)');
legend([legenden_handles], [legenden_beschriftungen]);

xlim([1, 512]); % X-Achse auf den Bereich 1 bis 512 setzen
xticks('auto');