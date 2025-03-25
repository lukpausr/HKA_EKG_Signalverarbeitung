% CSV-Datei einlesen
T = readtable('867_0.csv');

% Daten extrahieren (letzte 6 Spalten)
daten = T{:, end-5:end};

% Farben für die Plots
farben = ['g', 'g', 'r', 'r', "#D95319", "#D95319"];
marker = ['none', "o", 'none', "x",'none' "+"];

% Diagramm erstellen
figure;

% Schleife durch die Spalten
for spalte = 1:size(daten, 2)
    % Unterplot erstellen
    subplot(6, 1, spalte);

    % Daten plotten
    plot(daten(:, spalte), 'Color', farben(spalte), 'LineWidth', 2);

    % Marker hinzufügen, wenn Spalte ungerade ist
    if mod(spalte, 2) == 0
        % Indices der Einsen finden
        eins_indices = find(daten(:, spalte) == 1);

        % Marker für die Einsen hinzufügen
        hold on; % Hold on, um Marker zum Plot hinzuzufügen
        plot(eins_indices, daten(eins_indices, spalte), 'Marker', marker(spalte), 'LineStyle', 'none', 'Color', farben(spalte), 'LineWidth', 2);
        hold off; % Hold off wieder deaktivieren
    end

    % Achsenbeschriftungen und Titel hinzufügen
    ylabel([T.Properties.VariableNames{spalte+1}]);
    if spalte == 1
        title('Segmentierung');
    end
    xlim([1, 512]);
end

% X-Achsenbeschriftung für den letzten Plot hinzufügen
xlabel('Sample');