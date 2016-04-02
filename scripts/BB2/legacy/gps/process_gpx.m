function process_gpx(file_name)
[lla, enu, times] = gps2enu(file_name);
fig = figure('Position', [100, 100, 800, 800]);
plot(enu(:,1), enu(:,2));
axis equal;
dlmwrite('gps_data.txt', [enu, lla, times], 'delimiter', ' ');
