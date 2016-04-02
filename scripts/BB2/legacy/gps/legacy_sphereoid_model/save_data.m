[route, times] = readgpx('path_crop2.gpx');
fig = figure('Position', [100, 100, 800, 800]);
plot(route(:,1), route(:,2));
fig=plot(route(:,1), route(:,2));
axis equal;
dlmwrite('gps_route.txt',[route, times],'delimiter', ' ');
