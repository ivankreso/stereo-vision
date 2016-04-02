gps_pts = importdata('gps_route.txt');
viso_pts = importdata('viso_points.txt');
fig = figure('Position', [100, 100, 800, 800]);
plot(gps_pts(:,1), gps_pts(:,2));
hold on;
plot(viso_pts(:,1), viso_pts(:,3));
axis equal;
