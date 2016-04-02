
gps_pts = importdata('gps_pts.txt');
vo_pts = importdata('vo_inter_pts.txt');

%r_euler = eye(3);
r_euler = [0, 0, 0];
r_euler = lsqnonlin(@(r_euler)error_func(r_euler, gps_pts, vo_pts, 4), r_euler);

rx = r_euler(1);
ry = r_euler(2);
rz = r_euler(3);

r_euler
R = [[ cos(ry)*cos(rz), -cos(ry)*sin(rz), sin(ry)];
[ cos(rx)*sin(rz) + cos(rz)*sin(rx)*sin(ry), cos(rx)*cos(rz) - sin(rx)*sin(ry)*sin(rz), -cos(ry)*sin(rx)];
[ sin(rx)*sin(rz) - cos(rx)*cos(rz)*sin(ry), cos(rz)*sin(rx) + cos(rx)*sin(ry)*sin(rz), cos(rx)*cos(ry)]]
