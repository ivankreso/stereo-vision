function rot_err = error_func(r_vec, gps_pts, vo_pts, nframes)

rot_err = 0.0;
rx = r_vec(1);
ry = r_vec(2);
rz = r_vec(3);

R = [[ cos(ry)*cos(rz), -cos(ry)*sin(rz), sin(ry)];
[ cos(rx)*sin(rz) + cos(rz)*sin(rx)*sin(ry), cos(rx)*cos(rz) - sin(rx)*sin(ry)*sin(rz), -cos(ry)*sin(rx)];
[ sin(rx)*sin(rz) - cos(rx)*cos(rz)*sin(ry), cos(rz)*sin(rx) + cos(rx)*sin(ry)*sin(rz), cos(rx)*cos(ry)]];

for i = 1:nframes

   v_vo = vo_pts(i+1,:).' - vo_pts(i,:).';
   v_gps = gps_pts(i+1,:).' - gps_pts(i,:).';
   vr_vo = R * v_vo;
   phi = acos(dot(v_gps, vr_vo) / norm(v_gps) / norm(vr_vo));
   rot_err = rot_err + phi^2;

end

rot_err = rot_err / nframes;
%rot_err = 1000 - rot_err;
fprintf('%f\n', rot_err);

% function error sum of first 4 scalar products <R x T_vo, T_gps>

end
