function enu = ecef2enu(ecef,ecef_ref)
%----------------------------------------------------------------------
%               function enu = ecef2enu(ecef,ecef_ref)
%
%   converts a position vector given in ECEF coordinates to a vector in
%   North, East Down coordinates centered at the coordinates given
%   by ecef_ref.
%
%   Demoz Gebre 12/31/98
%---------------------------------------------------------------------
lla_ref = ecef2lla(ecef_ref);
lat = lla_ref(1)*pi/180;
lon = lla_ref(2)*pi/180;
lla_ref(3) = 0;
%decef = ecef-lla2ecef(lla_ref);
decef = ecef;
enu(3,1)= cos(lat)*cos(lon)*decef(1)+cos(lat)*sin(lon)*decef(2)+sin(lat)*decef(3);
enu(1,1)=-sin(lon)*decef(1) + cos(lon)*decef(2);
enu(2,1)=-sin(lat)*cos(lon)*decef(1)-sin(lat)*sin(lon)*decef(2)+cos(lat)*decef(3);
