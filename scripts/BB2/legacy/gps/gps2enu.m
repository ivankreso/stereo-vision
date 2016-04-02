function [lla, enu, times] = gps2enu(fileName)
% Reads GPX XML file.
% Outputs points in metric rectangular and gps coordinates
% with corresponding datetime.

COL_X    = 1;
COL_Y    = 2;
COL_Z    = 3;

COL_LAT  = 1;
COL_LNG  = 2;
COL_ALT  = 3;


% load xml file
d = xmlread(fileName);

% check file type
if ~strcmp(d.getDocumentElement.getTagName,'gpx')
    warning('ss:formaterror','file is not in GPX format');
end

points = d.getElementsByTagName('trkpt');
lla = nan(points.getLength,3);
enu = nan(points.getLength,3);
times = nan(points.getLength,6);

for i=1:points.getLength
    point = points.item(i-1);
    lla(i,COL_LAT) = str2double(point.getAttribute('lat'));
    lla(i,COL_LNG) = str2double(point.getAttribute('lon'));
    elevation = point.getElementsByTagName('ele');
    lla(i,COL_ALT) = str2double(elevation.item(0).getTextContent);
    
    time = point.getElementsByTagName('time') ; 
    timChar = char(time.item(0).getTextContent) ; 
    times(i,:) = datevec([timChar(1:10) ' ' timChar(12:19)]);
end

for i = 1:size(lla,1)
   [enu(i,1), enu(i,2), enu(i,3)] = geodetic2enu(lla(i,1), lla(i,2), lla(i,3), lla(1,1), lla(1,2), lla(1,3), referenceEllipsoid('WGS84'));
end
