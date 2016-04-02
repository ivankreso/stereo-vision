function [lla, ecef, times]=lla_to_ecef(fileName)
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
ecef = nan(points.getLength,3);
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


%ecef = lla2ecef(lla);

% WGS84 ellipsoid constants:
a = 6378137;
e = 8.1819190842622e-2;

% intermediate calculation
% (prime vertical radius of curvature)
N = a ./ sqrt(1 - e^2 .* sin(lla(:,COL_LAT)).^2);

% results:
ecef(:,COL_X) = (N + lla(:,COL_ALT)) .* cos(lla(:,COL_LAT)) .* cos(lla(:,COL_LNG));
ecef(:,COL_Y) = (N + lla(:,COL_ALT)) .* cos(lla(:,COL_LAT)) .* sin(lla(:,COL_LNG));
ecef(:,COL_Z) = ((1-e^2) .* N + lla(:,COL_ALT)) .* sin(lla(:,COL_LAT));

% BUGED
%delta_t = ecef(1,:)
%for i = 1:size(ecef,1)
%   ecef(i,:) = ecef(i,:) - delta_t;
%end


%a = 6378137.0;
%e2 = 6.69437999014 * 10^−3;

%N = a / sqrt(1 - (e2 * sin(route(:,COL_LAT))^2))
%X = N + route(:,COL_ALT) * cos() * cos()

% delta
%route(:,[COL_Y,COL_X]) = route(:,[COL_LAT,COL_LNG]) - ones(points.getLength,1)*route(1,COL_LAT:COL_LNG);

% convert to meters
%lat_mean=mean(route(:,COL_LAT));
%KM_PER_ARCMINUTE = 1.852; 
%route(:,COL_X:COL_Y) = KM_PER_ARCMINUTE*1000*60*route(:,COL_X:COL_Y); 
%route(:,COL_X) = route(:,COL_X)*cos(lat_mean/180*pi); 

end
