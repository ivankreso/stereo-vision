function [route,times]=readgpx(fileName)
% Reads GPX XML file.
% Outputs points in metric rectangular and gps coordinates
% with corresponding datetime.

COL_X    = 1;
COL_Y    = 2;
COL_Z    = 3;
COL_LAT  = 4;
COL_LNG  = 5;

% load xml file
d = xmlread(fileName);

% check file type
if ~strcmp(d.getDocumentElement.getTagName,'gpx')
    warning('ss:formaterror','file is not in GPX format');
end


points = d.getElementsByTagName('trkpt');
route = nan(points.getLength,5);
times = nan(points.getLength,6);

for i=1:points.getLength
    point = points.item(i-1);
    route(i,COL_LAT) = str2double(point.getAttribute('lat'));
    route(i,COL_LNG) = str2double(point.getAttribute('lon'));
    
    elevation = point.getElementsByTagName('ele');
    route(i,COL_Z) = str2double(elevation.item(0).getTextContent);
    
    time = point.getElementsByTagName('time') ; 
    timChar = char(time.item(0).getTextContent) ; 
    times(i,:) = datevec([timChar(1:10) ' ' timChar(12:19)]);
end

% delta
route(:,[COL_Y,COL_X]) = route(:,[COL_LAT,COL_LNG]) - ones(points.getLength,1)*route(1,COL_LAT:COL_LNG);

% convert to meters
lat_mean=mean(route(:,COL_LAT));
KM_PER_ARCMINUTE = 1.852; 
route(:,COL_X:COL_Y) = KM_PER_ARCMINUTE*1000*60*route(:,COL_X:COL_Y); 
route(:,COL_X) = route(:,COL_X)*cos(lat_mean/180*pi); 

end