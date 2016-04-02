% demonstrates mono visual odometry on an image sequence
disp('===========================');
clear all; close all; dbstop error;

% parameter settings (for an example, please download
% sequence '2010_03_09_drive_0019' from www.cvlibs.net)
%img_dir      = '/home/geiger/5_Data/karlsruhe_dataset/2011_stereo/2010_03_09_drive_0019';
%img_dir     = 'C:\Users\geiger\Desktop\2010_03_09_drive_0019';
%img_dir		= '/home/kreso/projects/master_thesis/src/stereo-master/StereoVision_build/data/libviso/2010_03_09_drive_0019';
%param.f      = 645.2;
%param.cu     = 635.9;
%param.cv     = 194.1;
%param.height = 1.6;
%param.pitch  = -0.08;
%first_frame  = 0;
%last_frame   = 372;


img_dir	 = '/home/kreso/projects/master_thesis/src/stereo-master/StereoVision_build/data/cropped_full_fixed';
param.f  = 530;
paramcu = 325.419;
param.cv = 240.353;
param.height = 1.6;
param.pitch  = -0.08;
first_frame  = 0;
last_frame   = 2917;

% init visual odometry
visualOdometryMonoMex('init',param);

% init transformation matrix array
Tr_total{1} = eye(4);

% create figure
figure('Color',[1 1 1]);
ha1 = axes('Position',[0.05,0.7,0.9,0.25]);
axis off;
ha2 = axes('Position',[0.05,0.05,0.9,0.6]);
set(gca,'XTick',-500:10:500);
set(gca,'YTick',-500:10:500);
axis equal, grid on, hold on;

% for all frames do
replace = 0;
step = 3;
for frame=first_frame:step:last_frame
  
  % 1-based index
  k = (frame/step)-first_frame+1;
  
  % read current images
  %I = imread([img_dir '/I1_' num2str(frame,'%06d') '.png']);
  I = imread([img_dir '/fc2_save_2012-10-31-161114-' num2str(frame,'%04d') '_left.png']);

  % compute egomotion
  Tr = visualOdometryMonoMex('process',I,replace);
  
  % accumulate egomotion, starting with second frame
  if k>1
    
    % if motion estimate failed: set replace "current frame" to "yes"
    % this will cause the "last frame" in the ring buffer unchanged
    if isempty(Tr)
      replace = 1;
      Tr_total{k} = Tr_total{k-1};
      
    % on success: update total motion (=pose)
    else
      replace = 0;
      Tr_total{k} = Tr_total{k-1}*inv(Tr);
    end
  end

  % update image
  axes(ha1); cla;
  imagesc(I); colormap(gray);
  axis off;
  
  % update trajectory
  axes(ha2);
  if k>1
    plot([Tr_total{k-1}(1,4) Tr_total{k}(1,4)], ...
         [Tr_total{k-1}(3,4) Tr_total{k}(3,4)],'-xb','LineWidth',1);
  end
  pause(0.05); refresh;

  % output statistics
  num_matches = visualOdometryMonoMex('num_matches');
  num_inliers = visualOdometryMonoMex('num_inliers');
  disp(['Frame: ' num2str(frame) ...
        ', Matches: ' num2str(num_matches) ...
        ', Inliers: ' num2str(100*num_inliers/num_matches,'%.1f') ,' %']);
end

% release visual odometry
visualOdometryMonoMex('close');
