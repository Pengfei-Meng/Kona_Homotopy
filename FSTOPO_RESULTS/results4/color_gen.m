% generating colorbar for Structural Problem
clear all

clc
thickness = importdata('thickness');
stress = importdata('stress');

ne = length(thickness);   % number of elements
ny = sqrt(length(thickness)/2);
nx = 2*ny; 

% generate nx * ny * 3 RGB map
R = zeros(1,ne);
G = zeros(1,ne);
B = zeros(1,ne);

tmin = min(thickness) - 1e-2;
tmax = max(thickness) + 1e-2;

rgbbreak = [0.0, 0.05, 0.1, 0.325, 0.55, 0.755, 1.0];
rgbvals = [69, 145, 224, 255, 254, 252, 215;
     117, 191, 243, 255, 224, 141, 48;
     180, 219, 248, 191, 144, 89, 39]/255;
%  rgbvals = fliplr(rgbvals);
 
 for i = 1:ne
    tval = (thickness(i) - tmin)/(tmax - tmin);
    for j = 1:6
       if tval >= rgbbreak(j) && tval <= rgbbreak(j+1) 
            u = (tval - rgbbreak(j))/(rgbbreak(j+1) - rgbbreak(j));
%             R(i) = (1.0 - u)*rgbvals(1,j) + u*rgbvals(1,j+1);
%             G(i) = (1.0 - u)*rgbvals(2,j) + u*rgbvals(2,j+1);
%             B(i) = (1.0 - u)*rgbvals(3,j) + u*rgbvals(3,j+1);
            R(i) = u*rgbvals(1,j) + (1-u)*rgbvals(1,j+1);
            G(i) = u*rgbvals(2,j) + (1-u)*rgbvals(2,j+1);
            B(i) = u*rgbvals(3,j) + (1-u)*rgbvals(3,j+1);

       end
    end
    
 end
 
RGB_vals = [R; G; B];

% RGB_vals = fliplr(RGB_vals);
 
thickness_mat = reshape(thickness, [nx, ny]);
stress_mat = reshape(stress, [nx, ny]);
Red = reshape(R, [nx, ny]);
Gre = reshape(G, [nx, ny]);
Blu = reshape(B, [nx, ny]);

% thick_mat = flipud(thickness_mat');
% red_mat = flipud(Red');
% gre_mat = flipud(Gre');
% blu_mat = flipud(Blu');

% RGB = zeros(ny, nx, 3);
% RGB(:,:,1) = red_mat;
% RGB(:,:,2) = gre_mat;
% RGB(:,:,3) = blu_mat;
% RGB = [reshape(red_mat, [ne,1]), reshape(gre_mat,[ne,1]), reshape(blu_mat, [ne,1])]; 

figure()
imagesc(flipud(thickness_mat'))
set(gca, 'Visible', 'off')
% imagesc(thickness_mat)
axis equal
% colormap(flipud(rgbvals'))
c = gray(256);
colormap(c(1:180,:))
% colormap(flipud(RGB_vals'))
cb = colorbar('FontSize', 12, 'TickLength',0.0,...
    'Location', 'eastoutside' );
set(cb, 'position', [0.95 0.3 0.03 0.45])

% 
% figure
% surf(peaks)
% colormap(map)
% 
% colorbar('Ticks', rgbbreak,...
%          'TickLabels',thks)
     
     



