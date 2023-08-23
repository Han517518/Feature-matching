clear all
clc

load('H:\SCI-V2.0\point2000\itr0.mat','plane3','plane4')
% plane3=PIC(:,:,3);
% plane4=PIC(:,:,4);


clear PIC
PIC(:,:,4) = plane4(:,src); %%某号相机
PIC(:,:,3) = plane3(:,dest); %%某号相机
  

%% 

load('H:\SCI-V2.0\calibartion_34_0824_withoutRC.mat','net')

   
worldpoint = net([PIC(:,:,3);PIC(:,:,4)]);

x = worldpoint(1,:);
y = worldpoint(2,:);
z = worldpoint(3,:);
figure
scatter3(x,y,z,'.');
x34=x;y34=y;z34=z;
% axis([0 250 20 220 -10 70])

%% 

% load('H:\SCI-V2.0\itr1_gst=50.mat','x','y','z')
% figure
% scatter3(x34,y34,z34,'.');

x34=x;y34=y;z34=z;
