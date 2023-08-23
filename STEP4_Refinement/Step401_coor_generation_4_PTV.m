% x34=x43;y34=y43;z34=z43;
clc
clearvars -except PIC x34 y34 z34
% 相机34，其中4为参考像平面

%% 空间点滤波处理

% [xData, yData, zData] = prepareSurfaceData(x34,y34,z34);   
% ft = 'thinplateinterp';
% [fitresult, gof, output] = fit( [xData, yData], zData, ft, 'Normalize', 'on' );
% ind_r=find(abs(output.residuals(:))<5);
% x34_r=x34(find(abs(output.residuals(:))<5));
% y34_r=y34(find(abs(output.residuals(:))<5));
% z34_r=z34(find(abs(output.residuals(:))<5));
% figure
% scatter3(x34,y34,z34,'r.')

%% 像平面坐标处理

plane3=PIC(:,:,3);
plane4=PIC(:,:,4);

load('J:\SCI\calibartion_allto4.mat')
net=net_3_4;

% plan A 加在循环里  plan B 加在循环外 plan C 再加一个4到3？
point1in2=net([plane3;z34]);
distance=vecnorm(point1in2-plane4);
index_rightpoint=find(distance>2&distance<3);

figure
scatter3(x34(index_rightpoint),y34(index_rightpoint),z34(index_rightpoint),'r.')



%% 中值滤波-前处理
% plane4_r=plane4(:,index_rightpoint);
% plane3_r=plane3(:,index_rightpoint);
% % 寻找每一坐标的邻居点 相机4
% clear D4 index_d4
% for i=1:size(plane4_r,2)
%     distance4=sqrt((plane4_r(1,:)-plane4_r(1,i)).^2+(plane4_r(2,:)-plane4_r(2,i)).^2);
%     [D4(i,:),index_d4(i,:)]=sort(distance4);
% end
% 
% % 寻找每一坐标的邻居点 相机3
% clear D3 index_d3
% for i=1:size(plane3_r,2)
%     distance3=sqrt((plane3_r(1,:)-plane3_r(1,i)).^2+(plane3_r(2,:)-plane3_r(2,i)).^2);
%     [D3(i,:),index_d3(i,:)]=sort(distance3);
% end

%% 中值滤波处理

% numtofilter=10; % 进行中值滤波的临近点数量
% x34_r=x34(:,index_rightpoint);
% y34_r=y34(:,index_rightpoint);
% z34_r=z34(:,index_rightpoint);
% 
% % 相机4
% clear z_mid4
% for i=1:size(plane4_r,2)
%     z_filter=z34_r(index_d4(i,1:numtofilter+1));
%     z_mid=z_filter(1+numtofilter/2);
%     z_mid4(i)=z_mid;   
% end
% 
% % 相机3
% clear z_mid3
% for i=1:size(plane3_r,2)
%     z_filter=z34_r(index_d3(i,1:numtofilter+1));
%     z_mid=z_filter(1+numtofilter/2);
%     z_mid3(i)=z_mid;   
% end
% 
% throffilter=0.5; % 进行中值滤波的阈值（左右像平面中值的距离）
% z_abs=abs(z_mid3-z_mid4);
% x_aftfilter=x34_r(find(z_abs<throffilter));
% y_aftfilter=y34_r(find(z_abs<throffilter));
% z_aftfilter=z34_r(find(z_abs<throffilter));
% 
% 
% figure
% scatter3(x_aftfilter,y_aftfilter,z_aftfilter,'r.')

%% 在像平面插值 （u v z）
u3_aftfilter=[];
v3_aftfilter=[];
u3_aftfilter=plane3(1,index_rightpoint);
v3_aftfilter=plane3(2,index_rightpoint);
z_aftfilter=z34(index_rightpoint);


[xData, yData, zData] = prepareSurfaceData(u3_aftfilter,v3_aftfilter,z_aftfilter);
% Set up fittype and options.(二选一)
% ft = 'linearinterp';
ft = 'biharmonicinterp';
% Fit model to data.
[fitresult, gof] = fit( [xData, yData], zData, ft, 'Normalize', 'on' );

load('H:\SCI-V2.0\point2000\itr0.mat','plane3','plane4')
z3_r=fitresult(plane3(1,:),plane3(2,:));

figure
scatter3(plane3(1,:),plane3(2,:),z3_r,'r.')
%% 重新投影（在局部精确位置）

load('J:\SCI\calibartion_allto4.mat','net_3_4')
net=net_3_4;


point1in2=net([plane3;z3_r]);

coordi2=point1in2(1,:);
coordj2=point1in2(2,:);

coordi1=plane4(1,:);
coordj1=plane4(2,:);

figure
plot(coordi1,coordj1,'.')
hold on
plot(coordi2,coordj2,'*')


