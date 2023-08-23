clearvars -except x34 y34 z34
close all
clc

% load('worldpoint.mat') % 神经网络重构出的三维点云坐标 worldpoint
load('H:\SCI-V2.0\point600\RT-POINT600.mat') % 坐标变换矩阵 已给出

z34=-z34+35;

WP=[x34;y34;z34];
%调平处理
% WP=T+R*WP;
% T=roty(1)*T;
% R=roty(1)*R;
% x1=WP(1,1:end);
% y1=WP(2,1:end);
% z1=WP(3,1:end);
% figure
% scatter3(x1,y1,z1,'.');

% Step1 筛除坏点
I=[]; 
L=[];
DD=[];
LL=[];
wp000=WP;
p2d=[];
z_real=[];z_test=[];
indempty=[];
T_Z=[];


% % 基于最大值与最小值筛除坏点
% for i=1:size(wp000,2)
%     if wp000(3,i)>100||wp000(2,i)>200
%         p2d=[p2d i];
%     end
% end
% wp000(:,p2d)=[];
x=wp000(1,1:end);
y=wp000(2,1:end);
z=wp000(3,1:end);
% % figure
% % scatter3(x,y,z,'.');
% wp111=wp000;

for i=1:size(wp000,2)
    fprintf('正在计算第 %g 个点呢...\n',i);
    x0=x(i);y0=y(i);z0=z(i);
    I=[I i];%点的索引
    % 所有点处z的测量值
    z_test=[z_test z0];
    % 求解t的值
    syms t
    f1=-0.006.*t.*t-0.03.*t+246;
    ff1=(f1-T(1)-R(1,1).*x0-R(1,2).*y0)./R(1,3);
    f2=0.18.*t.*sin(pi.*t/25);
    ff2=(f2-T(3)-R(3,1).*x0-R(3,2).*y0)./R(3,3);
    % 每一点处t的值
    l=double(vpasolve(ff1==ff2,t));
    % 寻找l无解处t的索引
    if(isempty(l))
        indempty=[indempty i];
    end
    % 所有点处t的值
    LL=[LL l];
    % 每一点处z的真实值
    z_reali=(0.18.*l.*sin(pi.*l/25)-T(3)-R(3,1).*x0-R(3,2).*y0)./R(3,3);
    % 所有点处z的真实值
    z_real=[z_real z_reali];

end

%去掉少数无解的点
I(:,indempty)=[];
z_test(:,indempty)=[]; 

%所有点的索引及对应的t和z的真实值
T_Z=[I;LL;z_test;z_real];


%所有点z测量值和真实值之间的差
delta=abs(T_Z(3,:)-T_Z(4,:));
ave=mean(delta);%误差的均值
u=std(delta);%误差的均方根值

x_test=x;y_test=y;x_test(:,indempty)=[];y_test(:,indempty)=[];
figure,scatter3(x_test,y_test,z_test,'.');
hold on
scatter3(x_test,y_test,z_real,'.');

% find(abs(output.residuals)>=mean(abs(output.residuals))*2.5);
% xwp_r=xwp1346;xwp_r(ans)=[];
% ywp_r=ywp1346;ywp_r(ans)=[];
% zwp_r=zwp1346;zwp_r(ans)=[];
% figure
% scatter3(xwp_r,ywp_r,zwp_r,'.');

num_E=find(delta>=3);  % 
outlier_ratio=size(num_E,2)/size(WP,2);
% productivity=size(WP,2)/1063;
% confidence=(size(WP,2)-64)/size(WP,2);
% totalaccurancy=productivity*confidence;
% delta_r=delta;delta_r(:,num_E)=[];
% positionaccurancy=sqrt(mean(delta_r.*delta_r))/68;
