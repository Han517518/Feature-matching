clear all;
close all;
clc;

load('E:\HanYK\SCI\calibartion_allto4.mat');
net=net_3_4;
clearvars -except net

m=1;
load(['E:\HanYK\coor_in_image\point1000\C4 (', num2str(m) ').mat'])
%ȡ�ο�����ĵõ������
coordi1=COOR(1,:);
coordj1=COOR(2,:);
%���崰�ڳߴ磨��PTV�㷨����ʵ�����壩
height=600;
width=800;

load(['E:\HanYK\coor_in_image\point1000\C3 (', num2str(m) ').mat'])


for k=1:7 %��13��ͼ��ӳ�� ���5mm k=7
    n=15+5*k;
%     n=42;
    clear point1in2
    for i=1:size(COOR,2)
        point1in2(:,i)=net([COOR(:,i);n]);
        coordi2=point1in2(1,:);
        coordj2=point1in2(2,:);
        filename=strcat('pic34_',num2str(n));
        save(['E:\HanYK\SCI\POINT1000\IMAGE', num2str(m) '\BEF\PIC34\',filename],'coordi1','coordj1','coordi2','coordj2','height','width');
    end
end



figure
plot(coordi1,coordj1,'.')
hold on
plot(coordi2,coordj2,'*')