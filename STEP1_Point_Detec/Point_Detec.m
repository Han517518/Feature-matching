clearvars -except J1 J2
close all
clc

I0=J2;

u=0;
%% 前处理

figure
imshow(I0);

%选择ROI区域
BW=roipoly(I0);

I=I0;
I(BW==0)=255;

figure,imshow(I);

%中值滤波去除噪音
in_roi=I;
in_roi=medfilt2(in_roi,[2 2]); %窗口尺寸可更改
figure; imshow(in_roi);

%% 边缘识别
% 
% ima=I;
% IBW=im2bw(ima,40/255); 
% figure;
% imshow(IBW)
% 
% F1 = imfill(IBW,'holes');
% SE = ones(5); %图像被结构元素SE膨胀
% F2 = imdilate(F1,SE,'same');%膨胀操作
% figure;
% imshow(F2)
% 
% [B,L]= bwboundaries(F2,8,'noholes');
% figure;
% imshow(I0)
% hold on
% for k = 1:length(B)
%    boundary = B{k};
%    plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
% end

%% 动态阈值法
ima=255-in_roi;
t_base=170; % 初始二值化阈值-全局
c_t=2;      % 对比度阈值
ask=1;
finalima=zeros(size(ima));


imasaved=ima;

indexname=0;
while ask==1
%   indexname=indexname+1;

    ima2=im2bw(ima,t_base/255);  
    [L, num] = bwlabel(ima2);  %在ima2中找到的8分量连通域 及数量num
    warning('off')
    [meanI]=regionprops(L,ima,'MeanIntensity');
    warning('on')
    pix_pos=regionprops(L,'PixelIdxList');
    mI=cell2mat(struct2cell(meanI));
  
    diff=abs(mI-t_base);    

    pos=find(diff<c_t);   
    finalima(cat(1,pix_pos(pos).PixelIdxList)')=1;    
    pos2=find(diff>=c_t);
    
    
    
    
    if length(pos2)==0
        ask=0;
    end
    
    if length(pos2)~=0
        ima(cat(1,pix_pos(pos2).PixelIdxList)')=ima(cat(1,pix_pos(pos2).PixelIdxList)')-1;
    end
    
end

figure,imshow(finalima);
[L, num] = bwlabel(finalima);
centro_pos=regionprops(L,'centroid');
C=cat(1,centro_pos.Centroid)';
i=C(2,:);
j=C(1,:);

figure,imshow(I0);
hold on
plot(j, i, 'r+')

%% STEP5.灰度重心法求解中心


X=[];Y=[];
pix_posi=[];
for i=1:size(centro_pos,1)-1
    pix_posc=round(centro_pos(i).Centroid);
    pix_posi=[];
    count=0;
    for ii=pix_posc(1)-2:pix_posc(1)+2
        for jj=pix_posc(2)-2:pix_posc(2)+2
            count=count+1;
            pix_posi(count,:)=[ii,jj];
        end
    end
    gray=[];
    for j=1:size(pix_posi,1)
    gray_j=I0(pix_posi(j,2),pix_posi(j,1));
    gray=[gray;gray_j];
    end
    x_gray=sum(pix_posi(:,2).*double(255-gray))/sum(255-gray);
    y_gray=sum(pix_posi(:,1).*double(255-gray))/sum(255-gray);
    X=[X x_gray];Y=[Y y_gray];
end
figure,imshow(I0);
hold on
plot(Y, X, 'r+')

coor=[Y;X];

save(['H:\featurepoint\cam2\' ...
    ,num2str(u),'.mat'],"coor")

%%
load('H:\featurepoint\cam1\0.mat')
figure
plot(coor(1,:),coor(2,:),'r.')
coordi1=coor(1,:);
coordj1=coor(2,:);
load('H:\featurepoint\cam2\0.mat')
hold on
plot(coor(1,:)+255,coor(2,:),'b.')
coordi2=coor(1,:)+255;
coordj2=coor(2,:);

