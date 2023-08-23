

    load('exp_data_o126.mat');
    pixelpoint=exp_data(:,1:6);
    worldpoint=exp_data(:,7:9);

    
    net1=nettrainer([pixelpoint(:,1:2) worldpoint(:,3)],pixelpoint(:,3:4)); %训练相机间映射网络
    net2=nettrainer([pixelpoint(:,1:2) worldpoint(:,3)],pixelpoint(:,5:6)); %训练相机间映射网络
    net_test=nettrainer([pixelpoint(:,1:6)],worldpoint(:,:));
    
    
    
%%

       net_2=nettrainer_parten_data_maker(exp_data);

       save('net')
       


    
%%
%新的点配对
