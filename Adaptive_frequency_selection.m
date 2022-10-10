clc
clear
close all
file_path =  'E:\windows\学术\汇报\33 开题\针对方案一的程序\FEM\PHM 2012\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\Full_Test_Set\Bearing1_3\';% 文件夹路径
% 全寿命振动信号
csv_acc_path_list = dir(strcat(file_path,'acc*.csv'));%获取该文件夹中所有csv格式的文件
csv_acc_num = length(csv_acc_path_list);%获取文件总数量
if csv_acc_num > 0 %有满足条件的文件
        for j = 1:csv_acc_num %逐一读取文件
            csv_acc_name = csv_acc_path_list(j).name;% 文件名
            csv_acc =  csvread(strcat(file_path,csv_acc_name));
            csv_acc_data(:,:,j)=csv_acc;
            fprintf('%d %d %s\n',csv_acc_num,j,strcat(file_path,csv_acc_name));% 显示正在处理的文件名
        end
end
% %针对full_test_set的bearing1_4
% file_path =  'E:\windows\学术\汇报\33 开题\针对方案一的程序\FEM\PHM 2012\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\Full_Test_Set\Bearing1_4\';% 文件夹路径
% %% 全寿命振动信号
% csv_acc_path_list = dir(strcat(file_path,'acc*.csv'));%获取该文件夹中所有csv格式的文件
% csv_acc_num = length(csv_acc_path_list);%获取文件总数量
% if csv_acc_num > 0 %有满足条件的文件
%         for j = 1:csv_acc_num %逐一读取文件
%             csv_acc_name = csv_acc_path_list(j).name;% 文件名
%             csv_acc=dlmread(strcat(file_path,csv_acc_name), ';');
%             csv_acc_data(:,:,j)=csv_acc;
%             fprintf('%d %d %s\n',csv_acc_num,j,strcat(file_path,csv_acc_name));% 显示正在处理的文件名
%         end
% end

% 合并矩阵 时间*通道
channel=6;   %信号的通道数
csv_acc_data_change=permute(csv_acc_data,[2 1 3]);
csv_acc_data=reshape(csv_acc_data_change,channel,prod(size(csv_acc_data))/channel)';

% % %% 全寿命温度信号
% % csv_temp_path_list = dir(strcat(file_path,'temp*.csv'));%获取该文件夹中所有csv格式的文件
% % csv_temp_num = length(csv_temp_path_list);%获取文件总数量
% % delimiter = ',';
% % formatSpec = '%s%s%s%s%s%s%[^\n\r]';
% % if csv_temp_num > 0 %有满足条件的文件
% %         for j = 1:csv_temp_num %逐一读取文件
% %             csv_temp_name = csv_temp_path_list(j).name;% 文件名
% %             csv_temp_fileID = fopen(strcat(file_path,csv_temp_name),'r');
% %             csv_temp = textscan(csv_temp_fileID, formatSpec, 'Delimiter', delimiter);
% %             for i=1:size(csv_temp{1,1},1)
% %                 csv_temp_data(i,:,j)=str2num(csv_temp{1,1}{i,1})';
% %             end
% %             fprintf('%d %d %s\n',csv_temp_num,j,strcat(file_path,csv_temp_name));% 显示正在处理的文件名
% %             fclose(csv_temp_fileID);
% %         end
% % end
% % 
% % 合并矩阵 时间*通道
% % channel=5;   %信号的通道数
% % csv_temp_data_change=permute(csv_temp_data,[2 1 3]);
% % csv_temp_data=reshape(csv_temp_data_change,channel,prod(size(csv_temp_data))/channel)';
% 
%% 全寿命振动信号和温度信号的时域图
clearvars -except csv_acc_data csv_temp_data csv_acc_num
figure;subplot 211;plot(csv_acc_data(:,5));title('水平振动信号');
       subplot 212;plot(csv_acc_data(:,6));title('竖直振动信号');
% % %figure;plot(csv_temp_data(:,5));title('温度信号')

a1=csv_acc_data(:,5);
a2=csv_acc_data(:,6);
save ('E:\windows\学术\汇报\33 开题\针对方案一的程序\FEM\PHM 2012\a1.mat')
save ('E:\windows\学术\汇报\33 开题\针对方案一的程序\FEM\PHM 2012\a2.mat')

load ('E:\windows\学术\汇报\33 开题\针对方案一的程序\FEM\PHM 2012\a1.mat')
load ('E:\windows\学术\汇报\33 开题\针对方案一的程序\FEM\PHM 2012\a2.mat')
x_input1=a1;
x_input2=a2;

plot(x_input1);title('输入信号时域图像')   %绘制输入信号时域图像
x=x_input1;       
fs=25.6e3;

t1=0:1/fs:0.1*size(a1,1)/2560;
t1=t1(1:end-1);
x_noise1=2*sin(2*pi*150.*t1);
x_noise2=10*sin(2*pi*50.*t1);
x_noise3=2*sin(2*pi*100.*t1);
x_noise4=4*sin(2*pi*20.*t1);
x_noise5=6*sin(2*pi*180.*t1);
x_noise6=3*sin(2*pi*75.*t1);
x_noise7=5*sin(2*pi*108.*t1);
x_noise8=7*sin(2*pi*159.*t1);
x_noise=x_noise1+x_noise2+x_noise3+x_noise4+x_noise5+x_noise6+x_noise7+x_noise8;
x_cobination=x_noise'+x_input1;
plot(t1,x_cobination);title('基于井下输入信号时域图像')   %绘制输入信号时域图像

N=length(x_cobination); %采样点个数
signalFFT=abs(fft(x_cobination,N));%真实的幅值
Y=2*signalFFT/N;
f=(0:N/2)*(fs/N);
figure;plot(f,Y(1:N/2+1));
ylabel('amp'); xlabel('frequency');title('输入信号的频谱');grid on
D_layers=5;
wpt=wpdec(x_cobination,D_layers,'dmey');        %进行D_layers层小波包分解
plot(wpt);                          %绘制小波包树
A = (1:2^D_layers);
for i=1:D_layers-1
    A = A+2^(i);
end
A=A';
nodes=[A];   %第3层的节点号
ord=wpfrqord(nodes);  %小波包系数重排，ord是重排后小波包系数索引构成的矩阵　如3层分解的[1;2;4;3;7;8;6;5]
nodes_ord=nodes(ord); %重排后的小波系数
for i=1:2^D_layers
rex3(:,i)=wprcoef(wpt,nodes_ord(i));  %实现对节点小波节点进行重构        
end
 
figure;                         %绘制第3层各个节点分别重构后信号的频谱
for i=0:2^D_layers-1
subplot(8,4,i+1);
x_sign= rex3(:,i+1); 
N=length(x_sign); %采样点个数
signalFFT=abs(fft(x_sign,N));%真实的幅值
Y=2*signalFFT/N;
f=(0:N/2)*(fs/N);
plot(f,Y(1:N/2+1));
ylabel('amp'); xlabel('frequency');grid on
axis([0 12000 0 20e-3]); title(['小波包第3层',num2str(i),'节点信号频谱']);
end

%% wavelet packet coefficients. 求取小波包分解的各个节点的小波包系数
for i=1:2^D_layers
cfs{i}=wpcoef(wpt,nodes_ord(i));  %对重排序后第3层i节点的小波包系数[(fs/2)*(i-1)-(fs/2)*i]Hz
E_cfs(i)=norm(cfs{i},2)^2;  %% 1-范数：就是norm(...,1)，即各元素绝对值之和；2-范数：就是norm(...,2)，即各元素平方和开根号；
end
E_total=sum(E_cfs);
for i=1:2^D_layers
  p_node(i)= 100*E_cfs(i)/E_total;           % 求得每个节点的占比
end
[value,location]=sort(p_node,'descend');

pick_p_node=location(1:6);
for i=1:6
pick_frequency_band(i,:)=[(pick_p_node(i)-0.5)*(fs/2/2^D_layers) (pick_p_node(i)+0.5)*(fs/2/2^D_layers)];
end
figure;
x=1:2^D_layers;
bar(x,p_node);
title('各个频段能量所占的比例');
xlabel('频率 Hz');
ylabel('能量百分比/%');
for j=1:2^D_layers
text(x(j),p_node(j),num2str(p_node(j),'%0.2f'),...
    'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
end

[p,f,t] = pspectrum(x_cobination,fs,'spectrogram');

for i=1:6
    a=find(f>=pick_frequency_band(i,1)&f<=pick_frequency_band(i,2));
    b=[a(1) a(end)];
    pick_p{i}=p(b(1):b(end),:);
    pick_p{i}=pick_p{i}';
    figure;
    waterfall(f(b(1):b(end)),t,pick_p{i})
    xlabel('Frequency (Hz)')
    ylabel('Time (seconds)')
    wtf = gca;
    wtf.XDir = 'reverse';
    view([30 45])
end







    
    
    
    








