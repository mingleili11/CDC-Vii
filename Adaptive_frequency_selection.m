clc
clear
close all
file_path =  'E:\windows\ѧ��\�㱨\33 ����\��Է���һ�ĳ���\FEM\PHM 2012\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\Full_Test_Set\Bearing1_3\';% �ļ���·��
% ȫ�������ź�
csv_acc_path_list = dir(strcat(file_path,'acc*.csv'));%��ȡ���ļ���������csv��ʽ���ļ�
csv_acc_num = length(csv_acc_path_list);%��ȡ�ļ�������
if csv_acc_num > 0 %�������������ļ�
        for j = 1:csv_acc_num %��һ��ȡ�ļ�
            csv_acc_name = csv_acc_path_list(j).name;% �ļ���
            csv_acc =  csvread(strcat(file_path,csv_acc_name));
            csv_acc_data(:,:,j)=csv_acc;
            fprintf('%d %d %s\n',csv_acc_num,j,strcat(file_path,csv_acc_name));% ��ʾ���ڴ�����ļ���
        end
end
% %���full_test_set��bearing1_4
% file_path =  'E:\windows\ѧ��\�㱨\33 ����\��Է���һ�ĳ���\FEM\PHM 2012\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\Full_Test_Set\Bearing1_4\';% �ļ���·��
% %% ȫ�������ź�
% csv_acc_path_list = dir(strcat(file_path,'acc*.csv'));%��ȡ���ļ���������csv��ʽ���ļ�
% csv_acc_num = length(csv_acc_path_list);%��ȡ�ļ�������
% if csv_acc_num > 0 %�������������ļ�
%         for j = 1:csv_acc_num %��һ��ȡ�ļ�
%             csv_acc_name = csv_acc_path_list(j).name;% �ļ���
%             csv_acc=dlmread(strcat(file_path,csv_acc_name), ';');
%             csv_acc_data(:,:,j)=csv_acc;
%             fprintf('%d %d %s\n',csv_acc_num,j,strcat(file_path,csv_acc_name));% ��ʾ���ڴ�����ļ���
%         end
% end

% �ϲ����� ʱ��*ͨ��
channel=6;   %�źŵ�ͨ����
csv_acc_data_change=permute(csv_acc_data,[2 1 3]);
csv_acc_data=reshape(csv_acc_data_change,channel,prod(size(csv_acc_data))/channel)';

% % %% ȫ�����¶��ź�
% % csv_temp_path_list = dir(strcat(file_path,'temp*.csv'));%��ȡ���ļ���������csv��ʽ���ļ�
% % csv_temp_num = length(csv_temp_path_list);%��ȡ�ļ�������
% % delimiter = ',';
% % formatSpec = '%s%s%s%s%s%s%[^\n\r]';
% % if csv_temp_num > 0 %�������������ļ�
% %         for j = 1:csv_temp_num %��һ��ȡ�ļ�
% %             csv_temp_name = csv_temp_path_list(j).name;% �ļ���
% %             csv_temp_fileID = fopen(strcat(file_path,csv_temp_name),'r');
% %             csv_temp = textscan(csv_temp_fileID, formatSpec, 'Delimiter', delimiter);
% %             for i=1:size(csv_temp{1,1},1)
% %                 csv_temp_data(i,:,j)=str2num(csv_temp{1,1}{i,1})';
% %             end
% %             fprintf('%d %d %s\n',csv_temp_num,j,strcat(file_path,csv_temp_name));% ��ʾ���ڴ�����ļ���
% %             fclose(csv_temp_fileID);
% %         end
% % end
% % 
% % �ϲ����� ʱ��*ͨ��
% % channel=5;   %�źŵ�ͨ����
% % csv_temp_data_change=permute(csv_temp_data,[2 1 3]);
% % csv_temp_data=reshape(csv_temp_data_change,channel,prod(size(csv_temp_data))/channel)';
% 
%% ȫ�������źź��¶��źŵ�ʱ��ͼ
clearvars -except csv_acc_data csv_temp_data csv_acc_num
figure;subplot 211;plot(csv_acc_data(:,5));title('ˮƽ���ź�');
       subplot 212;plot(csv_acc_data(:,6));title('��ֱ���ź�');
% % %figure;plot(csv_temp_data(:,5));title('�¶��ź�')

a1=csv_acc_data(:,5);
a2=csv_acc_data(:,6);
save ('E:\windows\ѧ��\�㱨\33 ����\��Է���һ�ĳ���\FEM\PHM 2012\a1.mat')
save ('E:\windows\ѧ��\�㱨\33 ����\��Է���һ�ĳ���\FEM\PHM 2012\a2.mat')

load ('E:\windows\ѧ��\�㱨\33 ����\��Է���һ�ĳ���\FEM\PHM 2012\a1.mat')
load ('E:\windows\ѧ��\�㱨\33 ����\��Է���һ�ĳ���\FEM\PHM 2012\a2.mat')
x_input1=a1;
x_input2=a2;

plot(x_input1);title('�����ź�ʱ��ͼ��')   %���������ź�ʱ��ͼ��
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
plot(t1,x_cobination);title('���ھ��������ź�ʱ��ͼ��')   %���������ź�ʱ��ͼ��

N=length(x_cobination); %���������
signalFFT=abs(fft(x_cobination,N));%��ʵ�ķ�ֵ
Y=2*signalFFT/N;
f=(0:N/2)*(fs/N);
figure;plot(f,Y(1:N/2+1));
ylabel('amp'); xlabel('frequency');title('�����źŵ�Ƶ��');grid on
D_layers=5;
wpt=wpdec(x_cobination,D_layers,'dmey');        %����D_layers��С�����ֽ�
plot(wpt);                          %����С������
A = (1:2^D_layers);
for i=1:D_layers-1
    A = A+2^(i);
end
A=A';
nodes=[A];   %��3��Ľڵ��
ord=wpfrqord(nodes);  %С����ϵ�����ţ�ord�����ź�С����ϵ���������ɵľ�����3��ֽ��[1;2;4;3;7;8;6;5]
nodes_ord=nodes(ord); %���ź��С��ϵ��
for i=1:2^D_layers
rex3(:,i)=wprcoef(wpt,nodes_ord(i));  %ʵ�ֶԽڵ�С���ڵ�����ع�        
end
 
figure;                         %���Ƶ�3������ڵ�ֱ��ع����źŵ�Ƶ��
for i=0:2^D_layers-1
subplot(8,4,i+1);
x_sign= rex3(:,i+1); 
N=length(x_sign); %���������
signalFFT=abs(fft(x_sign,N));%��ʵ�ķ�ֵ
Y=2*signalFFT/N;
f=(0:N/2)*(fs/N);
plot(f,Y(1:N/2+1));
ylabel('amp'); xlabel('frequency');grid on
axis([0 12000 0 20e-3]); title(['С������3��',num2str(i),'�ڵ��ź�Ƶ��']);
end

%% wavelet packet coefficients. ��ȡС�����ֽ�ĸ����ڵ��С����ϵ��
for i=1:2^D_layers
cfs{i}=wpcoef(wpt,nodes_ord(i));  %����������3��i�ڵ��С����ϵ��[(fs/2)*(i-1)-(fs/2)*i]Hz
E_cfs(i)=norm(cfs{i},2)^2;  %% 1-����������norm(...,1)������Ԫ�ؾ���ֵ֮�ͣ�2-����������norm(...,2)������Ԫ��ƽ���Ϳ����ţ�
end
E_total=sum(E_cfs);
for i=1:2^D_layers
  p_node(i)= 100*E_cfs(i)/E_total;           % ���ÿ���ڵ��ռ��
end
[value,location]=sort(p_node,'descend');

pick_p_node=location(1:6);
for i=1:6
pick_frequency_band(i,:)=[(pick_p_node(i)-0.5)*(fs/2/2^D_layers) (pick_p_node(i)+0.5)*(fs/2/2^D_layers)];
end
figure;
x=1:2^D_layers;
bar(x,p_node);
title('����Ƶ��������ռ�ı���');
xlabel('Ƶ�� Hz');
ylabel('�����ٷֱ�/%');
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







    
    
    
    








