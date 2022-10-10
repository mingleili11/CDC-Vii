clear;clc;close all
%%
fs = 25.6e3;
g = 9.81;
map = jet(255);%figure中使用的颜色矩阵就是被扩充到256色的jet颜色，而不是原来64色的jet颜色。

%% Extract train features
path = fullfile('..', 'Dataset', 'Train_set');
folders = dir(path);
folders = folders(3:end); % 所有文件夹

for folderIdx = 4:length(folders)
    curfolder = folders(folderIdx).name % 当前文件夹
    files = dir(fullfile(path, curfolder, 'acc*.csv')); % 当前文件夹下所有csv文件
    N = length(files);
    for i = 1:N 
        
        if folderIdx == 4;
            data=dlmread(fullfile(files(i).folder, files(i).name), ';' );
        else
            data = csvread(fullfile(files(i).folder, files(i).name));
        end
        acc = data(:, 5);
        [coeff, f] = cwt(acc, fs);
        H = abs(coeff);
        windows_size=40;
        for k = 1:size(H,1)-windows_size
            A(k) = sum(sum(H(k:k+windows_size,:)));
        end
        a = find(max(A));
        H = H(a:windows_size+a,:);
        %tt=[1:1:fs/10];
        %pcolor(tt,f,H);shading interp
        H = imresize(H, [128, 128]);
        pcolor(H);shading interp
        filename = sprintf('%s_%s_%d', folders(folderIdx).name, files(i).name(1:end-4), N);
        %saveas(gcf,['../Dataset/cwt_train_image/', filename, '.jpg'])
        %fprintf('%d %d %s\n',N,i,strcat(filename));
        save(['../Dataset/cwt_train_set/', filename, '.mat'], 'H')
    end     
end
%%
% save_path='E:\windows\学术\汇报\35 基于现有数据集的创新点子\transformer_informer_exe\project_for_paper\reference\Dataset\cwt_train_image\';   
% img_path_list = dir(strcat(save_path,'*.jpg')); 
% img_num=length(img_path_list);   %判断图片个数
% for i = 1:img_num     %因为拍照片的时候固定好了位置所以用一个for循环就可以截取出所有的图片的数字
%     picture_name =img_path_list(i).name;
%     picture = imread(strcat(save_path,picture_name));
%     imshow(picture);
%     %[x,y]=ginput(2);   %先用的ginput函数获取图片中数字的起始坐标
%     x=[116,790];
%     y=[53,584];
%     picture_1 =imcrop(picture,[x(1),y(1),abs(x(1)-x(2)),abs(y(1)-y(2))]);  %切割图像，起始坐标点（x1,y1）截取到终止坐标点(x2,y2)
%     imwrite(picture_1,['../Dataset/cwt_train_cropimage/', picture_name]);%将图片保存在程序所在文件夹中
% end

%% Extract test features
path = fullfile('..', 'Dataset', 'Test_set');
folders = dir(path);
folders = folders(3:end); % 所有文件夹

for folderIdx = 1:length(folders) 
    curfolder = folders(folderIdx).name % 当前文件夹
    files = dir(fullfile(path, curfolder, 'acc*.csv')); % 当前文件夹下所有csv文件
    N = length(files);
    for i = 1:N        
        data = csvread(fullfile(files(i).folder, files(i).name));
        acc = data(:, 5);
        [coeff, f] = cwt(acc, fs);
        
        H = abs(coeff);
        windows_size=40;
        for k = 1:size(H,1)-windows_size
            A(k) = sum(sum(H(k:k+windows_size,:)));
        end
        a = find(max(A));
        H = H(a:windows_size+a,:);
        %tt=[1:1:fs/10];
        %pcolor(tt,f,H);shading interp
        H = imresize(H, [128, 128]);
        pcolor(H);shading interp
        
        
        H = imresize(H, [128, 128]);
        pcolor(H);shading interp
        filename = sprintf('%s_%s_%d', folders(folderIdx).name, files(i).name(1:end-4), N);
        %saveas(gcf,['../Dataset/cwt_test_image/', filename, '.jpg'])

        save(['../Dataset/cwt_test_set/', filename, '.mat'], 'H')
        
    end
end
%%
% save_path='E:\windows\学术\汇报\35 基于现有数据集的创新点子\transformer_informer_exe\project_for_paper\reference\Dataset\cwt_test_image\';   
% img_path_list = dir(strcat(save_path,'*.jpg')); 
% img_num=length(img_path_list);   %判断图片个数
% for i = 1:img_num     %因为拍照片的时候固定好了位置所以用一个for循环就可以截取出所有的图片的数字
%     picture_name =img_path_list(i).name;
%     picture = imread(strcat(save_path,picture_name));
%     imshow(picture);
%     %[x,y]=ginput(2);   %先用的ginput函数获取图片中数字的起始坐标
%     x=[217,1480];
%     y=[92.75,1037.8];
%     picture_1 =imcrop(picture,[x(1),y(1),abs(x(1)-x(2)),abs(y(1)-y(2))]);  %切割图像，起始坐标点（x1,y1）截取到终止坐标点(x2,y2)
%     imwrite(picture_1,['../Dataset/cwt_test_cropimage/', picture_name, '.jpg']);%将图片保存在程序所在文件夹中
% end

% %%
% [coeff, f] = cwt(acc, fs);
% H = abs(coeff);
% % H = min(max(H, 0.03), 3);
% t = (1:length(acc))/fs;
% % H = (H - min(H(:)))./(max(H(:))- min(H(:)));
% 
% figure()
% pcolor(t, f/1000, H)
% xlabel('Time [s]')
% ylabel('Frequency [kHz]')
% yticks(2.^(-6:3))
% set(gca,'YScale','log') 
% shading interp
% colormap('jet')
% colorbar



