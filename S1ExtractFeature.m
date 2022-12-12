clear;clc;close all
%%
fs = 25.6e3;
g = 9.81;
map = jet(255);%figure中使用的颜色矩阵就是被扩充到256色的jet颜色，而不是原来64色的jet颜色。

%% Extract train features
path =  'G:\windows\projection\project for measurement\exe0\Dataset_new\phm-ieee-2012-data-challenge-dataset\Full_Test_Set\';% 文件夹路径
folders = dir(path);
folders = folders(3:end); % 所有文件夹

for folderIdx =4:5%length(folders)
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
        low = max(find(f>800))+1;
        high = max(find(f>2400));
        H = abs(coeff);
        %tt=[1:1:fs/10];
        %pcolor(tt,f,H);shading interp
        H=H(high:low,:);
        H = imresize(H, [128, 128]);
        H=flipud(H);
         %figure;
        %pcolor(H);shading interp
        filename = sprintf('%s_%s_%d', folders(folderIdx).name, files(i).name(1:end-4), N);
        %saveas(gcf,['../Dataset/cwt_train_image/', filename, '.jpg'])
        %fprintf('%d %d %s\n',N,i,strcat(filename));
        save(['G:\windows\projection\project for measurement\exe0\Dataset_new\cwt_test_set_use_1\', filename, '.mat'], 'H')
    end     
end



