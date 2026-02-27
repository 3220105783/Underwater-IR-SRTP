

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  Chongyi Li, Chunle Guo, Wenqi Ren, Runmin Cong, Junhui Hou, Sam Kwong, Dacheng Tao , "An Underwater Image Enhancement Benchmark Dataset and Beyond" IEEE TIP 2019 %%%%%%%%%%%%%%%%%                                                                                                           %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;

folder2 = 'folder of raw images';
filepaths2 = dir(fullfile(folder2,'*.jpg'));

global count
count =1;

if ~exist('real_test') 
    mkdir('real_test')         
end 

if ~exist('real_wb') 
    mkdir('real_wb')         
end 
if ~exist('real_ce') 
    mkdir('real_ce')         
end 


if ~exist('real_gc') 
    mkdir('real_gc')         
end 

for i=1:10000
    
    image1 = im2double(imread(fullfile(folder2,filepaths2(i).name)));
    
    %%% white balance
    hazy_wb = SimplestColorBalance(uint8(255*image1));
    hazy_wb = uint8(hazy_wb);
    
    %%% CLAHE
    lab1 = rgb_to_lab(uint8(255*image1));
    lab2 = lab1;
    lab2(:, :, 1) = adapthisteq(lab2(:, :, 1));
    img2 = lab_to_rgb(lab2);
    hazy_cont = img2;
    %%% gamma correction
    hazy_gamma = image1.^0.7;
    hazy_gamma =  uint8(255*hazy_gamma);
    
    image1 = uint8(255*image1);
    
    imwrite(image1,fullfile('real_test',filepaths2(i).name ));
    imwrite(hazy_wb,fullfile('real_wb',filepaths2(i).name ));
    imwrite(hazy_cont,fullfile('real_ce',filepaths2(i).name ));
    imwrite(hazy_gamma,fullfile('real_gc',filepaths2(i).name ));


end

