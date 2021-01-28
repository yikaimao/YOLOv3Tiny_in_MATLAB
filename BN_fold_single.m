% Yikai Mao 1/12/2021
% fold batch norm parameters into conv weights/biases

weights_folded = cell(1,13);
bias_folded = cell(1,13);

[weights_folded{1},bias_folded{1}] = fold(1,3,3,16);
[weights_folded{2},bias_folded{2}] = fold(2,3,16,32);
[weights_folded{3},bias_folded{3}] = fold(3,3,32,64);
[weights_folded{4},bias_folded{4}] = fold(4,3,64,128);
[weights_folded{5},bias_folded{5}] = fold(5,3,128,256);
[weights_folded{6},bias_folded{6}] = fold(6,3,256,512);
[weights_folded{7},bias_folded{7}] = fold(7,3,512,1024);
[weights_folded{8},bias_folded{8}] = fold(8,1,1024,256);%1*1*1024*256
[weights_folded{9},bias_folded{9}] = fold(9,3,256,512);

% YOLO output 1
weights_folded{10} = cell2mat(struct2cell(load('conv2d_10_weights.mat')));%1*1*512*255
bias_folded{10} = cell2mat(struct2cell(load('conv2d_10_bias.mat')));

[weights_folded{11},bias_folded{11}] = fold(11,1,256,128);%1*1*256*128
[weights_folded{12},bias_folded{12}] = fold(12,3,384,256);%3*3*(256+128)*256

% YOLO output 2
weights_folded{13} = cell2mat(struct2cell(load('conv2d_13_weights.mat')));%1*1*256*255
bias_folded{13} = cell2mat(struct2cell(load('conv2d_13_bias.mat')));

function [weights_folded, bias_folded] = fold(layer, size, channels, filters)
weights = cell2mat(struct2cell(load(strcat('conv2d_',num2str(layer),'_weights.mat'))));
variance = permute(cell2mat(struct2cell(load(strcat('batch_norm_',num2str(layer),'_variance.mat')))), [4,1,2,3]);
scale = permute(cell2mat(struct2cell(load(strcat('batch_norm_',num2str(layer),'_scale.mat')))), [4,1,2,3]);
offset = permute(cell2mat(struct2cell(load(strcat('batch_norm_',num2str(layer),'_offset.mat')))), [4,1,2,3]);
mean = permute(cell2mat(struct2cell(load(strcat('batch_norm_',num2str(layer),'_mean.mat')))), [4,1,2,3]);

weights_folded = zeros(size,size,channels,filters, 'single');
for x = 1:1:size
    for y = 1:1:size
        for channel = 1:1:channels
            weights_folded(x,y,channel,:) = (scale .* weights(x,y,channel,:))./(sqrt(variance + 1e-3));
        end
    end
end

% previous bias is always 0
bias_folded = zeros(1,1,filters, 'single');
bias_folded(1,1,:) = ((scale(1,1,1,:) .* (0 - mean(1,1,1,:)))./(sqrt(variance(1,1,1,:) + 1e-3))) + offset(1,1,1,:);

save("weights_folded", 'weights_folded');
save("bias_folded", 'bias_folded');
end