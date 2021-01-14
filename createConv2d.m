function [lgraph, readSizeOut] = createConv2d(lgraph, layerInfo, fid, creationIdx, out, fold, weights_folded, bias_folded)

    global layersList
    global numlayers
    global routeInfo
    
    persistent numconv
    if isempty(numconv)
        numconv = 0;
    end
    persistent readSize
    if isempty(readSize)
        readSize = 0;
    end    
    
    %weightsファイルから読み出したパラメータサイズ格納用変数
    rbnsize = 0;
    rbsize = 0;
    rwsize = 0;
    lnames = [];
    
    numconv = numconv + 1;
    %フィルタ数抽出
    filters = contains(layerInfo, 'filters');
    filters = str2double(extractAfter(layerInfo{filters}, '='));
    %フィルタサイズ抽出
    sz = contains(layerInfo, 'size');
    sz = str2double(extractAfter(layerInfo{sz}, '='));
    %ストライド数抽出
    stride = contains(layerInfo, 'stride');
    stride = str2double(extractAfter(layerInfo{stride}, '='));
    %パディングサイズ抽出
    pad = contains(layerInfo, 'pad');
    pad = str2double(extractAfter(layerInfo{pad}, '='));
    %Activation(ReLU)有無
    activation = contains(layerInfo, 'activation');
    activation = extractAfter(layerInfo{activation}, '=');
    %Batch Norm有無
    bn = contains(layerInfo, 'batch_normalize');
    if ~sum(bn)
        bn = 0;
    else
        bn = str2double(extractAfter(layerInfo{bn}, '='));
    end
    %ストライド数に応じてパディングの仕方を変更    
    if stride > 1
        lname = ['zero_pad_', num2str(numconv)];
        layer_pad = nnet.keras.layer.ZeroPadding2dLayer(lname, [1, 0, 1, 0]);
        layers = layer_pad;
        lnames = [lnames; {lname, 0}];
        numlayers(creationIdx) = numlayers(creationIdx) + 1;
        padding = [0 0 0 0]; %デフォルト
        %表示
        txt = ['Zero padding layer : ',...
            'Padding size - ',num2str([1, 0, 1, 0])];
        disp(txt)   
    else
        layers = [];
        padding = 'same';
    end
    %畳み込み層のBias情報を抽出
    rbsize = filters;
    conv_bias = fread(fid, rbsize, '*single');
    conv_bias = reshape(conv_bias, [1 1 filters]);
    readSize = readSize + rbsize;
    %BatchNormalization有りの場合は関連パラメータ抽出
    if bn
        bn_scale = fread(fid, filters, '*single');
        bn_scale = reshape(bn_scale, [1 1 filters]);        
        bn_mean = fread(fid, filters, '*single');
        bn_mean = reshape(bn_mean, [1 1 filters]);             
        bn_var = fread(fid, filters, '*single');
        bn_var = reshape(bn_var, [1 1 filters]);
        rbnsize = (filters * 3);
        readSize = readSize + rbnsize;
    end
    %Routing層で分岐が選択された場合
    if routeInfo.flag
        idx = find(strcmp(layersList.All, routeInfo.source));
        channels = layersList.All{idx, 2};
    else
        channels = layersList.All{end, 2};
    end
    %畳み込み層のWeights情報を抽出
    rwsize = sz*sz*channels*filters;
    conv_weights = fread(fid, [sz*sz*channels*filters], '*single');
    readSize = readSize + rwsize;        
    conv_weights = reshape(conv_weights, [sz sz channels filters]);
    conv_weights = permute(conv_weights, [2 1 3 4]);
    %Batch Normalization有りの場合の層定義&追加
    if bn
        %畳み込み層の定義&追加
        % modified by Yikai Mao to output weights/bias
        % modified by Yikai Mao to accommodate for BN folding
        lname = ['conv2d_', num2str(numconv)];
        layer_conv = convolution2dLayer(sz, filters, 'Stride', stride, 'Padding', padding, 'Name', lname);
        if fold
            layer_conv.Weights = weights_folded{numconv};
            layer_conv.Bias = bias_folded{numconv};
        else
            layer_conv.Weights = conv_weights;
            layer_conv.Bias = zeros(1,1,filters,'single');
            if out
                %writematrix(layer_conv.Weights, strcat(lname,'_weights'));
                %writematrix(layer_conv.Bias, strcat(lname,'_bias'));
                save(strcat(lname,'_weights'), 'conv_weights');
                temp = layer_conv.Bias;
                save(strcat(lname,'_bias'), 'temp');
            end
        end
        layers = [layers; layer_conv];
        lnames = [lnames; {lname, filters}];
        numlayers(creationIdx) = numlayers(creationIdx) + 1;
        %表示
        txt = ['2D conv layer ', num2str(numconv), ' : ',...
            'Filtersize - ',num2str(filters),' : Stride - ',num2str(stride), ' : Read param# - ', num2str(rwsize)];
        disp(txt)                   
        %BatchNormalization層の定義&追加
        % modified by Yikai Mao to output weights/bias
        % modified by Yikai Mao to accommodate for BN folding
        lname = ['batch_norm_', num2str(numconv)];
        layer_bn = batchNormalizationLayer('Name', lname);
        if fold
            txt = ['Skipping batch norm ', num2str(numconv)];
            disp(txt)
        else
            layer_bn.Offset = conv_bias;
            layer_bn.Scale = bn_scale;
            layer_bn.TrainedMean = bn_mean;
            layer_bn.TrainedVariance = bn_var;
            layer_bn.Epsilon = 1e-3;
            if out
                %writematrix(layer_bn.Offset, strcat(lname,'_offset'));
                %writematrix(layer_bn.Scale, strcat(lname,'_scale'));
                %writematrix(layer_bn.TrainedMean, strcat(lname,'_mean'));
                %writematrix(layer_bn.TrainedVariance, strcat(lname,'_variance'));
                save(strcat(lname,'_offset'), 'conv_bias');
                save(strcat(lname,'_scale'), 'bn_scale');
                save(strcat(lname,'_mean'), 'bn_mean');
                save(strcat(lname,'_variance'), 'bn_var');
            end
            layers = [layers; layer_bn];
            lnames = [lnames; {lname, filters}];
            numlayers(creationIdx) = numlayers(creationIdx) + 1;
            %表示
            txt = ['Batch norm layer ', num2str(numconv), ' : Read param# - ', num2str(rbsize+rbnsize)];
            disp(txt)
        end
    %Batch Normalization無しの場合の層定義&追加
    else
        %畳み込み層の定義&追加
        % modified by Yikai Mao to output weights/bias
        % modified by Yikai Mao to accommodate for BN folding
        lname = ['conv2d_', num2str(numconv)];
        layer_conv = convolution2dLayer(sz, filters, 'Stride', stride, 'Padding', padding, 'Name', lname);
        if fold
            layer_conv.Weights = weights_folded{numconv};
            layer_conv.Bias = bias_folded{numconv};
        else
            layer_conv.Weights = conv_weights;
            layer_conv.Bias = conv_bias;
            if out
                %writematrix(layer_conv.Weights, strcat(lname,'_weights'));
                %writematrix(layer_conv.Bias, strcat(lname,'_bias'));
                save(strcat(lname,'_weights'), 'conv_weights');
                save(strcat(lname,'_bias'), 'conv_bias');
            end
        end
        layers = [layers; layer_conv];
        lnames = [lnames; {lname, filters}];
        numlayers(creationIdx) = numlayers(creationIdx) + 1;
        %表示
        txt = ['2D conv layer ', num2str(numconv), ' : ',...
            'Filtersize - ',num2str(filters),' : Stride - ',num2str(stride), ' : Read param# - ', num2str(rwsize+rbsize)];
        disp(txt)                      
    end
    %Activation(ReLu)層有りの場合
    if strcmp(activation, 'leaky')
        lname = ['leaky_relu_', num2str(numconv)];
        layer_act = leakyReluLayer(0.1, 'Name', lname);
        layers = [layers; layer_act];
        lnames = [lnames; {lname, filters}];
        numlayers(creationIdx) = numlayers(creationIdx) + 1;
        %表示
        txt = ['ReLU layer ', num2str(numconv), ' : ',...
            'Scale - ',num2str(0.1)];
        disp(txt)                      
    end
        
    lgraph = addLayers(lgraph, layers);
    %Routing層で分岐が選択されていた場合は接続先を変更
    if routeInfo.flag
        lgraph = connectLayers(lgraph,routeInfo.source, lnames{1, 1});
        routeInfo.flag = 0;
    else
        lgraph = connectLayers(lgraph,layersList.All{end, 1}, lnames{1, 1});
    end
    %書き込んだレイヤ情報の追加
    layersList.All = [layersList.All; lnames];
    layersList.Top = [layersList.Top; lnames(1)];
    readSizeOut = readSize;    
end

%% 
% Copyright 2019 The MathWorks, Inc.