function [lgraph, anchors, classes, mask] = importDarknetNetwork(weightsfile, cfgfile)

%% ダウンロードしたファイルのオープン
fid = fopen(weightsfile, 'rb');
fidcfg = fopen(cfgfile);

%% 検証用にweightsファイルのサイズ確認
weights = fread(fid);
wSize = size(weights,1);
clear weights
fseek(fid,0,'bof'); %先頭に移動

%% weightsヘッダ読み込み
header = fread(fid, 3, '*int32');
%txt = ['Major : ',num2str(data(1)),' Minor : ',num2str(data(2)),' Revision : ',num2str(data(3))];
%disp(txt)
if header(2) > 1
    header2 = fread(fid, 1, '*int64');
else
    header2 = fread(fid, 1, '*int32');
end
txt = ['Darknet weights file version info', newline,...
    'Major : ',num2str(header(1)),' Minor : ',num2str(header(2)),' Revision : ',num2str(header(3))];
disp(txt)
headerSize = ftell(fid);

%% cfgファイルの読み込み
txt = 'Loading cfg file';
disp(txt)

cfg = textscan(fidcfg, '%s', 'Delimiter',{'   '});
fclose(fidcfg);
cfg = cfg{1,1};

%% 各レイヤの情報抽出
% []の数をカウント
stlayer = strncmp(cfg, '[', 1);
layerIdx = find(stlayer);

%% ネットワークの作成
% cfgファイルからネットワークのアーキテクチャを読み込み、
% 該当weightsをロードして各レイヤに設定
txt = ['Creating graph of network layers for deep learing', newline,...
    '--------------------------------------------------------------------'];
disp(txt)
%YOLO用パラメータ
mask = 0;
anchors = 0;
classes = 0;
%書き込んだ層情報格納用変数
global layersList;
global numlayers;
global routeInfo;
routeInfo.flag = 0;
numlayers = zeros(size(layerIdx,1), 1);
%cfgファイルで定義されている各層を追加
for i = 1:size(layerIdx,1)
    %定義するレイヤ名およびパラメータを抽出
    layerName = cfg{layerIdx(i)};
    if i == size(layerIdx,1)
        layerInfo = cfg(layerIdx(i)+1:end);
    else
        layerInfo = cfg(layerIdx(i)+1:layerIdx(i+1)-1);
    end
    
    if strcmp(layerName, '[net]')
        %入力サイズ抽出
        h = contains(layerInfo, 'height');
        h = str2double(extractAfter(layerInfo{h}, '='));
        w = contains(layerInfo, 'width');
        w = str2double(extractAfter(layerInfo{w}, '='));
        c = contains(layerInfo, 'channels');
        c = str2double(extractAfter(layerInfo{c}, '='));
        %レイヤ定義
        lnames = 'input';
        layers = imageInputLayer([h w c],'Normalization','none','Name',lnames);
        numlayers(i) = 1;
        lgraph = layerGraph(layers);
        layersList.All = {lnames, c};
        layersList.Top = {lnames};
        %表示
        txt = ['Image input layer : ',...
            'Height - ',num2str(h),' Width - ',num2str(w),' Channel - ',num2str(c)];
        disp(txt)
    elseif strcmp(layerName, '[convolutional]')
        % Convolutional層(conv2d, batchNorm, ReLu)の作成
        [lgraph, readSize] = createConv2d(lgraph, layerInfo, fid, i);
        
    elseif strcmp(layerName, '[maxpool]')
        % Max Pooling層の作成
        lgraph = createMaxpool(lgraph, layerInfo, i);        

    elseif strcmp(layerName, '[avgpool]')
        % Average Pooling層の作成
        lgraph = createAvgpool(lgraph, i);

    elseif strcmp(layerName, '[softmax]')
        % Softmax層の作成
        lgraph = createSoftmax(lgraph, i);
        
    elseif strcmp(layerName, '[shortcut]')
        % Shortcut層の作成(addition)
        lgraph = createShortcut(lgraph, layerInfo, i);
        
    elseif strcmp(layerName, '[route]')
        % Route層の作成(分岐か結合(depthConcat))
        % 分岐の場合はrouteInfo必要
        lgraph = createRoute(lgraph, layerInfo, i);

    elseif strcmp(layerName, '[reorg]')
        % Reorganization層の作成
        lgraph = createReorg(lgraph, layerInfo, i);

    elseif strcmp(layerName, '[upsample]')
        % カスタムレイヤ : Upsampling2D層の作成(YOLOv3で利用)
        lgraph = createUpsample(lgraph, layerInfo, i);
        
        % modified by Yikai Mao, change [region] to [yolo]
        % YOLOv3 uses [yolo] layer instead of [region] layer
    elseif strcmp(layerName, '[yolo]')
        %アンカーの情報抽出
        anchors = contains(layerInfo, 'anchors');
        %スペースがが含まれる場合は削除
        anchors = strrep(layerInfo{anchors},' ','');
        %'='の右側を取得
        anchors = extractAfter(anchors, '=');
        anchors = str2double(strsplit(anchors, ','));
        anchors = reshape(anchors, [2, size(anchors, 2)/2, ])';
        
        % modified by Yikai Mao, extract mask values
        temp = contains(layerInfo, 'mask');
        temp = strrep(layerInfo{temp},' ','');
        temp = extractAfter(temp, '=');
        if mask == 0
            mask = str2double(strsplit(temp, ','));
        else
            mask = [mask;str2double(strsplit(temp, ','))];
        end
        
        %クラス数抽出
        classes = contains(layerInfo, 'classes');
        %スペースがが含まれる場合は削除
        classes = strrep(layerInfo{classes},' ','');
        %'='の右側を取得
        classes = extractAfter(classes, '=');
        classes = str2double(strsplit(classes, ','));
    end
    
end
txt = ['--------------------------------------------------------------------', newline,...
    'Total params ', num2str(headerSize + (readSize*4)), newline,...
    '--------------------------------------------------------------------'];
disp(txt)
txt = ['Read ', num2str(headerSize + (readSize*4)), ' of ' ,num2str(wSize), ' params from Darknet weights', newline,...
    '--------------------------------------------------------------------'];
disp(txt)

mask = num2cell(mask+1, 2);

fclose(fid);

end

%% 
% Copyright 2019 The MathWorks, Inc.