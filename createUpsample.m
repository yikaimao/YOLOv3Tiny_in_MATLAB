function lgraph = createUpsample(lgraph, layerInfo, creationIdx)

    global layersList
    global numlayers
    global routeInfo    
    
    persistent numupsample
    if isempty(numupsample)
        numupsample = 0;
    end
    
    numupsample = numupsample + 1;
    lname = ['upsampling_layer_', num2str(numupsample)];
    %アップサンプリングサイズ抽出
    stride = contains(layerInfo, 'stride');
    stride = str2double(extractAfter(layerInfo{stride}, '=')); 
    %Upsampleレイヤ追加
    layer_upsamp = upsample2dLayer(lname, stride);
    lgraph = addLayers(lgraph, layer_upsamp);
    
    %Routing層で分岐が選択されていた場合は接続先を変更
    if routeInfo.flag
        lgraph = connectLayers(lgraph,routeInfo.source, lname);
        routeInfo.flag = 0;
    else
        lgraph = connectLayers(lgraph,layersList.All{end, 1}, lname);
    end
    
    %出力チャネル数
    outsize = layersList.All{end, 2};
    layersList.All = [layersList.All; {lname, outsize}];
    layersList.Top = [layersList.Top; lname];        
    numlayers(creationIdx) = numlayers(creationIdx) + 1;        
    %表示
    txt = ['Upsampling layer ', num2str(numupsample)];
    disp(txt)       
end

%% 
% Copyright 2019 The MathWorks, Inc.