function lgraph = createMaxpool(lgraph, layerInfo, creationIdx)

    global layersList
    global numlayers
    global routeInfo    
    
    persistent nummpool
    if isempty(nummpool)
        nummpool = 0;
    end
    
    nummpool = nummpool + 1;
    %フィルタサイズ抽出
    sz = contains(layerInfo, 'size');
    sz = str2double(extractAfter(layerInfo{sz}, '='));
    %ストライド数抽出
    stride = contains(layerInfo, 'stride');
    stride = str2double(extractAfter(layerInfo{stride}, '='));
    %Max Pooling層の定義&追加
    lname = ['max_pooling_', num2str(nummpool)];
    layer_pool = maxPooling2dLayer(sz, 'Stride', stride,...
        'Padding', 'same', 'Name', lname);
    lgraph = addLayers(lgraph, layer_pool);
    
    %Routing層で分岐が選択されていた場合は接続先を変更
    if routeInfo.flag
        lgraph = connectLayers(lgraph,routeInfo.source, lname);
        routeInfo.flag = 0;
    else
        lgraph = connectLayers(lgraph,layersList.All{end, 1}, lname);
    end
    %lgraph = connectLayers(lgraph,layersList.All{end, 1}, lname);    
    
    % 書き込んだレイヤ情報の追加
    layersList.All = [layersList.All; {lname, layersList.All{end, 2}}];
    layersList.Top = [layersList.Top; lname];        
    numlayers(creationIdx) = numlayers(creationIdx) + 1;
    %表示
    txt = ['Max pooling layer ', num2str(nummpool), ' : ',...
        'Poolsize - ',num2str(sz),' Stride - ',num2str(stride)];
    disp(txt)
end

%% 
% Copyright 2019 The MathWorks, Inc.