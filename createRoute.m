function lgraph = createRoute(lgraph, layerInfo, creationIdx)

    global layersList
    global numlayers
    global routeInfo
    
    persistent numroute
    if isempty(numroute)
        numroute = 0;
    end
    
    numroute = numroute + 1;
    lname = ['routing_layer_', num2str(numroute)];
    %分岐 or 結合元のレイヤ特定
    layers_info = contains(layerInfo, 'layers');
    layers_info = extractAfter(layerInfo{layers_info}, '=');
    if contains(layers_info, ',')
        idxs = strsplit(layers_info, ',');
        mode = 0; %分岐モード
    else
        idxs = str2double(layers_info);
        mode = 1; %結合モード
    end
        
    %分岐の場合
    %減算するレイヤの総数を算出
    if mode
        prevlayers = sum(numlayers(creationIdx+idxs+1:end)) + 1;
        tmptbl = flipud(layersList.All);
        routeInfo.source = tmptbl{prevlayers};
        routeInfo.flag = 1;
        layersList.All = [layersList.All; {lname, layersList.All{end, 2}}];
        layersList.Top = [layersList.Top; lname];                   
    %結合の場合
    else
        routeInfo.flag = 0;
        tmptbl = flipud(layersList.All);
        sz = size(idxs, 2);
        layer_reorg = depthConcatenationLayer(sz, 'Name',lname);
        lgraph = addLayers(lgraph, layer_reorg);  
        channelsz = 0;
        for ii = 1:sz
            if str2double(idxs{ii}) > 0
                concat_source = layersList.Top{str2double(idxs{ii})+2};
                % 畳み込み層がBatch NormalizationとReluでフォローされる場合、
                % 結合元ソースはReluとする
                idx = find(strcmp(layersList.All, concat_source));
                if ischar(layersList.All{idx+2}) && startsWith(layersList.All{idx+2}, 'leaky_relu')
                    idx = idx+2;
                    concat_source = layersList.All{idx};
                end
                channelsz = channelsz + layersList.All{idx,2};
            else
                prevlayers = sum(numlayers(creationIdx+str2double(idxs{ii})+1:end)) + 1;
                concat_source = tmptbl{prevlayers};
                channelsz = channelsz + tmptbl{prevlayers,2};
            end
            port = [lname, '/in', num2str(ii)];
            lgraph = connectLayers(lgraph,concat_source,port);
        end
        layersList.All = [layersList.All; {lname, channelsz}];
        layersList.Top = [layersList.Top; lname];
    end
    numlayers(creationIdx) = numlayers(creationIdx) + 1;
    %表示
    txt = ['Routing layer ', num2str(numroute)];
    disp(txt)
end

%% 
% Copyright 2019 The MathWorks, Inc.