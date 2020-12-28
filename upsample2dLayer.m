classdef upsample2dLayer < nnet.layer.Layer

    properties
        size
    end
    
    methods
        function layer = upsample2dLayer(name, size)
            layer.Name = name;
            text = ['[', num2str(1), ' ', num2str(1), '] upsampling for YOLOv3'];
            layer.Description = text;
            layer.Type = ['up sampling 2d'];
            layer.size = size;
        end
        
        function Z = predict(layer, X)
               Z = repelem(X, layer.size, layer.size);
        end
        
        function [dX] = backward( layer, X, ~, dZ, ~ )
            dX = dZ(1:layer.size:end, 1:layer.size:end, :, :);
        end        
    end
end

%% 
% Copyright 2019 The MathWorks, Inc.