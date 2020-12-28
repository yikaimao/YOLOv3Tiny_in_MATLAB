function out = letterbox_image(img, imgSz)

[img_h, img_w, ~] = size(img);

ratio = min(imgSz/img_w, imgSz/img_h);

% リサイズ後の画像サイズ設定
% Image height and width after resizing image
w = round(img_w * ratio);
h = round(img_h * ratio);

%リサイズ
%Resize Image
rimg = imresize(img, [h, w],'Method','bilinear','AntiAliasing',false);
%rimg = imresize(img, [h, w],'Method','nearest','AntiAliasing',false);

st_h = round((imgSz - h)/2) + 1;
st_w = round((imgSz - w)/2) + 1;

%キャンバス作成
%Creating background
if isfloat(img)
    out = ones(imgSz, imgSz, 3, 'like', img) * 0.5;
else
    out = ones(imgSz, imgSz, 3, 'like', img) * 128;
end

%キャンバスに画像挿入
out(st_h:st_h+h-1, st_w:st_w+w-1, :) = rimg;
end