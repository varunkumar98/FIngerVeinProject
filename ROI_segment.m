function [region, edges] = ROI_segment(img, mask_h, mask_w)

[img_h, img_w] = size(img);


if mod(img_h,2) == 0
    half_img_h = img_h/2 + 1;
else
    half_img_h = ceil(img_h/2);
end


mask = zeros(mask_h,mask_w);
mask(1:mask_h/2,:) = -1;
mask(mask_h/2 + 1:end,:) = 1;

img_filt = imfilter(img, mask,'replicate'); 



img_filt_up = img_filt(1:floor(img_h/2),:);
[~, y_up] = max(img_filt_up); 


img_filt_lo = img_filt(half_img_h:end,:);
[~,y_lo] = min(img_filt_lo);


region = zeros(size(img));
for i=1:img_w
    region(y_up(i):y_lo(i)+size(img_filt_lo,1), i) = 1;
end


edges = zeros(2,img_w);
edges(1,:) = y_up;
edges(2,:) = round(y_lo + size(img_filt_lo,1));