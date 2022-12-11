%image feature extraction: signature
function [signature_gradient, redness, lightness, laplace_texture] = leaveFeature(path)
%輸入葉片影像

img1 = double(imread(path));

%根據背景色(白色)，建立二值遮罩(mask)。背景數值為0，樹葉(前景)數值為1。
[h w ch] = size(img1);
mask = true(h,w); %建立二值遮罩(mask)
mask(find(img1(:,:,3)>=240)) = 0; %假設影像藍色channel為240者, 必定為白色。
mask(find(mask>0)) = 1; %其餘設為0

%計算樹葉的面積。也就是mask的數值總和(A)。
A = sum(mask(:));

%計算葉子的質心座標(c)與邊界與質心間的距離r(i)
%透過(x,y)雙層迴圈，分別累加mask中，前景畫素(數值為1)的x位置與y位置的數值。
sum_x=0;
sum_y=0;
for x=1:h
    for y=1:w
        if mask(x,y) == 1
            sum_x = x + sum_x;
           sum_y = y + sum_y;
        end
    end
end

cx = round(sum_x/A); %質心的x座標=累加x的總和除以A。
cy = round(sum_y/A); %質心的y座標=累加y的總和除以A。

%在mask上繪製交會於樹葉質心的紅色十字線。
img2 = double(cat(3,mask,mask,mask));
img2(cx,:,2:3) = 0;
img2(:,cy,2:3) = 0;
img2(cx,:,1) = 1;
img2(:,cy,1) = 1;
img3 = zeros(h+2,w+2,3);
img3(2:end-1,2:end-1,:) = img1;

%建立一個有60個bin的r_histogram。60個bin的初始值都是0。
bin=60; %可改成15, 30, 60,120, 360(不建議,會有空bin)
interval = 360/bin;
r_histogram = zeros(bin,1);

%透過(x,y)雙層迴圈，計算mask中，前景畫素(數值為1)的畫素，其遠離質心c的角度theta與距離r。
%i = ceil(theta/6+0.01)，如果r_histogram(i)小於r，則r_histogram(i) = r。
for x=1:h
    for y=1:w
        if mask(x,y) ==1
           theta = mod(atan2(cx-x,y-cy)*180/pi,360); %換算theta角
           i = ceil(theta/interval+.01); %0.01是為了防止i=0
           r = sqrt((cx-x)^2+(cy-y)^2); %遠離質心c的距離
           if r > r_histogram(i)
                r_histogram(i) = r; %更新r_histogram中的最大r值
           end
        end
    end
end

%% calculate gradient of signature
signature_gradient = mean(abs(gradient(r_histogram)));

%% Redness
img4 = double(cat(3,mask,mask,mask));
img4 = img1.*img4;
sum_r = sum((img4(:,:,1)),"all");
sum_rgb = sum(img4,"all");
redness = sum_r/sum_rgb;

%% Lightness
img5 = double(rgb2gray(uint8(img1)));
img5 = img5.*mask;
lightness = sum(img5,"all")/sum(mask,"all");

%% Laplace Texture
laplace_filter = fspecial("laplacian");
mean_filter = fspecial("average",size(laplace_filter));
img6 = abs(imfilter(img5, laplace_filter));
img6 = imfilter(img6,mean_filter);
laplace_texture = sum(img6,"all")/sum(img6>0,"all");
end