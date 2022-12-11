function white = featureDistribution(rows, colums) 

white = ones(1000,1000,3)*255;
for i=1:90
    path=strcat("leaves/",int2str(i),".jpg");
    
    img=double(imread(path));
    [h w ch] = size(img);
    mask = true(h,w); %建立二值遮罩(mask)
    mask(find(img(:,:,3)>=240)) = 0; %假設影像藍色channel為240者, 必定為白色。
    mask(find(mask>0)) = 1; %其餘設為0
    mask = double(cat(3,mask,mask,mask));

    row=uint32(max(rows)-rows(i)+1);
    colum=colums(i);
    white(row:row+h-1,colum:colum+w-1,:) = white(row:row+h-1,colum:colum+w-1,:).*(not(mask)) + img.*mask;
end
white=uint8(white);
end