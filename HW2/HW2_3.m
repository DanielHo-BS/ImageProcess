%HW2.3
clear;clc;
%feature = [signature_gradient, redness, lightness, laplace_texture]
signature_gradient= zeros(90,1);
redness= zeros(90,1);
lightness= zeros(90,1);
laplace_texture= zeros(90,1);

for i=1:90
path=strcat("leaves/",int2str(i),".jpg");
[signature_gradient(i), redness(i), lightness(i), laplace_texture(i)]=leaveFeature(path);
end
feature = [signature_gradient, redness, lightness, laplace_texture];
% normalization feature = uint32[1:800]
for i=1:4
    feature(:,i) = uint32((feature(:,i)-min(feature(:,i)))/(max(feature(:,i))-min(feature(:,i)))*800+1);
end

figure()
white = featureDistribution(feature(:,3),feature(:,2));
subplot(121)
imshow(white)
xlabel("Redness")
ylabel("Lightness")

white = featureDistribution(feature(:,4),feature(:,1));
subplot(122)
imshow(white)
xlabel("Absolute gradient of the signature cure")
ylabel("Laplacian Texture")
