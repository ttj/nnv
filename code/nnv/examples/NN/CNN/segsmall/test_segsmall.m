% run example first
% openExample('deeplearning_shared/TrainSemanticSegmentationNetworkUsingDilatedConvolutionsExample')
% TrainSemanticSegmentationNetworkUsingDilatedConvolutionsExample

nnvNet = CNN.parse(net, 'SegSmall');


V(:,:,:,1) = double(imgTest);
V(:,:,:,2) = double(imgTest) * 0.99;

l = [0.98999];
delta = 0.0000002;
n = length(l);

pred_lb = zeros(n, 1);
pred_ub = zeros(n, 1);
robust_exact = zeros(n, 1);
robust_approx = zeros(n, 1);
VT_exact = zeros(n, 1);
VT_approx = zeros(n, 1);


for i=1:n
    pred_lb(i) = l(i);
    pred_ub(i) = l(i) + delta;
    
    C = [1;-1];   % pred_lb % <= alpha <= pred_ub percentage of FGSM attack
    d = [pred_ub(i); -pred_lb(i)]; 
    IS = ImageStar(V, C, d, pred_lb(i), pred_ub(i));
 
    % obj.reach(in_image, method, numOfCores);
    nnvNet.reach(IS, 'approx-star');
%    t = tic;
%    [robust_exact(i), ~] = nnvNet.verifyRobustness(IS, correct_id, 'exact-star');
%    VT_exact(i) = toc(t);
%    t = tic; 
%    [robust_approx(i), ~] = nnvNet.verifyRobustness(IS, correct_id, 'approx-star');
%    VT_approx(i) = toc(t);
end

nI = size(nnvNet.reachSet{end}.V(:,:,:,1),3);

for i=1:nI
    figure('units','normalized','outerposition',[0 0 1 1]);
    hold on;
    imshow(nnvNet.reachSet{end}.V(:,:,i,1))
    imshow(nnvNet.reachSet{end}.V(:,:,i,2))
end