close all
clear all
decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_avenue/';
addpath('../')
%gt
testSeqPath = '/usr/not-backed-up/1_DATABASE/Avenue Dataset';
load(fullfile(testSeqPath,'Avenue_FrameLevel_GT.mat'))%,'gt')   % FROM Avenue GroundTruth
% load(fullfile(testSeqPath,'Avenue_FrameLevel_cvprGT.mat'))%,'FrameGt')   % FROM CVPR GroundTruth
TestFoldersResult = [];
 
for numTestFolders = 12%1:21
    numTestFolders
    frameGt = gt{1,numTestFolders};
%     frameGt = FrameGt{1,numTestFolders}; %Avenue_FrameLevel_cvprGT.mat
    frame_error = h5read(fullfile(decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');    
    
   %  --------------------------------------------
    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error);        
    frame_error = 1 - frame_error;
    
    nSam = size(frame_error,1);
    cur_gt = frameGt(10:nSam+9);
%     cur_gt = frameGt(6:nSam+5);
    Result = [];
    for thres = 0:0.01:1 
        MyMask = frame_error <= thres;
        MyMask = MyMask';
        cur_mask = MyMask(1:nSam);        
        PosFrame = sum(cur_gt);
        NegFrame = sum(not(cur_gt));
        TPFr = sum(and(cur_gt,cur_mask));
        FPFr = sum(and(not(cur_gt),cur_mask));
        Result = [Result [thres; TPFr;PosFrame;FPFr;NegFrame]];     
    end
    TestFoldersResult(:,:,numTestFolders) = Result;
end
% save(fullfile(savePath1,'PixelAbMap_FrameLevel.mat'),'TestFoldersResult')
%% DRAW ROC curve
TestFoldersResult_ = sum(TestFoldersResult(2:end,:,:),3);
% frame level
Thresh = TestFoldersResult(1,:,numTestFolders);
FrameLevel = zeros(3,size(Thresh,2));
FrameLevel(1,:) =  Thresh;
FrameLevel(2,:) = TestFoldersResult_(1,:)./TestFoldersResult_(2,:);
FrameLevel(3,:) = TestFoldersResult_(3,:)./TestFoldersResult_(4,:);
% FrameLevel(:,end+1) = [FrameLevel(1,end)+0.1 0 0];
%
% Area Under Curve
FrameAUC = trapz(FrameLevel(3,:),FrameLevel(2,:));
sprintf('AUC for Frame level ROC: %d', FrameAUC)
% [x_frame,y_frame] = EER(FrameLevel(3,:),FrameLevel(2,:));
x_frame = EER(FrameLevel(3,:),FrameLevel(2,:));
sprintf('frame EER: %d',x_frame)
% save(fullfile(savePath1,'Frame_EER_AUC_ver2.mat'),'FrameAUC','x_frame')
% quit;
figure
plot(FrameLevel(3,:),FrameLevel(2,:),'gs-')
title('ROC curve on Ped1 dataset')
xlabel('FPR')
ylabel('TPR')
legend('Frame level','Location','NorthWest')
hold on
x = 0:0.1:1;
y = 1 - x;
plot(y,x,'r:')               



