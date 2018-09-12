close all
clear all

% decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_entrance/';
% decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_entrance/RemoveTimeStamp';
decision_map_path = '/usr/not-backed-up/1_convlstm/prediction6_aug2_entrance/RemoveTimeStamp';
addpath('../')
%gt
testSeqPath = '/usr/not-backed-up/1_DATABASE';
load(fullfile(testSeqPath,'Entrance_FrameLevel_GT.mat'))
TestFoldersResult = [];

for numTestFolders = 1:6
    numTestFolders
%     frameGt = gt{1,numTestFolders};
    frameGt = FrameGt{1,numTestFolders};
    frame_error = h5read(fullfile(decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');    
    if numTestFolders == 6
        frame_error = frame_error(1:end-708);
        frameGt = frameGt(1:end-708);
    end
    
    %  --------------------------------------------
    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error);        
    frame_regular = 1 - frame_error;
    
    nSam = size(frame_error,1);
    cur_gt = frameGt(10:nSam+9);
%     cur_gt = frameGt(6:nSam+5);
    Result = [];
    for thres = 0:0.001:1 
%         MyMask = frame_error >= thres;
        MyMask = frame_regular <= thres;
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
x_frame = EER(FrameLevel(3,:),FrameLevel(2,:));
sprintf('frame EER: %d',x_frame)

% quit;
figure
plot(FrameLevel(3,:),FrameLevel(2,:),'gs-')
title('ROC curve on Exit dataset')
xlabel('FPR')
ylabel('TPR')
legend('Frame level','Location','NorthWest')
hold on
x = 0:0.1:1;
y = 1 - x;
plot(y,x,'r:')      


