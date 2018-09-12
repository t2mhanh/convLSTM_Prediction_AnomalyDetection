close all
clear all

decision_map_path = '/usr/not-backed-up/1_convlstm/Ped1_AE6/';
% decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_ped1/';
addpath('../')
%gt
testSeqPath = '/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/';
% load(fullfile(testSeqPath,'TestFrameGT.mat'))%,'FrameGt')   % FROM UCSD 
load(fullfile(testSeqPath,'TestFrameGT_original.mat'))%,'FrameGt')   % FROM UCSD 
% load(fullfile(testSeqPath,'TestFrameGT_Antic.mat'),'FrameGt')   
TestFoldersResult = [];
%%% max and min value of frame_error
% min_error = [];
% max_error = [];

% for folders = 1:36
%     frame_error = h5read(fullfile(decision_map_path,['test_' num2str(folders) '_error.h5']),'/frame_error');
%     size(frame_error)
%     %   --------- smooth the error ----------------
%     nfr_sm = 5; % 5 give the best results
%     nSam = size(frame_error,1);
%     frame_error_sm = zeros(nSam-nfr_sm,1);
%     for fr = 1:nSam-nfr_sm
%         frame_error_sm(fr) = mean(frame_error(fr:fr+nfr_sm));
%     end
%     frame_error = frame_error_sm;
%     %--------------------------------------
%     min_error(folders) = min(frame_error);
%     max_error(folders) = max(frame_error);
% end

% for numTestFolders = 1:36
%     numTestFolders    
%     frame_error = h5read(fullfile(decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');    
%     %   --------- smooth the error ----------------
%     nfr_sm = 5; % 5 give the best results
%     nSam = size(frame_error,1);
%     frame_error_sm = zeros(nSam-nfr_sm,1);
%     for fr = 1:nSam-nfr_sm
%         frame_error_sm(fr) = mean(frame_error(fr:fr+nfr_sm));
%     end
%     frame_error = frame_error_sm;
%     %  --------------------------------------------
%     frame_error = (frame_error - min(frame_error));
%     frame_error = frame_error / max(frame_error);  
% end
% figure
% plot(1:size(frame_error,1),frame_error)

% 
for numTestFolders = 1:36
    numTestFolders
    frameGt = FrameGt{1,numTestFolders};
    frame_error = h5read(fullfile(decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');    
%     frame_error = frame_error(5:end);

    %  --------------------------------------------
    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error);        
% GLOBAL NORMALIZATION -> WORSE RESULT    
%     frame_error = (frame_error - min(min_error));
%     frame_error = frame_error / max(max_error);        
    
    nSam = size(frame_error,1);
    cur_gt = frameGt(4:nSam+3);
%     cur_gt = frameGt;
%     cur_gt = frameGt(6:nSam+5);
    Result = [];
    for thres = 0:0.01:1 
        MyMask = frame_error >= thres;
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
FrameLevel(:,end+1) = [FrameLevel(1,end)+0.1 0 0];
%
% Area Under Curve
FrameAUC = trapz(FrameLevel(3,:),FrameLevel(2,:));
sprintf('AUC for Frame level ROC: %d', FrameAUC)
x_frame = EER(FrameLevel(3,:),FrameLevel(2,:));
% [x_frame,y_frame] = EER(FrameLevel(3,:),FrameLevel(2,:));
sprintf('patial pixel EER: %d',x_frame)
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

%% 1D PERSISTENCE (not change the results)
% addpath(genpath('../Persistence1D/reconstruct1d'))
% for numTestFolders = 1:36%31:36%1:36
%     numTestFolders
%     frameGt = FrameGt{1,numTestFolders};
%     frame_error = h5read(fullfile(decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');   
%     single_precision_data = single(frame_error);
% 
%     % Run Persistence1D on the data
%     [minIndices maxIndices persistence globalMinIndex globalMinValue] = run_persistence1d(single_precision_data); 
% 
%     % Set threshold for surviving features
%     threshold = 0.9;
% 
%     % Filter Persistence1D paired extrema for relevant features
%     pairs = filter_features_by_persistence(minIndices, maxIndices, persistence, threshold);
% 
%     % interpolated minima and maxima
%     mins = get_min_indices(pairs);
%     maxs = get_max_indices(pairs);
% 
%     % Set the data weight. Choosing 0.0 constructs smoother function
%     data_weight = 0;
% 
%     % Set the smoothness for the results. 
%     bi_smoothness = 'biharmonic';
% 
%     % Call reconstruct1d_with_persistence_res. Calling this function directly avoids
%     % re-running Persistence1D for different reconstruction parameters
%     x_bi_smooth  = reconstruct1d_with_persistence_res( frame_error, ...
%                                                         mins, ...
%                                                         maxs, ...
%                                                         globalMinIndex, ...
%                                                         bi_smoothness, ...
%                                                         data_weight);
%     %   --------- smooth the error ----------------
% %     nfr_sm = 5; % 5 give the best results
% %     nSam = size(x_bi_smooth,1);
% %     frame_error_sm = zeros(nSam-nfr_sm,1);
% %     for fr = 1:nSam-nfr_sm
% %         frame_error_sm(fr) = mean(x_bi_smooth(fr:fr+nfr_sm));
% %     end
% %     x_bi_smooth = frame_error_sm;
%     %  --------------------------------------------
%     %  --------------------------------------------
%     x_bi_smooth = (x_bi_smooth - min(x_bi_smooth));
%     x_bi_smooth = x_bi_smooth / max(x_bi_smooth);        
% % GLOBAL NORMALIZATION -> WORSE RESULT    
% %     frame_error = (frame_error - min(min_error));
% %     frame_error = frame_error / max(max_error);        
%     
%     nSam = size(x_bi_smooth,1);
% %     cur_gt = frameGt(6:nSam+5);
%     cur_gt = frameGt(9:nSam+8);
%     Result = [];
%     for thres = 0:0.05:1 %[-0.02:0.001:0.1 0.11:0.01:1]%0.004%   
%         MyMask = x_bi_smooth >= thres;
%         MyMask = MyMask';
%         cur_mask = MyMask(1:nSam);        
%         PosFrame = sum(cur_gt);
%         NegFrame = sum(not(cur_gt));
%         TPFr = sum(and(cur_gt,cur_mask));
%         FPFr = sum(and(not(cur_gt),cur_mask));
%         Result = [Result [thres; TPFr;PosFrame;FPFr;NegFrame]];     
%     end
%     TestFoldersResult(:,:,numTestFolders) = Result;
% end
% % save(fullfile(savePath1,'PixelAbMap_FrameLevel.mat'),'TestFoldersResult')
% %% DRAW ROC curve
% TestFoldersResult_ = sum(TestFoldersResult(2:end,:,:),3);
% % frame level
% Thresh = TestFoldersResult(1,:,numTestFolders);
% FrameLevel = zeros(3,size(Thresh,2));
% FrameLevel(1,:) =  Thresh;
% FrameLevel(2,:) = TestFoldersResult_(1,:)./TestFoldersResult_(2,:);
% FrameLevel(3,:) = TestFoldersResult_(3,:)./TestFoldersResult_(4,:);
% FrameLevel(:,end+1) = [FrameLevel(1,end)+0.1 0 0];
% %
% % Area Under Curve
% FrameAUC = trapz(FrameLevel(3,:),FrameLevel(2,:));
% sprintf('AUC for Frame level ROC: %d', FrameAUC)
% x_frame = EER(FrameLevel(3,:),FrameLevel(2,:));
% % % [x_frame,y_frame] = EER(FrameLevel(3,:),FrameLevel(2,:));
% sprintf('patial pixel EER: %d',x_frame)
% % save(fullfile(savePath1,'Frame_EER_AUC_ver2.mat'),'FrameAUC','x_frame')
% % quit;
% figure
% plot(FrameLevel(3,:),FrameLevel(2,:),'gs-')
% title('ROC curve on Ped1 dataset')
% xlabel('FPR')
% ylabel('TPR')
% legend('Frame level','Location','NorthWest')
% hold on
% x = 0:0.1:1;
% y = 1 - x;
% plot(y,x,'r:')  

