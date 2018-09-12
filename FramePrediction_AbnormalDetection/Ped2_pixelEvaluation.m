close all
clear all

decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction16_ped2/pixel_error';
% decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_ped1/';
addpath('../')
%gt
testSeqPath = '/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/';
% load(fullfile(testSeqPath,'TestFrameGT.mat'))%,'FrameGt')   % FROM UCSD 
load(fullfile(testSeqPath,'TestFrameGT_original.mat'))%,'FrameGt')   % FROM UCSD 

GtDir = '/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/VideoSequences/UCSDped2/Test_gt'; % ground truth

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
    if numTestFolders < 10
        GroundTruthPath = fullfile(GtDir,['Test00' num2str(numTestFolders) '_gt.mat']);
    else
        GroundTruthPath = fullfile(GtDir,['Test0' num2str(numTestFolders) '_gt.mat']);
    end
    load(GroundTruthPath)
    groundTruthAll = M./max(M(:)); %M[nr x nc x numFrames]    
    frameGt = FrameGt{1,numTestFolders}; % pixel GT from Antic differs from Frame level GT from UCSD -> use UCSD GT to remove frame with wrong annotation
    
    clear M
    pixel_error = h5read(fullfile(decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/pixel_error');    
    pixel_error = permute(pixel_error,[3,2,4,1]);
    error = sum(pixel_error,3);
    nfr = size(error,4);
    figure(1)
    for i = 1:nfr   
%         i
        cur_gt = imresize(groundTruthAll(:,:,i+9),[227,227],'bilinear');
        cur_gt = frameGt(i+9) .* cur_gt;
        [I,J] = find(cur_gt == 1); %if frame is abnormal
        if length(I) ~= 0
%             gt = zeros(227,227,3);
            gt(:,:,1) = ones(227,227);% cur_gt;
            gt(:,:,2) = not(cur_gt);
            gt(:,:,3) = not(cur_gt);
        else
            gt = ones(227,227,3);
        end
%         subplot(1,2,1)
% %         subaxis(1,2,1, 'sh', 0.03, 'sv', 0.01, 'padding', 0, 'margin', 0);
%         imshow(gt)
%         axis off
%         axis equal
%         
%         cur_error = error(:,:,i);
%         cur_error(1,1) = 0;
%         max_thres = 0.8;
%         cur_error(end,end) = max_thres;
%         cur_error = min(cur_error,max_thres);
%         subplot(1,2,2)
% %         subaxis(1,2,2, 'sh', 0.03, 'sv', 0.01, 'padding', 0, 'margin', 0);
%         imagesc(cur_error)
%         axis off
%         axis equal
        
        
        
        %
        hAxis(1) = subplot(1,2,1);
%         subaxis(1,2,1, 'sh', 0.03, 'sv', 0.01, 'padding', 0, 'margin', 0);
        imshow(gt)
        axis off
        axis equal
        
        cur_error = error(:,:,i);
        cur_error(1,1) = 0;
        max_thres = 0.8;%0.8
        cur_error(end,end) = max_thres;
        cur_error = min(cur_error,max_thres);
        hAxis(2) = subplot(1,2,2);
%         subaxis(1,2,2, 'sh', 0.03, 'sv', 0.01, 'padding', 0, 'margin', 0);
        imagesc(cur_error)
        axis off
        axis equal
        
        pos1 = get(hAxis(1),'Position');
%         pos1(1) = 0.17;
%         pos1(1) = pos1(1) - 0.1;
        pos1(1) = pos1(1) + 0.04;
%         pos1(2) = pos1(2) - 0.1;
%         pos1(3) = pos1(3) + 0.4;
%         pos1(4) = pos1(4) + 0.4;
        set(hAxis(1),'Position',pos1)
        
        pos2 = get(hAxis(2),'Position');
%         pos2(1) = 0.53;%pos2(1) - 4;
%         pos2(1) = pos2(1) - 0.2;
        pos2(1) = pos2(1) - 0.04;
%         pos2(2) = pos2(2) - 0.1;
%         pos2(3) = pos2(3) + 0.4;
%         pos2(4) = pos2(4) + 0.4;
        set(hAxis(2),'Position',pos2)
        %
        pause(0.1)
    end
end



