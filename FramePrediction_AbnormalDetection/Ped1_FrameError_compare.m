close all
clear all

% aug1
aug1_decision_map_path = '/usr/not-backed-up/1_convlstm/Ped1_prediction6/';
% aug2
aug2_decision_map_path = '/usr/not-backed-up/1_convlstm/Ped1_prediction_aug2/';
% recons-aug1
recons_aug1 = '/usr/not-backed-up/1_convlstm/Ped1_AE6/';
addpath('../')
%gt
testSeqPath = '/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/';

load(fullfile(testSeqPath,'TestFrameGT_original.mat'))%,'FrameGt')   % FROM UCSD 
%% visualization

for numTestFolders = [1 5 24 17 23]%1:36
    numTestFolders
    % aug1 
    frame_error = h5read(fullfile(aug1_decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');   
    nSam = size(frame_error,1);
    
    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error); 
    frame_regular_aug1 = 1 - frame_error;
    
    % aug2 
    frame_error = h5read(fullfile(aug2_decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');   
        
    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error); 
    frame_regular_aug2 = 1 - frame_error;
    
    % recons-aug1
    frame_error = h5read(fullfile(recons_aug1,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');   
%     frame_error = frame_error(6:end,1); % ignore error of the first 5 frames to match it with prediction.
    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error); 
    frame_regular_recons_aug1 = 1 - frame_error;
    frame_regular_recons_aug1 = frame_regular_recons_aug1(6:end,1); % ignore error of the first 5 frames to match it with prediction.
    
    % groundtruth
    frameGt = FrameGt{1,numTestFolders};    
    
    cur_gt = frameGt(10:nSam+9); 
    [~,c] = find(cur_gt == 1);
    gap = c(2:end) - c(1:end-1);
    [~,c_gap] = find(gap~= 1);
    if length(c_gap) == 0
        new_cur_gt = cur_gt;
    else
        new_cur_gt  = zeros(length(c_gap)+1,size(cur_gt,2));    
        new_cur_gt(1,c(1):c(c_gap(1))) = 1;
        for i = 1:length(c_gap)
            if i == length(c_gap)
                new_cur_gt(i+1,c(c_gap(i)+ 1)  : c(end)) = 1;
            else
                new_cur_gt(i+1,c(c_gap(i)+ 1 ) : c(c_gap(i+1))) = 1;
            end
        end
    end
    
    
    
    close all    
    hAxis = figure(1);
%     plot(frame_regular_aug1,'b')
%     hold on
    plot(1:nSam,frame_regular_aug1,'b',1:nSam,frame_regular_aug2,'r',1:nSam,frame_regular_recons_aug1,'y','LineWidth',3)
    legend('pred-aug1','pred-aug2','recons-aug1')
    for i = 1:size(new_cur_gt,1)
        cur_gt_ = new_cur_gt(i,:);
        [~,one_gt] = find(cur_gt_ == 1);        
        hold on        
        rectangle('Position',[one_gt(1), 0, (one_gt(end)-one_gt(1)), 1],'FaceColor',[0.6 0.9 0.6],'EdgeColor',[0.6 0.9 0.6])      
    end
    
    hold on
%     plot(1:nSam,frame_regular_aug1,'b',1:nSam,frame_regular_aug2,'r','LineWidth',3)
    plot(1:nSam,frame_regular_aug1,'b',1:nSam,frame_regular_aug2,'r',1:nSam,frame_regular_recons_aug1,'y','LineWidth',3)
%     legend('aug1','aug2')
%     legend('aug1')
    xlabel('Frame Number')
    ylabel('Regularity Score')
    set(hAxis,'Position',[84 301 1899 451])
    set(gcf,'color','w')
    
        
    print(['ped1_FrameError_CompareAug1Aug2_seq' num2str(numTestFolders)],'-djpeg')
    
end

