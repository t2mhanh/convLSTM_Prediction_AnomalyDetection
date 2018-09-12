close all
clear all

% aug1
% aug1_decision_map_path =  '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_exit/';
aug1_decision_map_path =  '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_exit/RemoveTimeStamp';

% aug2
aug2_decision_map_path = '/usr/not-backed-up/1_convlstm/prediction6_aug2_exit/RemoveTimeStamp';
addpath('../')

%gt
testSeqPath = '/usr/not-backed-up/1_DATABASE';
% load(fullfile(testSeqPath,'gt1_exit.mat'))
load(fullfile(testSeqPath,'gt1_exit_new_H.mat'))
gt = gt_new;
% MPPCA ground-truth
% load(fullfile(testSeqPath,'Exit_adam.mat'))
%

for numTestFolders = 4%1:4%
    numTestFolders
    frameGt = gt{1,numTestFolders};    
    frame_error = h5read(fullfile(aug1_decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');       
    nSam = size(frame_error,1);
    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error); 
    frame_regular_aug1 = 1 - frame_error;
    
    % aug2
    frame_error = h5read(fullfile(aug2_decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');       
    nSam = size(frame_error,1);
    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error); 
    frame_regular_aug2 = 1 - frame_error;
    
    % gt
    frame_start = 10;
    num_ab_event_seq = size(frameGt,2);    
    new_cur_gt = zeros(num_ab_event_seq,nSam);
    
    frameGt = frameGt - frame_start + 1;
    
    for i = 1:num_ab_event_seq
        new_cur_gt(i,max(1,frameGt(1,i)):min(frameGt(2,i),nSam)) = 1;
    end
    
    close all    
    hAxis = figure(1);

%     plot(frame_regular_aug1,'b')
    plot(1:length(frame_regular_aug1),frame_regular_aug1,'b',1:length(frame_regular_aug1),frame_regular_aug2,'r','LineWidth',3)
    legend('pred-aug1','pred-aug2','Location','southeast')
    xlim([0 nSam])
    hold on
    for i = 1:size(new_cur_gt,1)
        cur_gt_ = new_cur_gt(i,:);
        [~,one_gt] = find(cur_gt_ == 1);        
        hold on        
        rectangle('Position',[one_gt(1), 0, (one_gt(end)-one_gt(1)), 1],'FaceColor',[0.6 0.9 0.6],'EdgeColor',[0.6 0.9 0.6])      
    end
    

    hold on
%     plot(frame_regular_aug1,'b','LineWidth',3)
    plot(1:length(frame_regular_aug1),frame_regular_aug1,'b',1:length(frame_regular_aug1),frame_regular_aug2,'r','LineWidth',3)
    xlabel('Frame Number')
    ylabel('Regularity Score')
    set(hAxis,'Position',[84 301 1899 451])
    set(gcf,'color','w')
    
    print(['exit_FrameError_CompareAug1Aug2_seq' num2str(numTestFolders)],'-djpeg')
    
end
% seq = 4
% load(fullfile('/usr/not-backed-up/1_DATABASE/Adam dataset/Exit/TestSeq',['Test' num2str(seq) '.mat']))


