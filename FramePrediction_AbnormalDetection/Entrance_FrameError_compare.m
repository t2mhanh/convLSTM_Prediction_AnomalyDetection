close all
clear all

% aug1
% aug1_decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_entrance/';
aug1_decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_entrance/RemoveTimeStamp';

% aug2
aug2_decision_map_path = '/usr/not-backed-up/1_convlstm/prediction6_aug2_entrance/RemoveTimeStamp';
addpath('../')

%gt
testSeqPath = '/usr/not-backed-up/1_DATABASE';
%%%%% ground-truth from CVPR paper
% load(fullfile(testSeqPath,'gt1_enter.mat'))

%%%%% cvpr ground-truth with 65 anomalous segments
load(fullfile(testSeqPath,'gt1_enter_new_H.mat'))
gt = gt_new;

% load(fullfile(testSeqPath,'Entrance_adam_gt.mat'))
%  gt = gt_adam;
for numTestFolders = 5%1:6%[1 2 3 4 10 12 18]%
    
    numTestFolders   
    %% aug1
    frame_error = h5read(fullfile(aug1_decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');   
    if numTestFolders == 6
        frame_error = frame_error(1:end-708);
    end
    nSam = size(frame_error,1);

    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error); 
    frame_regular_aug1 = 1 - frame_error;
    % chose a range to visualize
    % seq 5
    frame_start = 15000;
    frame_regular_aug1 = frame_regular_aug1(frame_start:end);

    %% aug2
    
    frame_error = h5read(fullfile(aug2_decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');   
    if numTestFolders == 6
        frame_error = frame_error(1:end-708);
    end
    
    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error); 
    frame_regular_aug2 = 1 - frame_error;
    % chose a range to visualize
    % seq 5
    frame_start = 15000;
    frame_regular_aug2 = frame_regular_aug2(frame_start:end);

    %%
%         % 1st way to process groundtruth -> can't visualize a part of
%         % 20,000 frames (let say can't visualize 1,000 frames)
%         frame_start = 10;%9;
%         num_ab_event_seq = size(frameGt,2);
%         
%         new_cur_gt = zeros(num_ab_event_seq,nSam);
% 
%         frameGt = frameGt - frame_start + 1;
% 
%         for i = 1:num_ab_event_seq
%             new_cur_gt(i,max(1,frameGt(1,i)):min(frameGt(2,i),nSam)) = 1;
%         end


    %%
    % second way to process ground-truth
    % can chose smaller range to visualize
    frameGt = gt{1,numTestFolders};    
    frame_gt = zeros(1,nSam);
    num_ab_event_seq = size(frameGt,2);
    for i = 1:num_ab_event_seq
        frame_gt(frameGt(1,i):frameGt(2,i)) = 1;
    end
    cur_gt = frame_gt(10:nSam+9); 
    cur_gt = cur_gt(frame_start:end);
    %
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
    plot(1:length(frame_regular_aug1),frame_regular_aug1,'b',1:length(frame_regular_aug1),frame_regular_aug2,'r','LineWidth',3)
    legend('pred-aug1','pred-aug2','Location','southeast')
    hold on
    for i = 1:size(new_cur_gt,1)
        cur_gt_ = new_cur_gt(i,:);
        [~,one_gt] = find(cur_gt_ == 1);        
        hold on        
        rectangle('Position',[one_gt(1), 0, (one_gt(end)-one_gt(1)), 1],'FaceColor',[0.6 0.9 0.6],'EdgeColor',[0.6 0.9 0.6])      
    end
    hold on
    plot(1:length(frame_regular_aug1),frame_regular_aug1,'b',1:length(frame_regular_aug1),frame_regular_aug2,'r','LineWidth',3)
    xlabel('Frame Number')
    ylabel('Regularity Score')    
    set(hAxis,'Position',[84 301 1899 451])
    set(gcf,'color','w')
    print(['entrance_FrameError_CompareAug1Aug2_seq' num2str(numTestFolders) 'fr' num2str(frame_start) '_' num2str(length(frame_regular_aug1)+frame_start)],'-djpeg')
end
% seq = 4
% load(fullfile('/usr/not-backed-up/1_DATABASE/Adam dataset/Entrance/TestSeq',['Test' num2str(seq) '.mat']))