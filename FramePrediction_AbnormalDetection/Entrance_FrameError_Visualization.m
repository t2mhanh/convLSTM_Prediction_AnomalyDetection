% close all
clear all
% decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_entrance/';
decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_entrance/RemoveTimeStamp';
addpath('../')
addpath(genpath('../Persistence1D'))

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
    frameGt = gt{1,numTestFolders};    
        frame_error = h5read(fullfile(decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');   
        if numTestFolders == 6
            frame_error = frame_error(1:end-708);
        end
        nSam = size(frame_error,1);
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
        frame_gt = zeros(1,nSam);
        num_ab_event_seq = size(frameGt,2);
        for i = 1:num_ab_event_seq
            frame_gt(frameGt(1,i):frameGt(2,i)) = 1;
        end
        cur_gt = frame_gt(10:nSam+9); % [1x191]
        % chose a range to visualize
        % seq 5
        
        frame_error = frame_error(15000:end);
        cur_gt = cur_gt(15000:end);
        range = reshape(15000:length(frame_error)+15000-1,[],1);
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
    
    %%
    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error); 
    frame_regular = 1 - frame_error;
    
    close all    
    hAxis = figure(1);
%     plot(1:size(frame_error,1),x_bi_smooth,'b')
    plot(frame_regular,'b')
    hold on
    for i = 1:size(new_cur_gt,1)
        cur_gt_ = new_cur_gt(i,:);
        [~,one_gt] = find(cur_gt_ == 1);        
        hold on        
        rectangle('Position',[one_gt(1), 0, (one_gt(end)-one_gt(1)), 1],'FaceColor',[0.6 0.9 0.6],'EdgeColor',[0.6 0.9 0.6])      
    end
    
%     for i = 1:size(new_ab_mask,1)
%         ab_mask_ = new_ab_mask(i,:);
%         [~,one_ab] = find(ab_mask_ == 1);
%         hold on        
%         if length(one_ab) == 1
%             rectangle('Position',[one_ab(1), 0.05, (one_ab(end)-one_ab(1)+4), 0.05],'FaceColor','y')        
%         else
%             rectangle('Position',[one_ab(1), 0.05, (one_ab(end)-one_ab(1)), 0.05],'FaceColor','y')        
%         end
%     end
    hold on
%     plot(range(:),frame_regular(:),'b','LineWidth',3)
    plot(frame_regular,'b','LineWidth',3)
    xlabel('Frame Number')
    ylabel('Regularity Score')
    set(hAxis,'Position',[84 301 1899 451])
    set(gcf,'color','w')
end
seq = 4
load(fullfile('/usr/not-backed-up/1_DATABASE/Adam dataset/Entrance/TestSeq',['Test' num2str(seq) '.mat']))