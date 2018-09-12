close all
clear all
decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_avenue/';
addpath('../')
addpath(genpath('../Persistence1D'))
%gt
testSeqPath = '/usr/not-backed-up/1_DATABASE';
load(fullfile(testSeqPath,'gt_avenue.mat'))
expand_range = 50;
%% EVALUATION CD/FA
% % global max min
% % load('x_bi_smooth_min_max.mat','x_bi_smooth_min','x_bi_smooth_max')
% 
num_ab_events = zeros(1,21);
cd_fa = [];
% Set threshold for surviving features
for threshold = 0.1:0.01:0.9
    num_ab_event_each_max = 10;
    GT = zeros(21,num_ab_event_each_max);
    CD = zeros(21,num_ab_event_each_max);
    FA = zeros(36,num_ab_event_each_max);
    for numTestFolders = 1:21
        numTestFolders
    frameGt = gt{1,numTestFolders};    
    frame_error = h5read(fullfile(decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');   
    nSam = size(frame_error,1);
    frame_start = 9;
    num_ab_event_seq = size(frameGt,2);
    num_ab_events(numTestFolders) = num_ab_event_seq;
    new_cur_gt = zeros(num_ab_event_seq,nSam);
    
    frameGt = frameGt - frame_start + 1;
    
    for i = 1:num_ab_event_seq
        new_cur_gt(i,frameGt(1,i):frameGt(2,i)) = 1;
    end
   
    num_ab_event(numTestFolders) = size(new_cur_gt,1);

    
    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error); 
    frame_regular = 1 - frame_error;
%     frame_regular = frame_error;
    single_precision_data = single(frame_regular);

    % Run Persistence1D on the data
    [minIndices, maxIndices, persistence, globalMinIndex, globalMinValue] = run_persistence1d(single_precision_data); 
       

    % Use filter_features_by_persistence to filter the pairs
    persistent_features = filter_features_by_persistence(minIndices, maxIndices, persistence, threshold);
    minidx = [persistent_features(:,1) ; globalMinIndex];
    maxidx = persistent_features(:,2);
    minidx = sort(minidx);
    if minidx(1) == 1
        if frame_regular(minidx(1) > 0.1)
        minidx(1) = [];
        end
    end
    
    maxidx = sort(maxidx);
    
    
    
    
    
    ab_mask = zeros(1,nSam);
    % --------- expand each abnormal point to a region of 50 frames
    % use maxima to limit the window range
    % eliminate maxima between 2 minimas (in range of 2*expand_range frames)    
%     elimidx = [];
%     for i = 1:length(maxidx)
%        curidx = maxidx(i);
%        [l,~] = find(minidx < curidx);
%        [u,~] = find(minidx > curidx);
%        if (length(l) > 0 && length(u) > 0)
%            min_lower = minidx(max(l));
%            min_upper = minidx(min(u));
%            if min_upper - min_lower < 2 * expand_range
%                 elimidx(end+1) = i;
%            end
%        end
%     end
%     maxidx(elimidx) = [];
%     for i = 1:length(minidx)
%         cur_minidx = minidx(i);
%         [l,~] = find(maxidx < cur_minidx);
%         [u,~] = find(maxidx > cur_minidx);
%         if length(l) > 0 && length(u) > 0
%             lower_range = ceil((cur_minidx + maxidx(max(l)))/2);
%             upper_range = ceil((cur_minidx + maxidx(min(u)))/2);
%             ab_mask(max(max(minidx(i)-expand_range,1),lower_range):min(min(minidx(i)+expand_range,nSam),upper_range)) = 1;        
%         else if length(l) > 0 
%                 lower_range = ceil((cur_minidx + maxidx(max(l)))/2);                
%                 ab_mask(max(max(minidx(i)-expand_range,1),lower_range):min(minidx(i)+expand_range,nSam)) = 1;
%             else if length(u) > 0                 
%                 upper_range = ceil((cur_minidx + maxidx(min(u)))/2);
%                 ab_mask(max(minidx(i)-expand_range,1):min(min(minidx(i)+expand_range,nSam),upper_range)) = 1;
%                 else
%                     ab_mask(max(minidx(i)-expand_range,1):min(minidx(i)+expand_range,nSam)) = 1;
%                 end
%             end
%         end
%     end                          
    
    % don't use maximum to limit the window range
    for i = 1:length(minidx)
        ab_mask(max(minidx(i)-expand_range,1):min(minidx(i)+expand_range,nSam)) = 1;
    end
    

    %--------------------------------------------------------------------------------
    [~,c] = find(ab_mask == 1);
    gap = c(2:end) - c(1:end-1);
    [~,c_gap] = find(gap~= 1);
    if length(c_gap) == 0
        new_ab_mask = ab_mask;
    else
        new_ab_mask  = zeros(length(c_gap)+1,size(ab_mask,2));    
        new_ab_mask(1,c(1):c(c_gap(1))) = 1;
        for i = 1:length(c_gap)
            if i == length(c_gap)
                new_ab_mask(i+1,c(c_gap(i)+ 1)  : c(end)) = 1;
            else
                new_ab_mask(i+1,c(c_gap(i)+ 1 ) : c(c_gap(i+1))) = 1;
            end
        end
    end
    
    %------------------------------------------------------------------------------
    % CVPR paper (1)
            % overlap > 0.5 -> Correct Detection ; non-cverlap -> False
%             Alarm
            CD_local = zeros(size(new_cur_gt,1),size(new_ab_mask,1));
            for num_detect = 1:size(new_ab_mask,1)        
                for num_truth = 1 : size(new_cur_gt,1)
        %             sum(and(new_ab_mask(num_detect,:),new_cur_gt(num_truth,:))) / sum(new_cur_gt(num_truth,:))
                    if sum(and(new_ab_mask(num_detect,:),new_cur_gt(num_truth,:)))/min(sum(new_cur_gt(num_truth,:)),sum(new_ab_mask(num_detect,:))) >= 0.5
                        CD_local(num_truth,num_detect) = 1;
                    end            
                end
                if (sum(CD_local(:,num_detect)) == 0)
                    overlap = 0;
                    for num_truth = 1 : size(new_cur_gt,1)                
                        if sum(and(new_ab_mask(num_detect,:),new_cur_gt(num_truth,:)))/sum(new_cur_gt(num_truth,:)) > 0
                            overlap = overlap + 1;
                        end                 
                    end
                    if overlap == 0 % detected segment does not overlap with ground-truth
                        FA(numTestFolders,num_detect) = 1;
                    end
                end
            end
%             
            %-----(2)----------------------------------------
            % overlap > 0.5 -> CD, overlap < 0 -> FA 
%             CD_local = zeros(size(new_cur_gt,1),size(new_ab_mask,1));
%             for num_detect = 1:size(new_ab_mask,1)        
%                 for num_truth = 1 : size(new_cur_gt,1)
%         %             sum(and(new_ab_mask(num_detect,:),new_cur_gt(num_truth,:))) / sum(new_cur_gt(num_truth,:))
%                     if sum(and(new_ab_mask(num_detect,:),new_cur_gt(num_truth,:)))/min(sum(new_cur_gt(num_truth,:)),sum(new_ab_mask(num_detect,:))) >= 0.5
%                         CD_local(num_truth,num_detect) = 1;
%                     end            
%                 end
%                 if sum(CD_local(:,num_detect)) == 0
%                     FA(numTestFolders,num_detect) = 1;
%                 end
%             end
% %             for i = 1:size(new_ab_mask,1)
% %                 if sum(CD_local(:,i)) == 0
% %                     FA(numTestFolders,i) = 1;
% %                 end
% %             end
                
            
            % ------ (3) ------------------------
            % % remove a detection that overlaps 2 GTs
%             CD_local_new = zeros(size(CD_local));            
%             for i = 1:size(CD_local_new,2)
%                 for j = 1:size(CD_local_new,1)
%                     if CD_local(j,i) == 1
%                         CD_local_new(j,i) = 1;
%                         break
%                     end
%                 end
%             end
            
            
%             CD_local = CD_local_new;
            %---------------------------------------
            for num_truth = 1 : size(new_cur_gt,1)        
                if sum(CD_local(num_truth,:)) > 0
                    CD(numTestFolders,num_truth) = 1;                    
                end
            end

        %% OLD VERSION


    %             for num_detect = 1:size(new_ab_mask,1)     
    %                 correct = 0;
    %                 for num_truth = 1 : size(new_cur_gt,1)
    %         %             sum(and(new_ab_mask(num_detect,:),new_cur_gt(num_truth,:))) / sum(new_cur_gt(num_truth,:))
    %                     if sum(and(new_ab_mask(num_detect,:),new_cur_gt(num_truth,:)))/min(sum(new_cur_gt(num_truth,:)),sum(new_ab_mask(num_detect,:))) >= 0.5
    %                         CD(numTestFolders,num_truth) = 1;
    %                         correct = 1;
    %                         break                            
    %                     end            
    %                 end
    %                 if correct ~= 1
    %                     FA(numTestFolders,num_detect) = 1;
    %                 end
    %             end
    
%     sum(CD(:))
%     sum(FA(:))
%     cd_fa(end+1,:) = [threshold sum(CD(:)) sum(FA(:))];
    end
    cd_fa(end+1,:) = [threshold sum(CD(:)) sum(FA(:))];
end






