close all
clear all
% decision_map_path = '/usr/not-backed-up/1_convlstm/entrance_5frames/';
decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_entrance/';
addpath('../')
addpath(genpath('../Persistence1D'))

expand_range = 50;
%gt
testSeqPath = '/usr/not-backed-up/1_DATABASE';
%%%%% ground-truth from CVPR paper
% load(fullfile(testSeqPath,'gt1_enter.mat'))

%%%%% cvpr ground-truth with 65 anomalous segments
load(fullfile(testSeqPath,'gt1_enter_new_H.mat'))
gt = gt_new;

%%%%%  ground-truth from MPPCA paper
%  load(fullfile(testSeqPath,'Entrance_adam_gt.mat'))
%  gt = gt_adam;
%% each segment is a event ???????????
% gt_new = cell(1,6);
% for i = 1:6
%     id = 1;    
%     cur_gt = gt{1,i};
%     new_gt = cur_gt(:,1);
%     for j = 2:size(cur_gt,2)
%         if new_gt(2,id) == cur_gt(1,j);
%             new_gt(2,id) = cur_gt(2,j);
%         else
%             id = id + 1;
%             new_gt(:,id) = cur_gt(:,j);
%         end
%     end
%     gt_new{i} = new_gt;
% end
% save(fullfile(testSeqPath,'gt1_enter_new_H.mat'),'gt_new','-v7.3')
% a = 0;
% for i = 1:6
%     a = a + size(gt_new{1,i},2);
% end
% a % 65 segments
% % cvpr ground-truth with 65 anomalous segments
% load(fullfile(testSeqPath,'gt1_enter_new_H.mat'))
% gt = gt_new;


%% COUNT THE EVENT AND DRAW REGULARITY SCORE
min_thres = 0.62;%0.67;
max_thres = 0.9;
num_ab_event_each_max = 10;
CD = zeros(21,num_ab_event_each_max);
FA = zeros(21,num_ab_event_each_max);

for numTestFolders = 6%1:6%
    numTestFolders
    frameGt = gt{1,numTestFolders};    
    frame_error = h5read(fullfile(decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');   
    if numTestFolders == 6
        frame_error = frame_error(1:end-708);
    end
    nSam = size(frame_error,1);
    frame_start = 9;%9;
    num_ab_event_seq = size(frameGt,2);
    num_ab_events(numTestFolders) = num_ab_event_seq;
    new_cur_gt = zeros(num_ab_event_seq,nSam);
    
    frameGt = frameGt - frame_start + 1;
    
    for i = 1:num_ab_event_seq
        new_cur_gt(i,max(1,frameGt(1,i)):min(frameGt(2,i),nSam)) = 1;
    end
   
    num_ab_event(numTestFolders) = size(new_cur_gt,1);

    
%     frame_error = (frame_error - min(frame_error));
%     frame_error = frame_error / max(frame_error); 
%     frame_regular = 1 - frame_error;
    frame_regular = frame_error;
    single_precision_data = single(frame_regular);

    % Run Persistence1D on the data
    [minIndices maxIndices persistence globalMinIndex globalMinValue] = run_persistence1d(single_precision_data); 

    % Set threshold for surviving features
    threshold = 0.1;%0.99999;

    % Filter Persistence1D paired extrema for relevant features
    pairs = filter_features_by_persistence(minIndices, maxIndices, persistence, threshold);

    % interpolated minima and maxima
    mins = get_min_indices(pairs);
    maxs = get_max_indices(pairs);

    % Set the data weight. Choosing 0.0 constructs smoother function
    data_weight = 0;

%     % Set the smoothness for the results. 
%     bi_smoothness = 'biharmonic';
% 
%     % Call reconstruct1d_with_persistence_res. Calling this function directly avoids
%     % re-running Persistence1D for different reconstruction parameters
%     x_bi_smooth  = reconstruct1d_with_persistence_res( frame_regular, ...
%                                                         mins, ...
%                                                         maxs, ...
%                                                         globalMinIndex, ...
%                                                         bi_smoothness, ...
%                                                         data_weight);
    tri_smoothness = 'triharmonic';		
    x_tri_smooth  = reconstruct1d_with_persistence_res( frame_regular, ...
													mins, ...
													maxs, ...
													globalMinIndex, ...
													tri_smoothness, ...
													data_weight);
    x_bi_smooth = x_tri_smooth;         

    x_bi_smooth = (x_bi_smooth - min(x_bi_smooth));
    x_bi_smooth = x_bi_smooth / max(x_bi_smooth);        
    x_bi_smooth = 1 - x_bi_smooth;    
    [maxima,maxidx] = findpeaks(x_bi_smooth);
    [minima,minidx] = findpeaks(1 - x_bi_smooth);
    minima = 1 - minima;
    
    % find distinct maxima
    [maxidx_,~] = find(maxima >= max_thres);
    maxidx = maxidx(maxidx_);
    % distinct minima   
    [minidx_,~] = find(minima <= min_thres);
    minidx = minidx(minidx_);
    % eliminate maxima between 2 minimas (in range of 2*expand_range frames)
    maxidx_new = maxidx;
    elimidx = [];
    for i = 1:length(maxidx)
       curidx = maxidx(i);
       [l,~] = find(minidx < curidx);
       [u,~] = find(minidx > curidx);
       if (length(l) > 0 && length(u) > 1)
           min_lower = minidx(max(l));
           min_upper = minidx(min(u));
           if min_upper - min_lower < 2 * expand_range
                elimidx(end+1) = i;
           end
       end
    end
    maxidx(elimidx) = [];
    maxima(elimidx) = [];
    
    ab_mask = zeros(1,nSam);
    % --------- expand each abnormal point to a region of 50 frames
    for i = 1:length(minidx)
        cur_minidx = minidx(i);
        [l,~] = find(maxidx < cur_minidx);
        [u,~] = find(maxidx > cur_minidx);
        if length(l) > 0 && length(u) > 0
            lower_range = ceil((cur_minidx + maxidx(max(l)))/2);
            upper_range = ceil((cur_minidx + maxidx(min(u)))/2);
            ab_mask(max(max(minidx(i)-expand_range,1),lower_range):min(min(minidx(i)+expand_range,nSam),upper_range)) = 1;        
        else if length(l) > 0 
                lower_range = ceil((cur_minidx + maxidx(max(l)))/2);                
                ab_mask(max(max(minidx(i)-expand_range,1),lower_range):min(minidx(i)+expand_range,nSam)) = 1;
            else if length(u) > 0                 
                upper_range = ceil((cur_minidx + maxidx(min(u)))/2);
                ab_mask(max(minidx(i)-expand_range,1):min(min(minidx(i)+expand_range,nSam),upper_range)) = 1;
                else
                    ab_mask(max(minidx(i)-expand_range,1):min(minidx(i)+expand_range,nSam)) = 1;
                end
            end
        end
    end                          
    
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
%     overlap > 0.5
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
    
    for num_truth = 1 : size(new_cur_gt,1)        
        if sum(CD_local(num_truth,:)) > 0
            CD(numTestFolders,num_truth) = 1;
        end
    end
%     
%     % IoU > 0.5
% %     for num_detect = 1:size(new_ab_mask,1)
% %         for num_truth = 1 : size(new_cur_gt,1)
% %             [~,a] = find(new_ab_mask(num_detect,:) == 1);
% %             [~,b] = find(new_cur_gt(num_truth,:) == 1);
% %             if length(intersect(a,b))/length(union(a,b)) >= 0.5
% %                 CD(numTestFolders,num_truth) = 1;
% %             end            
% %         end
% %         if (sum(CD(numTestFolders,:)) == 0)
% %             for num_truth = 1 : size(new_cur_gt,1)
% %                 if sum(and(new_ab_mask(num_detect,:),new_cur_gt(num_truth,:)))/sum(new_cur_gt(num_truth,:)) == 0
% %                     FA(numTestFolders,num_detect) = 1;
% %                 end            
% %             end
% %         end
% %     end
%     % overlap > 0.5, one normal frame is detected as anomaly -> one FA
% %     for num_detect = 1:size(new_ab_mask,1)
% %         for num_truth = 1 : size(new_cur_gt,1)
% %             if sum(and(new_ab_mask(num_detect,:),new_cur_gt(num_truth,:)))/sum(new_cur_gt(num_truth,:)) > 0 %= 0.5
% %                 CD(numTestFolders,num_truth) = 1;
% %             end            
% %         end
% %         
% %         for num_truth = 1 : size(new_cur_gt,1)
% %             if sum(and(new_ab_mask(num_detect,:),not(new_cur_gt(num_truth,:)))) > 0
% %                 FA(numTestFolders,num_detect) = 1;
% %             end            
% %         end
% %         
% %     end
%     
% --------------------------------------------------------------------------
%                           DRAW REGULARITY SCORE
% --------------------------------------------------------------------------

% %     
    close all
    figure(1)
%     plot(1:size(frame_error,1),(1-frame_error),'r',1:size(frame_error,1),x_bi_smooth,'b')
    plot(1:size(frame_error,1),x_bi_smooth,'b','LineWidth',2.5)
    hold on
    for i = 1:size(new_cur_gt,1)
        cur_gt_ = new_cur_gt(i,:);
        [~,one_gt] = find(cur_gt_ == 1);        
        hold on        
        rectangle('Position',[one_gt(1), 0, (one_gt(end)-one_gt(1)), 1],'FaceColor',[0.6 0.9 0.6],'EdgeColor',[0.6 0.9 0.6])      
    end
%-------- show detections -------------------------------
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
%     hold on
%     plot(1:size(frame_error,1),(1-frame_error),'r',1:size(frame_error,1),x_bi_smooth,'b')
    plot(1:size(frame_error,1),x_bi_smooth,'b','LineWidth',2.5)
    % show maxima
%     for id = 1:length(maxidx)
%         hold on
%         plot(maxidx,x_bi_smooth(maxidx),'or')
%     end
% ------- show minima ------------------------------------  
%     for id = 1:length(minidx)
%         hold on
%         plot(minidx,x_bi_smooth(minidx),'xm')
%     end
    xlabel('Frame Number')
    ylabel('Regularity Score')
end
% save('x_bi_smooth_min_max.mat','x_bi_smooth_min','x_bi_smooth_max')
sum(CD(:))
sum(FA(:))



%% EVALUATE WITH DIFFERENT MIN AND MAX THRESHOLD 

% %% EVALUATION CD/FA
% % % global max min
% % % load('x_bi_smooth_min_max.mat','x_bi_smooth_min','x_bi_smooth_max')
% % 
% % num_ab_events = zeros(1,21);
% % cd_fa = [];
% % for min_thres = 0.3:0.05:1
% %     for max_thres = 0.5:0.05:1;
% %         num_ab_event_each_max = 10;
% %         GT = zeros(21,num_ab_event_each_max);
% %         CD = zeros(21,num_ab_event_each_max);
% %         FA = [];
% %         for numTestFolders = 1%:21
% %             numTestFolders
% %             frameGt = gt{1,numTestFolders};    
% %             frame_error = h5read(fullfile(decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');   
% %             nSam = size(frame_error,1);
% %             frame_start = 9;
% %             num_ab_event_seq = size(frameGt,2);
% %             num_ab_events(numTestFolders) = num_ab_event_seq;
% %             new_cur_gt = zeros(num_ab_event_seq,nSam);
% % 
% %             frameGt = frameGt - frame_start + 1;
% % 
% %             for i = 1:num_ab_event_seq
% %                 new_cur_gt(i,max(1,frameGt(1,i)):frameGt(2,i)) = 1;
% %             end
% % 
% %             num_ab_event(numTestFolders) = size(new_cur_gt,1);
% % 
% % 
% %         %     frame_error = (frame_error - min(frame_error));
% %         %     frame_error = frame_error / max(frame_error); 
% %         %     frame_regular = 1 - frame_error;
% %             frame_regular = frame_error;
% %             single_precision_data = single(frame_regular);
% % 
% %             % Run Persistence1D on the data
% %             [minIndices maxIndices persistence globalMinIndex globalMinValue] = run_persistence1d(single_precision_data); 
% % 
% %             % Set threshold for surviving features
% %             threshold = 0.99999;
% % 
% %             % Filter Persistence1D paired extrema for relevant features
% %             pairs = filter_features_by_persistence(minIndices, maxIndices, persistence, threshold);
% % 
% %             % interpolated minima and maxima
% %             mins = get_min_indices(pairs);
% %             maxs = get_max_indices(pairs);
% % 
% %             % Set the data weight. Choosing 0.0 constructs smoother function
% %             data_weight = 0;
% % 
% %         %     % Set the smoothness for the results. 
% %         %     bi_smoothness = 'biharmonic';
% %         % 
% %         %     % Call reconstruct1d_with_persistence_res. Calling this function directly avoids
% %         %     % re-running Persistence1D for different reconstruction parameters
% %         %     x_bi_smooth  = reconstruct1d_with_persistence_res( frame_regular, ...
% %         %                                                         mins, ...
% %         %                                                         maxs, ...
% %         %                                                         globalMinIndex, ...
% %         %                                                         bi_smoothness, ...
% %         %                                                         data_weight);
% %             tri_smoothness = 'triharmonic';		
% %             x_tri_smooth  = reconstruct1d_with_persistence_res( frame_regular, ...
% %                                                             mins, ...
% %                                                             maxs, ...
% %                                                             globalMinIndex, ...
% %                                                             tri_smoothness, ...
% %                                                             data_weight);
% %             x_bi_smooth = x_tri_smooth;
% %             %   --------- smooth the error ----------------
% %         %     nfr_sm = 5; % 5 give the best results
% %         %     nSam = size(x_bi_smooth,1);
% %         %     frame_error_sm = zeros(nSam-nfr_sm,1);
% %         %     for fr = 1:nSam-nfr_sm
% %         %         frame_error_sm(fr) = mean(x_bi_smooth(fr:fr+nfr_sm));
% %         %     end
% %         %     x_bi_smooth = frame_error_sm;
% %             %  --------------------------------------------
% %             %  --------------------------------------------
% %         %     x_bi_smooth_min(numTestFolders) = min(x_bi_smooth);
% %         %     x_bi_smooth_max(numTestFolders) = max(x_bi_smooth);
% %         %     x_bi_smooth = (x_bi_smooth - min(x_bi_smooth_min));
% %         %     x_bi_smooth = x_bi_smooth / max(x_bi_smooth_max);        
% % 
% %             x_bi_smooth = (x_bi_smooth - min(x_bi_smooth));
% %             x_bi_smooth = x_bi_smooth / max(x_bi_smooth);        
% %             x_bi_smooth = 1 - x_bi_smooth;    
% %             [maxima,maxidx] = findpeaks(x_bi_smooth);
% %             [minima,minidx] = findpeaks(1 - x_bi_smooth);
% %             minima = 1 - minima;
% % 
% %             % find distinct maxima
% %             [maxidx_,~] = find(maxima >= max_thres);
% %             maxidx = maxidx(maxidx_);
% %             % distinct minima   
% %             [minidx_,~] = find(minima <= min_thres);
% %             minidx = minidx(minidx_);
% %             % eliminate maxima between 2 minimas (in range of 2*expand_range frames)
% %             maxidx_new = maxidx;
% %             elimidx = [];
% %             for i = 1:length(maxidx)
% %                curidx = maxidx(i);
% %                [l,~] = find(minidx < curidx);
% %                [u,~] = find(minidx > curidx);
% %                if (length(l) > 0 && length(u) > 1)
% %                    min_lower = minidx(max(l));
% %                    min_upper = minidx(min(u));
% %                    if min_upper - min_lower < 2 * expand_range
% %                         elimidx(end+1) = i;
% %                    end
% %                end
% %             end
% %             maxidx(elimidx) = [];
% %             maxima(elimidx) = [];
% % 
% %             ab_mask = zeros(1,nSam);
% %             % --------- expand each abnormal point to a region of 50 frames
% %             for i = 1:length(minidx)
% %                 cur_minidx = minidx(i);
% %                 [l,~] = find(maxidx < cur_minidx);
% %                 [u,~] = find(maxidx > cur_minidx);
% %                 if length(l) > 0 && length(u) > 0
% %                     lower_range = ceil((cur_minidx + maxidx(max(l)))/2);
% %                     upper_range = ceil((cur_minidx + maxidx(min(u)))/2);
% %                     ab_mask(max(max(minidx(i)-expand_range,1),lower_range):min(min(minidx(i)+expand_range,nSam),upper_range)) = 1;        
% %                 else if length(l) > 0 
% %                         lower_range = ceil((cur_minidx + maxidx(max(l)))/2);                
% %                         ab_mask(max(max(minidx(i)-expand_range,1),lower_range):min(minidx(i)+expand_range,nSam)) = 1;
% %                     else if length(u) > 0                 
% %                         upper_range = ceil((cur_minidx + maxidx(min(u)))/2);
% %                         ab_mask(max(minidx(i)-expand_range,1):min(min(minidx(i)+expand_range,nSam),upper_range)) = 1;
% %                         else
% %                             ab_mask(max(minidx(i)-expand_range,1):min(minidx(i)+expand_range,nSam)) = 1;
% %                         end
% %                     end
% %                 end
% %             end                          
% % 
% %             [~,c] = find(ab_mask == 1);
% %             gap = c(2:end) - c(1:end-1);
% %             [~,c_gap] = find(gap~= 1);
% %             if length(c_gap) == 0
% %                 new_ab_mask = ab_mask;
% %             else
% %                 new_ab_mask  = zeros(length(c_gap)+1,size(ab_mask,2));    
% %                 new_ab_mask(1,c(1):c(c_gap(1))) = 1;
% %                 for i = 1:length(c_gap)
% %                     if i == length(c_gap)
% %                         new_ab_mask(i+1,c(c_gap(i)+ 1)  : c(end)) = 1;
% %                     else
% %                         new_ab_mask(i+1,c(c_gap(i)+ 1 ) : c(c_gap(i+1))) = 1;
% %                     end
% %                 end
% %             end
% %             for num_detect = 1:size(new_ab_mask,1)     
% %                     correct = 0;
% %                     for num_truth = 1 : size(new_cur_gt,1)
% %             %             sum(and(new_ab_mask(num_detect,:),new_cur_gt(num_truth,:))) / sum(new_cur_gt(num_truth,:))
% %                         if sum(and(new_ab_mask(num_detect,:),new_cur_gt(num_truth,:)))/min(sum(new_cur_gt(num_truth,:)),sum(new_ab_mask(num_detect,:))) >= 0.5
% %                             CD(numTestFolders,num_truth) = 1;
% %                             correct = 1;
% %                             break                            
% %                         end            
% %                     end
% %                     if correct ~= 1
% %                         FA(numTestFolders,num_detect) = 1;
% %                     end
% %                 end
% %         end
% %         sum(CD(:))
% %         sum(FA(:))
% %         cd_fa(end+1,:) = [min_thres max_thres sum(CD(:)) sum(FA(:))];
% %     end
% % end
% % sum(CD(:))
% % sum(FA(:))
% % cd_fa(end+1,:) = [min_thres max_thres sum(CD(:)) sum(FA(:))];
% 
% 
% 
% 


%% read rgb frame
% load('/usr/not-backed-up/1_DATABASE/Adam dataset/Entrance/TestSeq/Test1.mat')
