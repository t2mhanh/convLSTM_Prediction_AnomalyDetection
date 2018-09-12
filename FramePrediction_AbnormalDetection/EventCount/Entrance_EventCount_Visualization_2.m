% close all
clear all
decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_entrance/';
addpath('../')
addpath(genpath('../Persistence1D'))

%gt
testSeqPath = '/usr/not-backed-up/1_DATABASE';
%%%%% ground-truth from CVPR paper
% load(fullfile(testSeqPath,'gt1_enter.mat'))

%%%%% cvpr ground-truth with 65 anomalous segments
% load(fullfile(testSeqPath,'gt1_enter_new_H.mat'))
% gt = gt_new;

load(fullfile(testSeqPath,'Entrance_adam_gt.mat'))
 gt = gt_adam;
expand_range = 50;

%%
% num_ab_event_each_max = 10;
% CD = zeros(21,num_ab_event_each_max);
% FA = zeros(21,num_ab_event_each_max);

for numTestFolders = 1:6%[1 2 3 4 10 12 18]%
    num_ab_event_each_max = 10;
    CD = zeros(21,num_ab_event_each_max);
    FA = zeros(21,num_ab_event_each_max);
    numTestFolders
    frameGt = gt{1,numTestFolders};    
        frame_error = h5read(fullfile(decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');   
        if numTestFolders == 6
            frame_error = frame_error(1:end-708);
        end
        nSam = size(frame_error,1);
        frame_start = 10;%9;
        num_ab_event_seq = size(frameGt,2);
        num_ab_events(numTestFolders) = num_ab_event_seq;
        new_cur_gt = zeros(num_ab_event_seq,nSam);

        frameGt = frameGt - frame_start + 1;

        for i = 1:num_ab_event_seq
            new_cur_gt(i,max(1,frameGt(1,i)):min(frameGt(2,i),nSam)) = 1;
        end

    
    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error); 
    frame_regular = 1 - frame_error;
%     frame_regular = frame_error;
    single_precision_data = single(frame_regular);

    % Run Persistence1D on the data
    [minIndices, maxIndices, persistence, globalMinIndex, globalMinValue] = run_persistence1d(single_precision_data); 

    % Set threshold for surviving features
    threshold = 0.16;%0.99999;

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
    % CVPR paper
            % overlap > 0.5 -> Correct Detection ; non-cverlap -> False
            % Alarm
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
            
            %---------------------------------------------
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
%                 

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
%             
%             
%             CD_local = CD_local_new;
            %---------------------------------------
            for num_truth = 1 : size(new_cur_gt,1)        
                if sum(CD_local(num_truth,:)) > 0
                    CD(numTestFolders,num_truth) = 1;                    
                end
            end
            
    
            
            
    figure(1)
    plot(frame_regular,'-b','LineWidth',3)
    xlabel('Frame Number')
    ylabel('Regularity Score') 
    hold on
    for i = 1:size(new_cur_gt,1)
        cur_gt_ = new_cur_gt(i,:);
        [~,one_gt] = find(cur_gt_ == 1);        
        hold on        
        rectangle('Position',[one_gt(1), 0, (one_gt(end)-one_gt(1)), 1],'FaceColor',[0.6 0.9 0.6],'EdgeColor',[0.6 0.9 0.6])      
    end
    
    for i = 1:size(new_ab_mask,1)
        ab_mask_ = new_ab_mask(i,:);
        [~,one_ab] = find(ab_mask_ == 1);
        hold on        
        if length(one_ab) == 1
            rectangle('Position',[one_ab(1), 0.05, (one_ab(end)-one_ab(1)+4), 0.05],'FaceColor','y')        
        else
            rectangle('Position',[one_ab(1), 0.05, (one_ab(end)-one_ab(1)), 0.05],'FaceColor','y')        
        end
    end
    
    hold on;
    plot(frame_regular,'-b','LineWidth',3)
    % Add a scatter plot for the filtered minima
    plot(minidx, frame_regular(minidx), 'o', 'MarkerSize', 9, 'MarkerFaceColor', [0.3 0.3 1], 'MarkerEdgeColor', [0 0 1]);

    % Add a scatter plot for the filtered maxima
    plot(maxidx, frame_regular(maxidx), 'o', 'MarkerSize', 9, 'MarkerFaceColor', [1 0.2 0.2], 'MarkerEdgeColor', [1 0 0]);

    hold off;                        
    sum(CD(:))
    sum(FA(:))   
    pause
end
% sum(CD(:))
% sum(FA(:))
% % cd_fa(end+1,:) = [min_thres max_thres sum(CD(:)) sum(FA(:))];

%% read rgb frame
% seq = 7;
% % v = VideoReader(['/usr/not-backed-up/1_DATABASE/Avenue Dataset/testing_videos/' num2str(seq) '.avi']);
% % vol = [];
% % id = 0;
% % while hasFrame(v)
% %     video = readFrame(v);
% %     id = id + 1;
% %     vol(:,:,:,id) = video;
% % end
% if seq < 10
%     load(['/usr/not-backed-up/1_DATABASE/Avenue Dataset/testing_vol/vol0' num2str(seq) '.mat']);
% else
%     load(['/usr/not-backed-up/1_DATABASE/Avenue Dataset/testing_vol/vol' num2str(seq) '.mat']);
% end
% load(['/usr/not-backed-up/1_DATABASE/Avenue Dataset/testing_label_mask/' num2str(seq) '_label.mat']);
% frame_error = h5read(fullfile(decision_map_path,['test_' num2str(seq) '_error.h5']),'/frame_error');   
% frame_error = (frame_error - min(frame_error));
% frame_error = frame_error / max(frame_error);
% frame_error = 1 - frame_error;
% figure
% % for id = 310:length(frame_error)% seq 14
% % for id = 500:length(frame_error) % seq 10, 12
% % for id = 100:length(frame_error) % seq 9
% % for id = 1%:length(frame_error) % seq 8    
% for id = 200:length(frame_error) % seq 7
%     id    
%     subplot(2,1,1)
% %     imshow(vol(:,:,:,400)/255) % from avi data
%     imshow(vol(:,:,id))
%     a = volLabel{id};
%     a = imresize(a,size(vol(:,:,id)),'nearest');
%     [i,j] = find(a~=0);
%     if length(i) ~= 0
%         i_min = min(i);
%         i_max = max(i);
%         j_min = min(j);
%         j_max = max(j);
% %         subplot(2,1,1)
% %         imshow(vol(:,:,id))
%         hold on        
%         rectangle('Position',[j_min i_min (j_max - j_min) (i_max - i_min)],'EdgeColor','r')
%     else
%         imshow(vol(:,:,id))
%     end
%     subplot(2,1,2)
%     plot(1:length(frame_error(1:id-8)),frame_error(1:id-8),'b','LineWidth',1.5)
%     xlim([0 length(frame_error)])
%     ylim([0 1])
%     pause(0.5)
% end