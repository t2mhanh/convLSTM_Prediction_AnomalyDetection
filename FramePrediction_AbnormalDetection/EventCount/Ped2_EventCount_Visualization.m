% close all
clear all
decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction16_ped2/';
% decision_map_path = '/usr/not-backed-up/1_convlstm/Ped2_5framesPrediction_ucsdAvenue_146/';
addpath('../')
%gt
testSeqPath = '/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/';

load(fullfile(testSeqPath,'TestFrameGT_original.mat'))%,'FrameGt')   % FROM UCSD 
addpath(genpath('../Persistence1D'))
expand_range = 25;
num_ab_event = zeros(1,12);

%%
min_thres = 0.1;
max_thres = 0.9;
num_ab_event_each_max = 2;
GT = zeros(12,num_ab_event_each_max);
CD = zeros(12,num_ab_event_each_max);
FA = [];
for numTestFolders = 1:12
    numTestFolders
    frameGt = FrameGt{1,numTestFolders};        
    frame_error = h5read(fullfile(decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');   
    nSam = size(frame_error,1);
    cur_gt = frameGt(9:nSam+8); % [1x191]
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
    num_ab_event(numTestFolders) = size(new_cur_gt,1);

    
%     frame_error = (frame_error - min(frame_error));
%     frame_error = frame_error / max(frame_error); 
%     frame_regular = 1 - frame_error;
    frame_regular = frame_error;
    single_precision_data = single(frame_regular);

    % Run Persistence1D on the data
    [minIndices maxIndices persistence globalMinIndex globalMinValue] = run_persistence1d(single_precision_data); 

    % Set threshold for surviving features
    threshold = 0.9999;

    % Filter Persistence1D paired extrema for relevant features
    pairs = filter_features_by_persistence(minIndices, maxIndices, persistence, threshold);

    % interpolated minima and maxima
    mins = get_min_indices(pairs);
    maxs = get_max_indices(pairs);

    % Set the data weight. Choosing 0.0 constructs smoother function
    data_weight = 0.0000001;%0;

    % Set the smoothness for the results. 
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
    %   --------- smooth the error ----------------
%     nfr_sm = 5; % 5 give the best results
%     nSam = size(x_bi_smooth,1);
%     frame_error_sm = zeros(nSam-nfr_sm,1);
%     for fr = 1:nSam-nfr_sm
%         frame_error_sm(fr) = mean(x_bi_smooth(fr:fr+nfr_sm));
%     end
%     x_bi_smooth = frame_error_sm;
    %  --------------------------------------------
    %  --------------------------------------------


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
    
    ab_mask = zeros(size(cur_gt));
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

    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error);   

     close all    
    figure(1)
%     plot(1:size(frame_error,1),x_bi_smooth,'b')
    plot(1:size(x_bi_smooth,1),x_bi_smooth,'b')
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
    hold on
    plot(1:size(frame_error,1),x_bi_smooth,'b','LineWidth',3)
    xlabel('Frame Number')
    ylabel('Regularity Score')
    pause
end
sum(CD(:))
sum(FA(:))