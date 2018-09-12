close all
clear all
% decision_map_path = '/usr/not-backed-up/1_convlstm/Ped1_5framesPrediction_3/';
decision_map_path = '/usr/not-backed-up/1_convlstm/Ped1_prediction6/';
% decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_ped1/';
% decision_map_path = '/usr/not-backed-up/1_convlstm/Ped1_5framesPrediction_ped1finetune_145/';
addpath('../')
%gt
testSeqPath = '/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/';

load(fullfile(testSeqPath,'TestFrameGT_original.mat'))%,'FrameGt')   % FROM UCSD 
addpath(genpath('../Persistence1D'))
expand_range = 50;
num_ab_event = zeros(1,36);
cd_fa = [];
for min_thres = 0.1:0.05:0.5%0.95%0.1:0.05:1%0.3:0.05:1 0.35%
    for max_thres = 0.5:0.05:1 %0.95%
        num_ab_event_each_max = 10;
        GT = zeros(36,num_ab_event_each_max);
        CD = zeros(36,num_ab_event_each_max);
        FA = zeros(36,num_ab_event_each_max);
%         FA = [];
        for numTestFolders = 1:36
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
            % --------- expand each abnormal point to a region of 2*expand_range frames
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
            
            % abnormal mask for each event separately 
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
                        
                

            %--------------------------------------------------
            
            % % remove a detection that overlaps 2 GTs
            CD_local_new = zeros(size(CD_local));            
            for i = 1:size(CD_local_new,2)
                for j = 1:size(CD_local_new,1)
                    if CD_local(j,i) == 1
                        CD_local_new(j,i) = 1;
                        break
                    end
                end
            end
            
            
            CD_local = CD_local_new;
            %---------------------------------------
            for num_truth = 1 : size(new_cur_gt,1)        
                if sum(CD_local(num_truth,:)) > 0
                    CD(numTestFolders,num_truth) = 1;                    
                end
            end
        %     
%         %     % overlap > 0.5
%             CD_local = zeros(size(new_cur_gt,1),size(new_ab_mask,1));
%             for num_detect = 1:size(new_ab_mask,1)        
%                 for num_truth = 1 : size(new_cur_gt,1)
%         %             sum(and(new_ab_mask(num_detect,:),new_cur_gt(num_truth,:))) / sum(new_cur_gt(num_truth,:))
%                     if sum(and(new_ab_mask(num_detect,:),new_cur_gt(num_truth,:)))/sum(new_cur_gt(num_truth,:)) >= 0.5
%                         CD_local(num_truth,num_detect) = 1;
%                     end            
%                 end
%                 if (sum(CD_local(:,num_detect)) == 0)
%                     for num_truth = 1 : size(new_cur_gt,1)
%         %                 if sum(and(new_ab_mask(num_detect,:),new_cur_gt(num_truth,:)))/sum(new_cur_gt(num_truth,:)) == 0
%                         if sum(and(new_ab_mask(num_detect,:),new_cur_gt(num_truth,:)))/sum(new_cur_gt(num_truth,:)) < 0.5
%                             FA(numTestFolders,num_detect) = 1;
%                         end            
%                     end
%                 end
%             end
% %             for num_truth = 1 : size(new_cur_gt,1)        
% %                 if sum(CD_local(num_truth,:)) > 0
% %                     CD(numTestFolders,num_truth) = 1;
% %                 end
% %             end

        end
        sum(CD(:))
        sum(FA(:))
        cd_fa(end+1,:) = [min_thres max_thres sum(CD(:)) sum(FA(:))];
    end
end
sum(CD(:))
sum(FA(:))
cd_fa(end+1,:) = [min_thres max_thres sum(CD(:)) sum(FA(:))];


