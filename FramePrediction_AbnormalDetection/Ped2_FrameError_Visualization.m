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

for numTestFolders = 1%1:12
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
%     im1 = imread('010.tif');
% im2 = imread('080.tif');
% im3 = imread('140.tif');
    
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
    plot(frame_regular,'b','LineWidth',3)
    xlabel('Frame Number')
    ylabel('Regularity Score')
%     set(hAxis,'Position',[84 301 1899 451])
%     set(gcf,'color','w')
% %     % seq1
%     axes('Position',[.05 .15 .3 .3])
%     imshow(im1)
%     axes('Position',[.35 .5 .3 .3])
%     imshow(im2)
%     axes('Position',[.65 .5 .3 .3])
%     imshow(im3)

%     print(['ped1_frameError_seq' num2str(numTestFolders)],'-djpeg')
%     pause
    %  --------------------------------------------   
    

end
