close all
clear all

% aug1
aug1_decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_avenue/';
% aug2
aug2_decision_map_path = '/usr/not-backed-up/1_convlstm/avenue_prediction_aug2/';
% recons-aug1
recons_aug1 = '/usr/not-backed-up/1_convlstm/avenue_AE6/';
decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_avenue/';
addpath('../')

%gt
testSeqPath = '/usr/not-backed-up/1_DATABASE';
load(fullfile(testSeqPath,'gt_avenue.mat'))


for numTestFolders = [5,7,15,12]
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
    frameGt = gt{1,numTestFolders};    
    frame_error = h5read(fullfile(decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');   
    nSam = size(frame_error,1);
    frame_start = 10;
    num_ab_event_seq = size(frameGt,2);
    
    new_cur_gt = zeros(num_ab_event_seq,nSam);
    
    frameGt = frameGt - frame_start + 1;
    
    for i = 1:num_ab_event_seq
        new_cur_gt(i,frameGt(1,i):frameGt(2,i)) = 1;
    end
    
    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error); 
    frame_regular = 1 - frame_error;
    close all    
    hAxis = figure(1);
    plot(1:nSam,frame_regular_aug1,'b',1:nSam,frame_regular_aug2,'r',1:nSam,frame_regular_recons_aug1,'y','LineWidth',3)
    legend('pred-aug1','pred-aug2','recons-aug1')
    for i = 1:size(new_cur_gt,1)
        cur_gt_ = new_cur_gt(i,:);
        [~,one_gt] = find(cur_gt_ == 1);        
        hold on        
        rectangle('Position',[one_gt(1), 0, (one_gt(end)-one_gt(1)), 1],'FaceColor',[0.6 0.9 0.6],'EdgeColor',[0.6 0.9 0.6])      
    end
    
    hold on
    plot(1:nSam,frame_regular_aug1,'b',1:nSam,frame_regular_aug2,'r',1:nSam,frame_regular_recons_aug1,'y','LineWidth',3)
    xlabel('Frame Number')
    ylabel('Regularity Score')
    set(hAxis,'Position',[84 301 1899 451])
    set(gcf,'color','w')
    
        
    print(['avenue_FrameError_CompareAug1Aug2_seq' num2str(numTestFolders)],'-djpeg')
end


%% read rgb frame
% seq = 10;
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
% for id = 50:300%800:length(frame_error) % seq 7
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



