close all
clear all
decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_avenue/';
addpath('../')

%gt
testSeqPath = '/usr/not-backed-up/1_DATABASE';
load(fullfile(testSeqPath,'gt_avenue.mat'))

ab_thres = 0.67;
v = VideoWriter(sprintf('avenue_4seqs_thres%0.2f.avi',ab_thres));
v.FrameRate = 24;
v.Quality = 100;
open(v)

for numTestFolders = [5 7 15 12]%1:21
        numTestFolders

    if numTestFolders < 10
        load(['/usr/not-backed-up/1_DATABASE/Avenue Dataset/testing_vol/vol0' num2str(numTestFolders) '.mat']);
    else
        load(['/usr/not-backed-up/1_DATABASE/Avenue Dataset/testing_vol/vol' num2str(numTestFolders) '.mat']);
    end
    
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
% %     im1 = vol(:,:,650);
% %     im2 = vol(:,:,920);
% %     im3 = vol(:,:,1010);
    
    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error); 
    frame_regular = 1 - frame_error;
    clf
%     figure(1)
    for fr = 1:nSam       
        figure(1)
        im = vol(:,:,fr+9);        
        im = imresize(im,[360 540],'nearest');
%         im = insertText(im,[21 18],'Abnormal','FontSize',35,'BoxOpacity',0.4,'TextColor','red');
        if frame_regular(fr) < ab_thres;
            im = insertText(im,[21 18],'Abnormal','FontSize',25,'BoxColor','red','TextColor','white');
        else
            im = insertText(im,[21 18],'Normal','FontSize',25,'BoxColor','green','TextColor','white');
        end
        h1 = subplot(2,1,1);
        pos1 = get(h1,'Position');
        pos = pos1 + [-0.08 -0.1 0.15 0.15];
        set(h1,'Position',pos)
        imshow(im)
        
        
        % regular score with green shade for GT
        subplot(2,1,2)
        if fr == 1
            
            for i = 1:size(new_cur_gt,1)
                cur_gt_ = new_cur_gt(i,:);
                [~,one_gt] = find(cur_gt_ == 1);        
                hold on        
                rectangle('Position',[one_gt(1), 0, (one_gt(end)-one_gt(1)), 1],'FaceColor',[0.6 0.9 0.6],'EdgeColor',[0.6 0.9 0.6])      
            end
            hold on
            plot(1:nSam,repmat(ab_thres,nSam),':r','LineWidth',0.02)
        else
        end
        plot(1:fr,frame_regular(1:fr),'b','LineWidth',3)
        xlim([0 nSam])
        ylim([0 1])
        xlabel('Frame Number')
        ylabel('Regularity Score')
        F = getframe(gcf);
        writeVideo(v,F.cdata)        
%         pause(0.1)                           
    end
end
close(v)

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



