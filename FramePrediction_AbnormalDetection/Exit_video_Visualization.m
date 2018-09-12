close all
clear all

decision_map_path = '/usr/not-backed-up/1_convlstm/convLSTM_prediction6_exit/RemoveTimeStamp';
addpath('../')

%gt
testSeqPath = '/usr/not-backed-up/1_DATABASE';
% load(fullfile(testSeqPath,'gt1_exit.mat'))
load(fullfile(testSeqPath,'gt1_exit_new_H.mat'))
gt = gt_new;
% MPPCA ground-truth
% load(fullfile(testSeqPath,'Exit_adam.mat'))

ab_thres = 0.9;
v = VideoWriter(sprintf('exit_seq4_thres%0.2f.avi',ab_thres));
v.FrameRate = 24;
v.Quality = 100;
open(v)
for numTestFolders = 4  
    numTestFolders        
    frame_start = 1;
    frame_stop = 3999;
    
    frame_error = h5read(fullfile(decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');       
    nSam = size(frame_error,1);
        
    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error); 
    frame_regular = 1 - frame_error;
    % chose a range to visualize       
    frame_regular = frame_regular(frame_start:frame_stop);       
    
    % video frames
    load(fullfile('/usr/not-backed-up/1_DATABASE/Adam dataset/Exit/TestSeq',['Test' num2str(numTestFolders) '.mat']))
    % just visualize part of a sequence    
    ims = ims(:,:,10:nSam+9);
    ims = ims(:,:,frame_start:frame_stop);
    
    % groundtruth
    frameGt = gt{1,numTestFolders};    
    frame_gt = zeros(1,nSam+9);
    num_ab_event_seq = size(frameGt,2);
    for i = 1:num_ab_event_seq
        frame_gt(frameGt(1,i):frameGt(2,i)) = 1;
    end
    cur_gt = frame_gt(10:nSam+9);     
    cur_gt = cur_gt(frame_start:frame_stop);    
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
    
    nSam_new = length(frame_regular);
    clf
    for fr = 1:nSam_new       
        figure(1)
        im = ims(:,:,fr);        
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
            plot(1:nSam_new,repmat(ab_thres,nSam_new),':r','LineWidth',0.02)
        else
        end
        plot(1:fr,frame_regular(1:fr),'b','LineWidth',3)
        xlim([0 nSam_new])
        ylim([0 1])
        xlabel('Frame Number')
        ylabel('Regularity Score')
        F = getframe(gcf);
        writeVideo(v,F.cdata)        
        pause(0.01)                           
    end            
end
close(v)
