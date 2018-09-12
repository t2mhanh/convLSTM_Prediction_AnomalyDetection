close all
clear all
decision_map_path = '/usr/not-backed-up/1_convlstm/Ped1_prediction6/';
addpath('../')
%gt
testSeqPath = '/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/';
ImDir = fullfile('/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/VideoSequences/UCSDped1/Test'); 
load(fullfile(testSeqPath,'TestFrameGT_original.mat'))%,'FrameGt')   % FROM UCSD 
%% visualization
ab_thres = 0.65;

% v = VideoWriter(['ped1_5seqs' num2str(ab_thres) '.avi']);
v = VideoWriter(sprintf('ped1_5seqs%0.2f.avi',ab_thres));
v.FrameRate = 8;
v.Quality = 100;
open(v)

for numTestFolders = [1 5 13 14 18 23 24]
    numTestFolders
    if numTestFolders < 10
       load(fullfile(ImDir,['Test00' num2str(numTestFolders) '.mat']));
    else
        load(fullfile(ImDir,['Test0' num2str(numTestFolders) '.mat']));
    end
    
    frameGt = FrameGt{1,numTestFolders};        
    frame_error = h5read(fullfile(decision_map_path,['test_' num2str(numTestFolders) '_error.h5']),'/frame_error');   
    nSam = size(frame_error,1);
    cur_gt = frameGt(10:nSam+9); % [1x191]
        
    
    frame_error = (frame_error - min(frame_error));
    frame_error = frame_error / max(frame_error); 
    frame_regular = 1 - frame_error;
    
    clf
%     figure(1)
    for fr = 1:nSam       
        figure(1)
        im = M(:,:,fr+9)/255;        
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
    %
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
