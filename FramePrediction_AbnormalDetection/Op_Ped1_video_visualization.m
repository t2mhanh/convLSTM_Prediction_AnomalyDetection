close all
clear all
   
Data = 'UCSDped1';% UCSDped1/UCSDped2
OptDir = fullfile('/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/OpticalFlow',Data,'Test');
GtDir = fullfile('/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/VideoSequences',Data,'Test_gt'); % ground truth
ImDir = fullfile('/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/VideoSequences',Data,'Test'); % ground truth
% savePath1 = '/usr/not-backed-up/MODELS_DATA/data/WTA_128dimOptFeature_24/Ped1_5frames/localSVM_oriSize_Grid6x9';
savePath1 = '/usr/not-backed-up/MODELS_DATA/data/convWTA_BMVCcheck/Ped1_512d_arc3_Grid6x9'; % August 2018 from Arc machine

% stride = 12;
id = 1;
nS = 10;
F(1:nS) = struct('cdata',[],'colormap',[]);
for numTestFolders = 14%
    numTestFolders
    % load ground truth maps
    if numTestFolders < 10
        GroundTruthPath = fullfile(GtDir,['Test00' num2str(numTestFolders) '_gt.mat']);
    else
        GroundTruthPath = fullfile(GtDir,['Test0' num2str(numTestFolders) '_gt.mat']);
    end
    load(GroundTruthPath)
    groundTruthAll = M./max(M(:)); %M[nr x nc x numFrames]
    clear M

    load(fullfile(savePath1,['decisionMap_' num2str(numTestFolders) '.mat']));     

    nSam = size(decisionMap,3);
   %%% VISUALIZATION
    if numTestFolders < 10
       load(fullfile(ImDir,['Test00' num2str(numTestFolders) '.mat']));
    else
        load(fullfile(ImDir,['Test0' num2str(numTestFolders) '.mat']));
    end
%         decisionMap_vis = (decisionMap - min(decisionMap(:)))./(max(decisionMap(:)) - min(decisionMap(:)));
%         decisionMap_vis(1,1,:) = 0;
%         decisionMap_vis(end,end,:) = 1;  

    %%%
    Result_Thresh = [];           
    frameSize = [240 360];
    for frame_num = 1:nSam
        % ground Truth for correspond frame
        groundTruth = groundTruthAll(:,:,frame_num+3);  
        groundTruth = imresize(groundTruth,frameSize,'nearest');
        [nr,nc] = size(groundTruth);
        groundTruth_ = sum(groundTruth(:));
        im = M(:,:,frame_num+3)/255; 
        im = imresize(im,frameSize,'nearest');
        cur_decisionMap = decisionMap(:,:,frame_num); 
        cur_decisionMap = imresize(cur_decisionMap,size(groundTruth),'bilinear');
        Result_ = [];
        for thres = 0%0.002%0.33%
            anomaly_map = zeros(nr,nc);                                               
            %
           [I,J] = find(cur_decisionMap >= thres);                        
            for i = 1 : length(I)
%                     anomaly_map(stride*(I(i)-1)+1:stride*(I(i)-1)+24,stride*(J(i)-1)+1:stride*(J(i)-1)+24) = 1;
                anomaly_map(I(i),J(i)) = 1;
            end
            %############ VISUALIZATION ##############
            % % % % % NOTE: red GT, green:My result, yellow: overlap
            im_rgb = [];
            im_rgb(:,:,1) = im.*or(or(not(anomaly_map).*groundTruth, anomaly_map.*groundTruth),not(groundTruth).*not(anomaly_map));
            im_rgb(:,:,2) = im.*or(or(anomaly_map.*not(groundTruth), anomaly_map.*groundTruth),not(groundTruth).*not(anomaly_map));
            im_rgb(:,:,3) = im.*not(groundTruth).*not(anomaly_map);
            
            im_rgb = [im_rgb; zeros(19,size(im_rgb,2),3)];
            [nr,nc,~] = size(im_rgb);
    %             figure(1)
            hFig = figure(1);
            set(hFig, 'Position', [0 0 242 420])
%             if frame_num < 15 % show legend at beginning
                text_str = cell(3,1);
                text_str{1} = 'missed;';
                text_str{2} = 'incorrectly;';
                text_str{3} = 'correctly';
                text_str{4} = 'predicted pixels';
%                 position = [1 1; 1 19; 1 37];
                position = [1 nr-18; 43 nr-18; 103 nr-18;152 nr-18];
                box_color = {'red','green','yellow','black'};
                im_rgb_ = insertText(im_rgb, position,text_str,'FontSize',10,'BoxColor',...
                    box_color,'BoxOpacity',0.4,'TextColor','white');            
                imshow(im_rgb_)
                truesize
%             else
%                 imshow(im_rgb)
%                 truesize
%             end
%             F(id) = getframe(gcf);
            F(id).cdata = getimage(gcf);
            id = id + 1;
    % %             figure(2)
    % %             imagesc(decisionMap_vis(:,:,frame_num))
            pause(0.1)         
        end
    end
end
v = VideoWriter(fullfile(savePath1,['UCSDSequences' num2str(numTestFolders) '_AbnormalThres0']),'MPEG-4');
v.FileFormat = 'mp4';
v.FrameRate = 8;
v.Quality = 100;
open(v)
writeVideo(v,F)
close()
%'VideoCompressionMethod','None','fileFormat','mp4','FrameRate',8);
% movie2avi(F,fullfile(savePath1,['UCSDSequences' num2str(numTestFolders) '_AbnormalThres0']),'compression','None','fps',8)  