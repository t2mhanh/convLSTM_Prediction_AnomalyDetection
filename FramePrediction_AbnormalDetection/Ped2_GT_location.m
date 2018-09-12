clear all
gt_path = '/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/VideoSequences/UCSDped2/Test_gt';
save_path = '/usr/not-backed-up/1_DATABASE/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset.v1p2/VideoSequences/UCSDped2/Test_gt_location';
if ~exist(save_path,'dir'), mkdir(save_path),else end 
new_size = [227 227];
for seq = 8%1:12
    if seq < 10
        load(fullfile(gt_path,['Test00' num2str(seq) '_gt.mat']));
    else
        load(fullfile(gt_path,['Test0' num2str(seq) '_gt.mat']));
    end
    nfr = size(M,3);
    gt_location = cell(1,nfr);
    for fr = 1:nfr
        cur_gt = M(:,:,fr);
        cur_gt = imresize(cur_gt,new_size,'nearest')/255;
        [L,nr,rp] = connectedcomponents(cur_gt);
        gt = zeros(1,4);
        if nr ~= 0
            for id = 1:nr                
                [i,j] = find((L == id)==1);
                if length(i) ~= 0
                    gt(id,:) = [min(j) min(i) (max(j) - min(j)) (max(i) - min(i))];
                end
            end        
            gt_location{1,fr} = gt;        
        end
    end
    save(fullfile(save_path,['Test' num2str(seq) '_gt_location.mat']),'gt_location','-v7.3')
end