clear all
gt_path = '/usr/not-backed-up/1_DATABASE/Avenue Dataset/testing_label_mask';
save_path = '/usr/not-backed-up/1_DATABASE/Avenue Dataset/Test_gt_location';
if ~exist(save_path,'dir'), mkdir(save_path),else end 
new_size = [227 227];
for seq = 1:21    
    load(fullfile(gt_path,[num2str(seq) '_label.mat']));
    
    nfr = size(volLabel,2);
    gt_location = cell(1,nfr);
    for fr = 1:nfr
        cur_gt = volLabel{fr};
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