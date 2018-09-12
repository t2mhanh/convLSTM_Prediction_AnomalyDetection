%% EXIT: Visualize testing frames
frame_path='/usr/not-backed-up/1_DATABASE/Adam dataset/Exit/TestSeq/';
for seq = 4%:4
    load(fullfile(frame_path,['Test' num2str(seq) '.mat']));%ims
    size(ims)
    figure
    % video 2: 
    % 2400 : 2550 a person appears and goes from the right to the left
    % 9700: 9800: a person appears and goes from the left to the right.
    % 14285:14295: a person appears and goes from the left to the right.
    % video 3:
    % 8000:8100: a person goes from the left to the right.
    
    for fr = 9934:12040
        fr
        im = ims(:,:,fr);
        imshow(im)
        pause(0.5)
    end
end