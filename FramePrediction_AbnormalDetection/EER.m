% function [x1,y1] = EER(x,y)
% % EER line x + y - 1 = 0
% % x : FPR  
% % y : TPR
% % x, y: vectors
% d = x + y - 1;
% %% 3rd version - 28April - CHECK AND CORRECT IT LATER
% % sign = d > 0;
% % sign_dif = sign(2:end) - sign(1:end-1);
% % [~,I] = find(sign_dif ~= 0);
% % for i = 1:length(I)
% % end
% %% 2nd version - has error when ROC decrease after increasing
% neg = find(d<0);
% neg_max = max(d(neg)); % ~ this point is close to EER line 
% neg_point_idx = find(d == neg_max);
% if length(neg_point_idx) > 1; neg_point_idx = neg_point_idx(1); else end
% neg_point = [x(neg_point_idx) y(neg_point_idx)];
% 
% pos = find(d>0);
% pos_min = min(d(pos));
% pos_point_idx = find(d == pos_min);
% if length(pos_point_idx) > 1; pos_point_idx = pos_point_idx(1); else end
% pos_point = [ x(pos_point_idx) y(pos_point_idx)];
% 
% %% use inverse _ has some error
% % a_b = [(x(neg_point_idx) - x(pos_point_idx)) -(y(neg_point_idx)-y(pos_point_idx))]; 
% % out = inv([1 1; a_b])*[1;a_b*neg_point'];
% % x1 = out(1);
% % y1 = out(2);
% %% 
% if (pos_point(1) == neg_point(1))
%     x1 = pos_point(1);
%     
% else
%     m = (pos_point(2) - neg_point(2))./(pos_point(1) - neg_point(1));
%     c = pos_point(2) - m.*pos_point(1);
%     x1 = (1 - c) ./ (1 + m);
% end
%     y1 = -x1 + 1;

%%
function eer = EER(x,y)
fpr = x;
fnr = 1 - y;
dist_min = 9999;
for i = 1:length(fpr)
    dist = abs(fpr(i) - fnr(i));
    if dist < dist_min
        dist_min = dist;
        eer = (fpr(i) + fnr(i))/2;
%         i
    end
    
end