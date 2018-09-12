clear all
% filename = 'AE_ped1_test.txt';
filename = 'prediction5_ped2_test.txt';
delimiterIn = ',';
headerlinesIn = 1;
A = importdata(filename,delimiterIn,headerlinesIn);
test1 = [];
id = 1;
for k = [1,4]
test1(:,id) = A.data(:,k);
id = id + 1;
end

figure
semilogy(test1(:,1),test1(:,2),'-xg')

%%
% filename = 'AE_ped1_train.txt';
% delimiterIn = ',';
% headerlinesIn = 1;
% A = importdata(filename,delimiterIn,headerlinesIn);
% train1 = [];
% id = 1;
% for k = [1,4]
% train1(:,id) = A.data(:,k);
% id = id + 1;
% end
% 
% filename = 'AE_ped1_test.txt';
% delimiterIn = ',';
% headerlinesIn = 1;
% A = importdata(filename,delimiterIn,headerlinesIn);
% test1 = [];
% id = 1;
% for k = [1,4]
% test1(:,id) = A.data(:,k);
% id = id + 1;
% end
% figure
% semilogy(train1(:,1),train1(:,2),'-xr')
% hold on
% semilogy(test1(:,1),test1(:,2),'-xg')

%%
% filename = 'AE_ped2_train.txt';
% delimiterIn = ',';
% headerlinesIn = 1;
% A = importdata(filename,delimiterIn,headerlinesIn);
% train1 = [];
% id = 1;
% for k = [1,4]
% train1(:,id) = A.data(:,k);
% id = id + 1;
% end
% 
% filename = 'AE_ped2_test.txt';
% delimiterIn = ',';
% headerlinesIn = 1;
% A = importdata(filename,delimiterIn,headerlinesIn);
% test1 = [];
% id = 1;
% for k = [1,4]
% test1(:,id) = A.data(:,k);
% id = id + 1;
% end
% figure
% semilogy(train1(:,1),train1(:,2),'-xr')
% hold on
% semilogy(test1(:,1),test1(:,2),'-xg')