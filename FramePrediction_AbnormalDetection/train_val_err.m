close all
filename = 'ped1_prediction6_train.txt';
delimiterIn = ',';
headerlinesIn = 1;
A = importdata(filename,delimiterIn,headerlinesIn);
train1 = [];
id = 1;
for k = [1,4]
train1(:,id) = A.data(:,k);
id = id + 1;
end
train1 = train1(1:36000,:);


filename = 'ped1_prediction6_test.txt';
delimiterIn = ',';
headerlinesIn = 1;
A = importdata(filename,delimiterIn,headerlinesIn);
test1 = [];
id = 1;
for k = [1,4]
test1(:,id) = A.data(:,k);
id = id + 1;
end
test1 = test1(1:80,:);

figure
semilogy(train1(:,1),train1(:,2),'-xr')
hold on
semilogy(test1(:,1),test1(:,2),'-xg')
ylim([10^0 10^4])
xlim([0 360000])
title('Training Euclidean Loss ')
xlabel('Iteration')
ylabel('Training Loss')
legend('Train','Validation','NorthWest')
%% ped2
% filename = 'ped2_prediction6_train.txt';
% delimiterIn = ',';
% headerlinesIn = 1;
% A = importdata(filename,delimiterIn,headerlinesIn);
% train1 = [];
% id = 1;
% for k = [1,4]
% train1(:,id) = A.data(:,k);
% id = id + 1;
% end
% train1 = train1(1:14000,:);
% 
% 
% filename = 'ped2_prediction6_test.txt';
% delimiterIn = ',';
% headerlinesIn = 1;
% A = importdata(filename,delimiterIn,headerlinesIn);
% test1 = [];
% id = 1;
% for k = [1,4]
% test1(:,id) = A.data(:,k);
% id = id + 1;
% end
% test1 = test1(1:80,:);
% 
% figure
% semilogy(train1(:,1),train1(:,2),'-xr')
% hold on
% semilogy(test1(:,1),test1(:,2),'-xg')
% ylim([10^0 10^4])
% title('Training Euclidean Loss ')
% xlabel('Iteration')
% ylabel('Training Loss')
% legend('Train','Validation','NorthWest')

%% avenue
% filename = 'avenue_prediction6_train.txt';
% delimiterIn = ',';
% headerlinesIn = 1;
% A = importdata(filename,delimiterIn,headerlinesIn);
% train1 = [];
% id = 1;
% for k = [1,4]
% train1(:,id) = A.data(:,k);
% id = id + 1;
% end
% train1 = train1(1:82000,:);
% 
% 
% filename = 'avenue_prediction6_test.txt';
% delimiterIn = ',';
% headerlinesIn = 1;
% A = importdata(filename,delimiterIn,headerlinesIn);
% test1 = [];
% id = 1;
% for k = [1,4]
% test1(:,id) = A.data(:,k);
% id = id + 1;
% end
% test1 = test1(1:80,:);
% 
% figure
% semilogy(train1(:,1),train1(:,2),'-xr')
% hold on
% semilogy(test1(:,1),test1(:,2),'-xg')
% ylim([10^0 10^4])
% xlim([0 820000])
% title('Training Euclidean Loss ')
% xlabel('Iteration')
% ylabel('Training Loss')
% legend('Train','Validation','NorthWest')
%% avenue
% filename = 'entrance_prediction6_train.txt';
% delimiterIn = ',';
% headerlinesIn = 1;
% A = importdata(filename,delimiterIn,headerlinesIn);
% train1 = [];
% id = 1;
% for k = [1,4]
% train1(:,id) = A.data(:,k);
% id = id + 1;
% end
% train1 = train1(1:80000,:);
% 
% 
% filename = 'entrance_prediction6_test.txt';
% delimiterIn = ',';
% headerlinesIn = 1;
% A = importdata(filename,delimiterIn,headerlinesIn);
% test1 = [];
% id = 1;
% for k = [1,4]
% test1(:,id) = A.data(:,k);
% id = id + 1;
% end
% test1 = test1(1:80,:);
% 
% figure
% semilogy(train1(:,1),train1(:,2),'-xr')
% hold on
% semilogy(test1(:,1),test1(:,2),'-xg')
% ylim([10^0 10^4])
% title('Training Euclidean Loss ')
% xlabel('Iteration')
% ylabel('Training Loss')
% legend('Train','Validation','NorthWest')
%% exit
% filename = 'exit_prediction6_train.txt';
% delimiterIn = ',';
% headerlinesIn = 1;
% A = importdata(filename,delimiterIn,headerlinesIn);
% train1 = [];
% id = 1;
% for k = [1,4]
% train1(:,id) = A.data(:,k);
% id = id + 1;
% end
% train1 = train1(1:40000,:);
% 
% 
% filename = 'exit_prediction6_test.txt';
% delimiterIn = ',';
% headerlinesIn = 1;
% A = importdata(filename,delimiterIn,headerlinesIn);
% test1 = [];
% id = 1;
% for k = [1,4]
% test1(:,id) = A.data(:,k);
% id = id + 1;
% end
% test1 = test1(1:80,:);
% 
% figure
% semilogy(train1(:,1),train1(:,2),'-xr')
% hold on
% semilogy(test1(:,1),test1(:,2),'-xg')
% ylim([10^0 10^4])
% title('Training Euclidean Loss ')
% xlabel('Iteration')
% ylabel('Training Loss')
% legend('Train','Validation','NorthWest')
