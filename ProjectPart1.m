% Starting code for Final Project: part 1
% 
%  Tasks
% ---------
%   1. Fill in the code for forward propagation and backward propagation
%   2. Play with step size
%   3. Modify the code by changing the number of hidden layers
%   4. Vary b, which controls the non-linearity to determine its impact. 
   


clear all
close all

%%
% Creating linearly non-separable random distributions
%
features = mvnrnd([0,0], eye(2),10000);
radius = sum(features.^2,2);
class1index = radius < 0.5;
class2index = radius > 4;
class1 = features(class1index,:);
class2 = features(class2index,:);
l1 = size(class1,1);

plot(class2(:,1),class2(:,2),'*');
hold on
plot(class1(:,1),class1(:,2),'*');
hold off;

X = [class1;class2];
X1 = [X,ones(size(X,1),1)]';
s = [-ones(size(class1,1),1);ones(size(class2,1),1)]';

%% 
Nh = 3;             % number of nodes in hidden layer 
W1 = randn(Nh,3);   % Weights of first layer
W2 = randn(1,Nh+1); % Weights of second layer

Niter = 5000;       % # iterations
stepsize = 0.0001; % stepsize
error = zeros(1,Niter); % error as a function of iteratiosn

h = @(x) sigmoid(x,1,0.5); % definition of non-linearity
dh = @(x) dsig(x,1,0.5);   % derivative of non-linearity

for iter = 1:Niter
    
    
   % Forward Propagation
   %---------------------------
    
    B = W1 * X1;
    z = h(B);
    a = W2 * [z;ones(1,size(z,2))];
    y = h(a);
               
    %---------------------------                    
           
    error(iter) = sum((y-s).^2);

    % Backward Propagation
    %---------------------------
    
    GradientA = 2 .* (y-s) .* dh(a);
    GradW2 = GradientA * [z;ones(1,size(z,2))]';
    GradientZ1 = W2' * GradientA;
    GradientZ = GradientZ1(1:Nh,:);
    GradientB = GradientZ .* dh(B);
    GradW1 = GradientB * X1';
               
    %---------------------------           
               
    % update weights using steepest descent
    
    W1 = W1 - stepsize*GradW1;
    W2 = W2 - stepsize*GradW2;
    
    % Display the intermediate results
    %----------------------------------------------
    if(mod(iter,10)==0)
        % plot original data and separating lines corresponding to first layer
        figure(1);
        plot(class2(:,1),class2(:,2),'*');
        hold on
        plot(class1(:,1),class1(:,2),'*');    
        xindex = [-25:0.1:25];
        yindex = -(W1(:,2)*xindex + W1(:,3))./W1(:,1);
        for nh=1:Nh
            plot(yindex(nh,:),xindex,'LineWidth',3);
        end
        axis([-10,10,-10,10]);
        hold off;
        title('Original features and separating planes')
        
        % plot output of first layer
        
        if(Nh==3)
            figure(4);plot3(z(1,1:l1),z(2,1:l1),z(3,1:l1),'*');
            hold on
            plot3(z(1,l1+1:end),z(2,l1+1:end),z(3,l1+1:end),'*');
            hold off;
        end
        title('Output of first layer; non-linear features')
        
        figure(2);plot(y,'r--','Linewidth',3); hold on; plot(s); hold off; axis([0,size(X,1),-1.5,1.5]);drawnow;
        title('Predicted and actual labels')
        
    end

    % Display DONE
    %----------------------------------------------
    

    
end

%%

figure(3);plot(error);title('Error - # of misclassified values');