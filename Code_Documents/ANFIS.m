%% Data
df = readtable("data.xlsx");
x = df(:,["AT","V","AP","RH"]);
x = table2array(x);
y = df(:,"PE");
y = table2array(y);

%% Database exploration
% AT Histogram
histogram(x(:,1:1))
xlabel('AT')
ylabel('Frequency')
title('AT Histogram')
#boxplot AT
figure
boxplot(x(:,1:1),'Notch','on','Labels',{'AT'})

% V Histogram
histogram(x(:,2:2))
xlabel('V')
ylabel('Frequency')
title('V Histogram')
%boxplot V
figure
boxplot(x(:,2:2),'Notch','on','Labels',{'V'})

% AP Histogram
histogram(x(:,3:3))
xlabel('AP')
ylabel('Frequency')
title('AP Histogram')
%boxplot AP
figure
boxplot(x(:,3:3),'Notch','on','Labels',{'AP'})

% RH Histogram
histogram(x(:,4:4))
xlabel('RH')
ylabel('Frequency')
title('RH Histogram')
%boxplot RH
figure
boxplot(x(:,4:4),'Notch','on','Labels',{'RH'})

% PE Histogram
histogram(y)
xlabel('PE')
ylabel('Frequency')
title('PE Histogram')
%boxplot AT
figure
boxplot(y,'Notch','on','Labels',{'PE'})


%% Cross validation
xX = x;
yY = y;
kFolds = 5;
rng('default'); % For reproducibility
cv = cvpartition(height(xX),'KFold',kFolds);
Error = zeros(kFolds,6);
for i=1:kFolds
    fprintf('Modelo gerado, fold %d\n',i);
    %partition indexes
    TrainInd = training(cv,i);
    TestInd = test(cv,i);
    % Train and Test
    x_train = xX(TrainInd,:);
    y_train = yY(TrainInd,:);
    x_test = xX(TestInd,:);
    y_test = yY(TestInd,:);
    % Outlier and normalization
    ncols = 4;
    for j = 1:ncols
        VarX = isoutlier(x_train(:,j:j));
        outlier_ind = find(VarX);
        % eliminate outlier
        if ~isempty(outlier_ind)
            x_train(outlier_ind,:) = [];
            y_train(outlier_ind,:) = [];
        end
        % Normalization of the training dataset
        minV = min(x_train(:,j:j));
        maxV = max(x_train(:,j:j));
        x_train(:,j:j) = (x_train(:,j:j) - minV)/(maxV - minV);
        x_test(:,j:j) = (x_test(:,j:j) - minV)/(maxV - minV);
    end
    % Save cross validation dataset
    sv_tr = [x_train, y_train];
    filename_tr0 = strcat('CrossV_tr', num2str(i),'.dat');
    writematrix(sv_tr, filename_tr0);
    sv_tr = array2table(sv_tr);
    sv_tr.Properties.VariableNames = ["AT","V","AP","RH","PE"];
    filename_tr = strcat('CrossV_tr', num2str(i),'.xlsx');
    writetable(sv_tr,filename_tr);
    
    sv_ts = [x_test, y_test];
    filename_ts0 = strcat('CrossV_ts', num2str(i),'.dat');
    writematrix(sv_ts, filename_ts0);
    sv_ts = array2table(sv_ts);
    sv_ts.Properties.VariableNames = ["AT","V","AP","RH","PE"];
    filename_ts = strcat('CrossV_ts', num2str(i),'.xlsx');
    writetable(sv_ts,filename_ts);

    % ANFIS Model
    %mod = fitlm(x_train, y_train)
    genOpt = genfisOptions('GridPartition');
    genOpt.NumMembershipFunctions = [2 4 2 4];
    genOpt.OutputMembershipFunctionType = 'linear';
    genOpt.InputMembershipFunctionType = ["gbellmf", "gaussmf", "gaussmf", "gbellmf"];

    inFIS = genfis(x_train, y_train,genOpt);
    %[in,out,rule] = getTunableSettings(inFIS);
    %Plot funcoes de pertinencia
    figure
    subplot(2,2,1)
    plotmf(inFIS,'input',1)
    subplot(2,2,2)
    plotmf(inFIS,'input',2)
    subplot(2,2,3)
    plotmf(inFIS,'input',3)
    subplot(2,2,4)
    plotmf(inFIS,'input',4)
    %Treinamento
    Epoch = 48;
    trainOpt = anfisOptions('InitialFIS',inFIS, ...
                            'EpochNumber',Epoch, ...
                            'OptimizationMethod',1);
    %Optimization method used in membership function parameter
    % training. Specify 0 to use backpropagation method or specify 1
    % to use a hybrid method, which combines least squares estimation
    % with backpropagation. The default value is set to 1.
    xtr = [x_train, y_train];
    [outFIS,trainError,stepSize] = anfis(xtr,trainOpt);
    figure
    plot(stepSize)
    % Plot error per training epoch.
    figure
    ep = [1:Epoch];
    plot(ep,trainError,'.b')
    % optimization
    %opt = tunefisOptions("Method","anfis");
    %outFIS = tunefis(inFIS,[in;out],x_train, y_train,opt);
    % Evaluation
    y_pred = evalfis(outFIS,x_test);
    % save predictions
    Spred = [x_test, y_test, y_pred];
    Spred = array2table(Spred );
    Spred .Properties.VariableNames = ["AT","V","AP","RH","PE", "y_pred"];
    filename_pred = strcat('Pred_ANFIS_CV', num2str(i),'.xlsx');
    writetable(Spred ,filename_pred);
    % ANFIS error ( y_pred  &  y_test )
    n = length(y_test);
    EMA = sum(abs(y_pred - y_test)) ./ n; % EMA
    REQM = sqrt(sum((y_test - y_pred).^2) ./ n); % REQM
    ERA = sum(abs(y_test - y_pred)) ./ sum(abs(y_test - mean(y_test))); % ERA
    EQR = sqrt(sum((y_test - y_pred).^2) ./ sum((y_test - mean(y_test)).^2)); % EQR
    Sup = (n .* sum(y_test.*y_pred)) - (sum(y_test) .* sum(y_pred)); % r
    Inf = sqrt(((n .* sum(y_test.^2)) - (sum(y_test).^2)) .* ((n .* sum(y_pred.^2)) - (sum(y_pred).^2)));
    r = Sup ./ Inf;
    R2 = 1 - (sum((y_test - y_pred).^2) ./ sum((y_test - mean(y_test)).^2));  % R2
    % Erro
    Error(i:i,1:1) = EMA;
    Error(i:i,2:2) = REQM;
    Error(i:i,3:3) = ERA;
    Error(i:i,4:4) = EQR;
    Error(i:i,5:5) = r;
    Error(i:i,6:6) = R2;
end

Error = array2table(Error);
Error.Properties.VariableNames = ["EMA", "REQM", "ERA", "EQR", "r", "R2"];
filename_ER = strcat('Anfis_er.xlsx');
writetable(Error,filename_ER);

% FIS parameters
inFIS.Inputs(1).MembershipFunctions
inFIS.Inputs(2).MembershipFunctions
inFIS.Inputs(3).MembershipFunctions
inFIS.Inputs(4).MembershipFunctions
outFIS.Inputs(1,1).MembershipFunctions
outFIS.Inputs(1,2).MembershipFunctions
outFIS.Inputs(1,3).MembershipFunctions
outFIS.Inputs(1,4).MembershipFunctions





















