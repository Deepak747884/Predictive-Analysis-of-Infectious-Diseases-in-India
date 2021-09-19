temp = xlsread('ty13to17');
week = (1:261);
temp = temp';
%%plot(x,y);

temp1 = mat2cell(temp, 1, ones(1,261));
week1 = mat2cell(temp, 1, ones(1,261));

%% open loop
trainFcn = 'trainbr';
feedbackdelays = 1:8;
hiddenLayerSize = 10;
net = narnet(feedbackdelays, hiddenLayerSize, 'open', trainFcn);
[x,xi,ai,t] = preparets(net,{},{},week1);
net.divideFcn = 'divideind';
net.dividemode = 'time';
net.divideParam.trainInd = 1:208; 
net.divideParam.valInd = 209:234;
net.divideParam.testInd = 235:261;
net.performFcn = 'mse';
[net,tr] = train(net, x,t,xi,ai);
y = net(x,xi,ai);

trainTargets = gmultiply(t,tr.trainMask);
valTargets = gmultiply(t,tr.valMask);
testTargets = gmultiply(t,tr.testMask);
trainPerformance = perform(net,trainTargets,y);
valPerformance = perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);
view(net)

%% closed loop
netc = closeloop(net);
netc.name = [net.name ' -Closed Loop'];
view(netc)
[xc,xic,aic,tc] = preparets(netc,{},{},week1);
yc = netc(xc,xic,aic);
closedLoopPerformance = perform(net, tc, yc);

%% multi step prediction
[x1,xio,aio,t] = preparets(net,{},{},week1);
[y1,xfo,afo] = net(x1,xio,aio);
[netc,xic,aic] = closeloop(net,xfo,afo);
[y2,xfc,afc] = netc(cell(0,20),xic,aic);
y1_1 = cell2mat(y1);
y2_1 = cell2mat(y2);

plot(1:253, y1_1, '-b');
hold on
plot(254:273, y2_1, '-k'); 
xlabel('Week');
ylabel('No. of cases');

%% step ahead prediction
nets = removedelay(net);
nets.name = [net.name ' - Predict One Step Ahead'];
view(nets)

[xs,xis,ais,ts] = preparets(nets,{},{},week1);
ys = nets(xs,xis,ais);
stepAheadPerformance = perform(net,ts,ys);
