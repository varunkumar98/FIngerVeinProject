function ClasOut=classifyCNN(net,TestData,TrainFeatures,YTrain)
TrainFeatures=double(TrainFeatures);
TestData=double(TestData);
YPr = classify(net,reshape(TestData,28,28,1));
for ik=1:size(TrainFeatures,1)
    Cv(ik)=corr2(TrainFeatures(ik,end-12:end),TestData(end-12:end));
end

[maxOut,Out]=max(Cv);
ClasOut=YTrain(Out);