require 'dp'

-- 加载Minst数据集
local ds = dp.Cifar10{input_preprocess = {dp.Standardize()}}
print('cifar10 loaded')


-- 参数设置
local imageChannel = 3
local outputClass = 10
local convSize = {3, 120, 72, 36}
local kenerlSize = {5, 3, 3}
local hiddenSize = {36*2*2, 160, 80, 10}
local learningRate = 0.3
local batchSize = 64

-- [model] --
cnn = nn.Sequential()

-- get output size of convolutional layers
outsize = cnn:outside{1,ds:imageSize('c'),ds:imageSize('h'),ds:imageSize('w')}
inputSize = outsize[2]*outsize[3]*outsize[4]
cnn:add(nn.Convert(ds:ioShapes(), 'bchw'))
print("input to dense layers has: "..inputSize.." neurons")

for i = 1, 3 do
    cnn:add(nn.SpatialConvolution(convSize[i], convSize[i+1], kenerlSize[i], kenerlSize[i]))
    cnn:add(nn.SpatialBatchNormalization(convSize[i+1]))
    cnn:add(nn.ReLU())
    cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    cnn:add(nn.SpatialConvolution(convSize[i+1], convSize[i+1], 1, 1))
end


cnn:add(nn.Collapse(2))
for i = 1, 3 do
    cnn:add(nn.Linear(hiddenSize[i], hiddenSize[i+1]))
    if (i ~= 2) then
        cnn:add(nn.BatchNormalization(hiddenSize[i+1]))
    end
    cnn:add(nn.Tanh())
end

cnn:add(nn.LogSoftMax())


print"Model:"
print(cnn)


-- [Propagators] --
ad = dp.AdaptiveDecay{max_wait = 4, decay_factor=0.9}

-- [train] --
train = dp.Optimizer{
    acc_update = false,
    loss = nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert()),
    epoch_callback = function(model, report) -- called every epoch
        if report.epoch > 0 then
            learningRate = learningRate*ad.decay
            learningRate = math.max(0.00001, learningRate)
            print("learningRate", learningRate)
        end
    end,
    callback = function(model, report) -- called every batch
        -- the ordering here is important
        model:updateGradParameters(0.1) -- affects gradParams
        model:updateParameters(learningRate) -- affects params
        model:maxParamNorm(1) -- affects params
        model:zeroGradParameters() -- affects gradParams
    end,
   feedback = dp.Confusion(),
   sampler = dp.ShuffleSampler{batch_size = batchSize},
   progress = true
}


valid = ds:validSet() and dp.Evaluator{
   feedback = dp.Confusion(),
   sampler = dp.Sampler{batch_size = batchSize}
}
test = ds:testSet() and dp.Evaluator{
   feedback = dp.Confusion(),
   sampler = dp.Sampler{batch_size = batchSize}
}

-- [Experiment] --
xp = dp.Experiment{
   model = cnn,
   optimizer = train,
   validator = valid,
   tester = test,
   observer = {
      dp.FileLogger(),
      dp.EarlyStopper{
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = 500
      },
      ad
   },
   random_seed = os.time(),
   max_epoch = 500
}

require 'cutorch'
require 'cunn'
xp:cuda()

xp:verbose(true)
xp:run(ds)
