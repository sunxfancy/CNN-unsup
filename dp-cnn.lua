require 'dp'
-- 预处理 --

local input_preprocess = {dp.Standardize(), dp.ZCA()}

-- 加载Minst数据集
local ds = dp.Cifar10{input_preprocess = input_preprocess}
print('cifar10 loaded')


-- 参数设置
local imageChannel = 3
local outputClass = 10
local dropoutProb = {0.2,0.3,0.5}
local convSize = {3, 8, 16}
local hiddenSize = {16*5*5, 200, 100, 10}
local learningRate = 0.2
local batchSize = 64
local cuda = false

-- [model] --
cnn = nn.Sequential()

for i = 1, 2 do -- 两层卷积
    cnn:add(nn.SpatialDropout(dropoutProb[i]))
    cnn:add(nn.SpatialConvolution(convSize[i], convSize[i+1], 5, 5))
    cnn:add(nn.SpatialBatchNormalization(convSize[i+1]))
    cnn:add(nn.Tanh())
    cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2))
end

-- get output size of convolutional layers
outsize = cnn:outside{1,ds:imageSize('c'),ds:imageSize('h'),ds:imageSize('w')}
inputSize = outsize[2]*outsize[3]*outsize[4]
cnn:insert(nn.Convert(ds:ioShapes(), 'bchw'), 1)

print("input to dense layers has: "..inputSize.." neurons")

cnn:add(nn.Collapse(3))
for i = 1, 3 do
    cnn:add(nn.Dropout(dropoutProb[i]))
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
         max_epochs = 50
      },
      ad
   },
   random_seed = os.time(),
   max_epoch = 50
}

if cude then
    require 'cutorch'
    require 'cunn'
    xp:cuda()
end

xp:verbose(true)
xp:run(ds)
