require 'torch'
require 'cutorch'
require 'nn'
require 'fbcunn'
require 'math'

function create_cnn()
    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 6 filters: 3x5x5
    net:add(nn.ReLU())
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- pooling 2x2

    -- for second layer
    net:add(nn.SpatialConvolution(6, 16, 5, 5)) -- 16 filters: 6x5x5
    net:add(nn.ReLU())
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- pooling 2x2

    -- for linear layer
    net:add(nn.View(16*5*5)) -- reshape
    net:add(nn.Linear(16*5*5, 120))
    net:add(nn.ReLU())
    net:add(nn.Linear(120, 84))
    net:add(nn.ReLU())
    net:add(nn.Linear(84, 10))
    return net
end

function train (net, trainset)
    criterion = nn.CrossEntropyCriterion():cuda()
    trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 0.001
    trainer.maxIteration = 50
    trainer:train(trainset)
end


function pre_work (trainset)
    mean = {} -- store the mean, to normalize the test set in the future
    stdv  = {} -- store the standard-deviation for the future
    for i = 1, 3 do -- over each image channel
        mean[i] = trainset.data[{ {i}, {}, {}, {}  }]:mean() -- mean estimation
        print('Channel ' .. i .. ', Mean: ' .. mean[i])
        trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

        stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
        print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
        trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end
end


function load_data (path)
    matio = require 'matio'
    trainset = matio.load(path)
    trainset.data = trainset.data:double():view(10000,3,32,32):cuda() -- convert the data from a ByteTensor to a DoubleTensor.
    trainset.labels = trainset.labels:double():cuda()
    function trainset:size()
        return self.data:size(1)
    end
    setmetatable(trainset, {__index = function(self, i)
        return {self.data[i], self.labels[i][1] + 1}
        -- 坑爹啊，错误里分明是0-9类，如下：
        -- Assertion `cur_target >= 0 && cur_target < n_classes' failed.
    end})
    return trainset
end


function main()
    dataset = load_data('cifar-10/data_batch_1.mat')
    pre_work(dataset)

    ok, net = xpcall('torch.load', 'model.t7')
    if not ok then
        net = create_cnn():cuda()
        train(net, dataset)
    end

    saved_net = net:clone('weight','bias','running_mean','running_std')
    torch.save('model.t7', saved_net)
end

main()
