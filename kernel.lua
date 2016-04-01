require 'math'
require 'torch'
require 'cutorch'
require 'nn'
require 'fbcunn'
require 'unsup'


function load_data (path)
    matio = require 'matio'
    trainset = matio.load(path)
    trainset.data = trainset.data:double():view(10000,3,32,32) -- convert the data from a ByteTensor to a DoubleTensor.
    trainset.labels = trainset.labels:double()
    print(trainset)
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


local dataset = load_data('cifar-10/data_batch_1.mat')
pre_work(dataset) -- 归一化

local patch = torch.Tensor(30000, 75)
for i = 1, 30000 do
    var = math.random(dataset:size())
    img = dataset[var][1]

    x = math.random(27)
    y = math.random(27)
    patch[i] = img[{{}, {x, x+4}, {y, y+4}}]:reshape(3*5*5)

end

-- require 'itorch'
-- itorch.image(dataset[{{1,25}}][1])

patch = unsup.zca_whiten(patch)
local centroids, count = unsup.kmeans(patch, 100, 100, 10000)

require 'itorch'
itorch.image(centroids:view(100, 3, 5, 5))
