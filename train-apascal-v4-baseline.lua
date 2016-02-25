require 'residual-layers'
require 'apascal-dataset'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'optim'
require 'logger'

opt = lapp[[
      --device          (default 1)        Gpu Device to use
      --batchSize       (default 128)      Sub-batch size
      --dataRoot        (default ./dataset/attribute/attribute_data/)        Data root folder
      --imageRoot       (default ./dataset/attribute/apascal_images/)       Image dir
      --baseModel       (default .)  Base model
      --loadFrom        (default "")      Model to load
      --experimentName  (default "snapshots/aPascal_weight_balance_root_v4_bl/")
      --lossFunction    (default "margin")  Target Loss Function - cross entropy or margin
]]

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)

cutorch.setDevice(opt.device)
cutorch.manualSeedAll(1)

logger = Logger()
if( opt.loadFrom ~= "" ) then
    logger:open( opt.experimentName, true )
else
    logger:open( opt.experimentName, false )
end
logger:info( opt )

dataset = Dataset('dataset/attribute/attribute_data/',
                  'dataset/attribute/apascal_images/',
                   torch.Tensor({0.0,0,0,0,0}),1.0
                   ,opt.batchSize
                   ,opt.lossFunction)
train_size, val_size = dataset:size()

balance_weights = {}
local txt = io.open('dataset/attribute/attribute_data/balancing_weights.txt')
for line in txt:lines() do
    line = line:split('\t')
    balance_weights[#balance_weights+1] = {pl = tonumber(line[1]), nl = tonumber(line[2])}
end
txt:close()

local model, sgdState
if( opt.loadFrom ~= "" ) then
    logger:info("Reloading Model...")
    model  = torch.load( opt.experimentName..'model'..opt.loadFrom..'.th' )
    sgdState = torch.load( opt.experimentName..'sgdState'..opt.loadFrom..'.th' )
else
    --logger:info("From BaseModel "..opt.baseModel )
    --base_model = torch.load( opt.baseModel )

    --------------------------------------------------------Building Model
    input = nn.Identity()()
    ------> 64, 224,224
    model = cudnn.SpatialConvolution(3, 64, 7,7, 2,2, 3,3)(input)
    ------> 64, 112,112
    --model = nn.SpatialBatchNormalization(64)(model)
    model = cudnn.ReLU(true)(model)
    model = cudnn.SpatialMaxPooling(3,3,  2,2,  1,1)(model)
    ------> 64, 56,56
    model = addResidualLayer2(model, 64)
    model = addResidualLayer2(model, 64)
    model = addResidualLayer2(model, 64)
    ------> 64, 56,56

    ------> 64, 56,56
    model = addResidualLayer2(model, 64, 128, 2)
    model = addResidualLayer2(model, 128)
    model = addResidualLayer2(model, 128)
    model = addResidualLayer2(model, 128)
    ------> 128, 28,28

    ------> 128, 28,28
    model = addResidualLayer2(model, 128, 256, 2)
    model = addResidualLayer2(model, 256)
    model = addResidualLayer2(model, 256)
    model = addResidualLayer2(model, 256)
    ------> 256, 14,14

    ------> 256, 14,14
    model = addResidualLayer2(model, 256, 512, 2)
    model = addResidualLayer2(model, 512)
    model = addResidualLayer2(model, 512)
    ------> 512, 7,7

    -- Level 4 features --
    ------> 512, 7,7
    l4 = cudnn.SpatialAveragePooling(7, 7, 1,1, 0,0)(model)
    ------> 512, 1, 1
    l4 = nn.Reshape(512)(l4)
    ------> 512
    l4 = nn.Linear(512,64)(l4)
    ------> 64
    l4 = nn.BatchNormalization(64)(l4)
    if( opt.lossFunction ~= 'margin' ) then
        l4 = nn.Sigmoid()(l4)
    end
    l4 = nn.SplitTable(2,64)(l4)
    --------------------------------------------------------Building Model End
    --------------------------------------------------------Parameter initialize
    --l1_model = nn.gModule({input}, {model})
    model = nn.gModule({input}, {l4})

    model:cuda()
    model:apply(function(m)
        -- Initialize weights
        local name = torch.type(m)
        if name:find('Convolution') then
            m.weight:normal(0.0, math.sqrt(2/(m.nInputPlane*m.kW*m.kH)))
            m.bias:fill(0)
        elseif name:find('BatchNormalization') then
            if m.weight then m.weight:normal(1.0, 0.002) end
            if m.bias then m.bias:fill(0) end
        end
    end)

    --local base_weight, _ = base_model:parameters()
    --local weight, _ = l1_model:parameters()
    --for i, t in ipairs(base_weight) do
    --    weight[i]:copy(t)
    --end

    --local base_modules = base_model:listModules()
    --local modules = l1_model:listModules()
    --for i, bm in ipairs(base_modules) do
    --    local m=modules[i]
    --    local name = torch.type(bm)
    --    if name:find('BatchNormalization') then
    --        m.running_std:copy(bm.running_std)
    --        m.running_mean:copy(bm.running_mean)
    --        --m.save_std:copy(bm.save_std)
    --        --m.save_mean:copy(bm.save_mean)
    --    end
    --end

    --------------------------------------------------------Parameter initialize End
    sgdState = {
       --- For SGD with momentum ---
       --[[
       -- My semi-working settings
       learningRate   = "will be set later",
       weightDecay    = 1e-4,
       -- Settings from their paper
       --learningRate = 0.1,
       --weightDecay    = 1e-4,

       momentum     = 0.9,
       dampening    = 0,
       nesterov     = true,
       --]]
       --- For rmsprop, which is very fiddly and I don't trust it at all ---
       ----[[
       learningRate = 1e-3,
       alpha = 0.9,
       whichOptimMethod = 'rmsprop',
       ----]]
       --- For adadelta, which sucks ---
       --[[
       rho              = 0.3,
       whichOptimMethod = 'adadelta',
       --]]
       --- For adagrad, which also sucks ---
       --[[
       learningRate = 3e-4,
       whichOptimMethod = 'adagrad',
       --]]
       --- For adam, which also sucks ---
       --[[
       learningRate = 0.005,
       whichOptimMethod = 'adam',
       --]]
       --- For the alternate implementation of NAG ---
       --[[
       learningRate = 0.01,
       weightDecay = 1e-6,
       momentum = 0.9,
       whichOptimMethod = 'nag',
       --]]
    }
end
--------------------------------------------------------Loss
loss = nn.ParallelCriterion()
for i = 1,64 do
    if ( opt.lossFunction == 'margin' ) then
        local mce = nn.MarginCriterion() -- Default margin is 1
        loss:add( mce )
    else
        local pl = balance_weights[i].pl
        local nl = balance_weights[i].nl
        local bce = nn.BCECriterion( torch.Tensor(opt.batchSize):fill( math.sqrt(nl)/math.sqrt(pl) ):float() ):cuda()
        loss:add( bce, math.sqrt(pl)/(math.sqrt(pl)+math.sqrt(nl)) )
    end
end
--
loss:cuda()
graph.dot(model.fg, 'mymodel', 'mymodel')
--------------------------------------------------------Loss End

weights, gradients = model:getParameters()
-- TODO: check whether saving this model is working or not. slight difference exists when test this code.
save_model = model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std','save_mean','save_std')

function forwardBackward()
    model:training()
    gradients:zero()

    ims,labels= dataset:get_samples('train')
    ims = ims:cuda()
    for i = 1,64 do
        labels[i] = labels[i]:cuda()
    end

    collectgarbage(); collectgarbage();
    local y = model:forward(ims)
    local loss_val = loss:forward(y, labels)
    local df_dw = loss:backward(y, labels)
    model:backward(ims, df_dw)

    local loss_per_attribute = {}
    for i = 1,64 do
        loss_per_attribute[i] = loss.criterions[i].output
    end

    return loss_val, loss_per_attribute, gradients, ims:size(1)
end

function eval( ims, labels )
    local true_positive = torch.Tensor(64):zero()
    local true_negative = torch.Tensor(64):zero()
    local false_positive = torch.Tensor(64):zero()
    local false_negative = torch.Tensor(64):zero()
    collectgarbage(); collectgarbage();
    local y = model:forward( ims:cuda() )
    for label_i = 1,64 do
        local prediction
        if( opt.lossFunction == 'margin' ) then
            prediction = torch.gt(y[label_i]:float(), torch.Tensor(y[label_i]:size()):fill(0.0)):float()
        else
            prediction = torch.gt(y[label_i]:float(), torch.Tensor(y[label_i]:size()):fill(0.5)):float()
        end
        local correct = torch.eq( prediction, labels[label_i] ):float()
        local not_correct = torch.ne( prediction, labels[label_i] ):float()

        local tp = torch.eq( correct + labels[label_i], torch.Tensor(correct:size()):fill(2.0) ):sum() -- tensor doesn't have and operation :(
        local fp = torch.eq( not_correct + prediction, torch.Tensor(not_correct:size()):fill(2.0) ):sum()

        true_positive[label_i] = true_positive[label_i] + tp
        true_negative[label_i] = true_negative[label_i] + correct:sum() - tp
        false_positive[label_i] = false_positive[label_i] + fp
        false_negative[label_i] = false_negative[label_i] + not_correct:sum() - fp
    end

    return ims:size(1), true_positive, true_negative, false_positive, false_negative
end
function eval_sample()
    model:evaluate()

    ims,labels= dataset:get_samples('val')
    return eval(ims, labels)
end
function eval_all() --Evaluate
    model:evaluate()

    local flag,ims,labels
    local co = dataset.get_valid_sample_co()

    local total = 0

    local correct = torch.Tensor(64):zero()

    local true_positive = torch.Tensor(64):zero()
    local true_negative = torch.Tensor(64):zero()
    local false_positive = torch.Tensor(64):zero()
    local false_negative = torch.Tensor(64):zero()

    while true do
        flag,ims,labels = coroutine.resume(co, dataset)
        if( ims == nil ) then
            break
        end
        local ret = {eval(ims,labels)}

        total = total + ret[1]
        true_positive = true_positive + ret[2]
        true_negative = true_negative + ret[3]
        false_positive = false_positive + ret[4]
        false_negative = false_negative + ret[5]

        xlua.progress(total,val_size)
    end

    return total,true_positive,true_negative,false_positive,false_negative
end


function afterEpoch(i)
    if( i % 10 == 0 ) then
        print 'saving model....'
        torch.save( opt.experimentName..'model'..string.format("%03d",i)..'.th', save_model )
        torch.save( opt.experimentName..'sgdState'..string.format("%03d",i)..'.th', sgdState )
    end

    print 'evaluate....'
    ret = {eval_all()}
    logger:eval( {state = {nSampledImage = sgdState.nSampledImages, nEvalCounter = sgdState.nEvalCounter, epochCounter = sgdState.epochCounter }, eval = {total = ret[1], tp = torch.totable(ret[2]), tn = torch.totable(ret[3]), fp = torch.totable(ret[4]), fn = torch.totable(ret[5])} } )

    print 'resume training....'
end

function train( fb, weights, sgdState, epochSize, maxEpoch, afterEpoch )
   sgdState.epochCounter = sgdState.epochCounter or 0
   sgdState.nSampledImages = sgdState.nSampledImages or 0
   sgdState.nEvalCounter = sgdState.nEvalCounter or 0

   if sgdState.whichOptimMethod then
       optimizer = optim[sgdState.whichOptimMethod]
   end

   while true do -- Each epoch
      collectgarbage(); collectgarbage()
      -- Run forward and backward pass on inputs and labels
      local loss_val, loss_per_attribute, gradients, batchProcessed = fb()
      -- print (loss_val)
      logger:train( {state = {nSampledImage = sgdState.nSampledImages, nEvalCounter = sgdState.nEvalCounter, epochCounter = sgdState.epochCounter }, total_loss = loss_val, loss_per_attributes = loss_per_attribute } )
      -- SGD step: modifies weights in-place
      optimizer(function() return loss_val, gradients end,
                       weights,
                       sgdState)
      -- Display progress and loss
      sgdState.nSampledImages = sgdState.nSampledImages + batchProcessed
      sgdState.nEvalCounter = sgdState.nEvalCounter + 1
      xlua.progress( math.max(sgdState.nSampledImages - epochSize * sgdState.epochCounter, epochSize) , epochSize)

      if math.floor(sgdState.nSampledImages / epochSize) ~= sgdState.epochCounter then
         -- Epoch completed!
         -- xlua.progress(sgdState.epochCounter, maxEpoch)
         sgdState.epochCounter = math.floor(sgdState.nSampledImages / epochSize)
         print("\n\n----- Epoch "..sgdState.epochCounter.." -----")

         if afterEpoch then afterEpoch(sgdState.epochCounter) end
         if sgdState.epochCounter > maxEpoch then
             break
         end
      end
   end
end

train( forwardBackward, weights, sgdState, train_size, 100, afterEpoch )

--total,true_positive,true_negative,false_positive,false_negative = eval_all()
--for i = 1,64 do
--    local accuracy = (true_positive[levelNum+1][i] + true_negative[levelNum+1][i]) / total
--    local precision = true_positive[levelNum+1][i] / (true_positive[levelNum+1][i] + false_positive[levelNum+1][i])
--    local recall = true_positive[levelNum+1][i] / (true_positive[levelNum+1][i] + false_negative[levelNum+1][i])
--    local f1 = 2 * precision * recall / (precision + recall)
--    print (string.format("%4d/%4d/%4d/%4d/%4d/%0.2f/%0.2f/%0.2f/%0.2f",total, true_positive[levelNum+1][i], true_negative[levelNum+1][i], false_positive[levelNum+1][i], false_negative[levelNum+1][i], accuracy, precision, recall, f1))
--end

logger:close()

-- save whole model for emergency
torch.save( opt.experimentName..'completed.th', model )

--[[ TODO :
--base model weight loading
--learning rate decay
--random sampling -> using shuffle ]]
