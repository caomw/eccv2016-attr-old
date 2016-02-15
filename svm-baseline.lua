require 'apascal-dataset'
require 'loadcaffe'
require 'cutorch'
require 'logger'

opt = lapp[[
      --device          (default 1)        Gpu Device to use
      --batchSize       (default 10)      Sub-batch size
      --dataRoot        (default ./dataset/attribute/attribute_data/)        Data root folder
      --imageRoot       (default ./dataset/attribute/apascal_images/)       Image dir
      --experimentName  (default "snapshots/svm/")
]]

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)

cutorch.setDevice(opt.device)
cutorch.manualSeedAll(1)

--model = loadcaffe.load('caffe_models/VGG_ILSVRC_19_layers_deploy.prototxt'
--                       ,'caffe_models/VGG_ILSVRC_19_layers.caffemodel'
--                       , 'cudnn')
--mean = torch.Tensor(1,3,224,224):float()
--mean:sub(1,1,1,1):fill(103.939)
--mean:sub(1,1,2,2):fill(116.779)
--mean:sub(1,1,3,3):fill(123.68)

logger = Logger()
if( opt.loadFrom ~= "" ) then
    logger:open( opt.experimentName, true )
else
    logger:open( opt.experimentName, false )
end
logger:info( opt )

dataset = Dataset( opt.dataRoot,
                   opt.imageRoot,
                   torch.Tensor({0.0,0,0,0,0}),1.0
                   ,opt.batchSize )
train_size, val_size = dataset:size()

balance_weights = {}
local txt = io.open('dataset/attribute/attribute_data/balancing_weights.txt')
for line in txt:lines() do
    line = line:split('\t')
    balance_weights[#balance_weights+1] = {pl = tonumber(line[1]), nl = tonumber(line[2])}
end
txt:close()

---------------------------------------
-- Feature extraction : Training images
---------------------------------------
--local co = dataset:get_train_sample_co()
--train_num, valid_num = dataset:size()
--features = torch.Tensor(train_num,4096):float()
--labels = {}
--for label_i = 1,64 do
--    labels[label_i] = torch.Tensor(train_num):zero():float()
--end
--
--index = 1
--while true do
--    flag,ims,target_labels = coroutine.resume(co,dataset)
--    if( ims == nil ) then
--        break
--    end
--
--    processed = ims*255 - torch.expand(mean,ims:size()[1],3,224,224)
--
--    collectgarbage();collectgarbage();
--    ret = model:forward(processed:cuda())
--
--    features:sub(index,index + ims:size()[1]-1):copy(model['modules'][40].output:float()) --FC7 features
--    for label_i = 1,64 do
--        labels[label_i]:sub(index,index + ims:size()[1]-1):copy(target_labels[label_i])
--    end
--    index = index + ims:size()[1]
--
--    xlua.progress(index,train_num)
--end
--torch.save( opt.experimentName..'train-fc7-features.th', features )
--torch.save( opt.experimentName..'train-attribute-labels.th', labels )

-----------------------------------------
-- Save to appropriate format for SVM : Training Images
-----------------------------------------
--train_num, valid_num = dataset:size()
--features = torch.load( opt.experimentName..'/train-fc7-features.th')
--labels = torch.load( opt.experimentName.. '/train-attribute-labels.th')
--
--files = {}
--for i = 1,64 do
--    file = io.open( opt.experimentName..'attr'..tostring(i)..'.txt','w+')
--    files[i] = file
--end
--for i = 1, train_num do
--    local t = {}
--    for j = 1,4096 do
--        table.insert(t,tostring(j)..':'..tostring(features[i][j])..' ')
--    end
--    s = table.concat(t,'')..'\n'
--    for label_i = 1, 64 do
--        if( labels[label_i][i] == 0 ) then
--            files[label_i]:write('-1'..' '..s)
--        else
--            files[label_i]:write('+1'..' '..s)
--        end
--    end
--    xlua.progress(i,train_num)
--end
--for i = 1,64 do
--    files[i]:close()
--end

---------------------------------------
-- Feature extraction : Validation set images
---------------------------------------
--local co = dataset:get_valid_sample_co()
--train_num, valid_num = dataset:size()
--features = torch.Tensor(valid_num,4096):float()
--labels = {}
--for label_i = 1,64 do
--    labels[label_i] = torch.Tensor(valid_num):zero():float()
--end
--
--index = 1
--while true do
--    flag,ims,target_labels = coroutine.resume(co,dataset)
--    if( ims == nil ) then
--        break
--    end
--
--    processed = ims*255 - torch.expand(mean,ims:size()[1],3,224,224)
--
--    collectgarbage();collectgarbage();
--    ret = model:forward(processed:cuda())
--
--    features:sub(index,index + ims:size()[1]-1):copy(model['modules'][40].output:float()) --FC7 features
--    for label_i = 1,64 do
--        labels[label_i]:sub(index,index + ims:size()[1]-1):copy(target_labels[label_i])
--    end
--    index = index + ims:size()[1]
--
--    xlua.progress(index,valid_num)
--end
--torch.save( opt.experimentName..'valid-fc7-features.th', features )
--torch.save( opt.experimentName..'valid-attribute-labels.th', labels )

-----------------------------------------
-- Save to appropriate format for SVM : Validation set Images
-----------------------------------------
--train_num, valid_num = dataset:size()
--features = torch.load( opt.experimentName..'/valid-fc7-features.th')
--labels = torch.load( opt.experimentName.. '/valid-attribute-labels.th')

--files = {}
--for i = 1,64 do
--    file = io.open( opt.experimentName..'valid-attr'..tostring(i)..'.txt','w+')
--    files[i] = file
--end
--for i = 1, valid_num do
--    local t = {}
--    for j = 1,4096 do
--        table.insert(t,tostring(j)..':'..tostring(features[i][j])..' ')
--    end
--    s = table.concat(t,'')..'\n'
--    for label_i = 1, 64 do
--        if( labels[label_i][i] == 0 ) then
--            files[label_i]:write('-1'..' '..s)
--        else
--            files[label_i]:write('+1'..' '..s)
--        end
--    end
--    xlua.progress(i,valid_num)
--end
--for i = 1,64 do
--    files[i]:close()
--end

---------------------------------------------------
-- Run SVM
---------------------------------------------------

true_positive = torch.Tensor(64):zero()
true_negative = torch.Tensor(64):zero()
false_positive = torch.Tensor(64):zero()
false_negative = torch.Tensor(64):zero()

require 'svm'
for label_i = 1,64 do
    collectgarbage(); collectgarbage();
    train_data = svm.ascread(opt.experimentName..'attr'..tostring(label_i)..'.txt')
    valid_data = svm.ascread(opt.experimentName..'valid-attr'..tostring(label_i)..'.txt')
    model = liblinear.train(train_data) -- libsvm? liblinear?
    labels,accuracy,dec = liblinear.predict(valid_data,model)

    --todo : labels is tensor...
    for i = 1, labels:size()[1] do
        if( labels[i] > 0 ) then
            if( valid_data[i][1] > 0 ) then
                true_positive[label_i] = true_positive[label_i] + 1
            else
                false_positive[label_i] = false_positive[label_i] + 1
            end
        else
            if( valid_data[i][1] > 0 ) then
                false_negative[label_i] = false_negative[label_i] + 1
            else
                true_negative[label_i] = true_negative[label_i] + 1
            end
        end
    end

    xlua.progress(label_i,64)
end

logger:eval( {total = labels:size()[1], tp = torch.totable(true_positive), tn = torch.totable(true_negative), fp = torch.totable(false_positive), fn = torch.totable(false_negative) } )

