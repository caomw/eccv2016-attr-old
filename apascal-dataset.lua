require 'pl'
require 'image'
-- stringx.import()

local Dataset = torch.class('Dataset')

function load_txt(filename,image_dir)
    local data = {}
    local txt = io.open(filename)
    for line in txt:lines() do
        line = line:split(' ')

        local instance = {}
        instance.fileloc = image_dir..line[1]
        instance.class = line[2]
        instance.crop = {lu = {x = tonumber(line[3]), y = tonumber(line[4]) },
                         rl = {x = tonumber(line[5]), y = tonumber(line[6]) } }
        instance.attribute = {}
        for i=7,70 do
            instance.attribute[i-6] = tonumber(line[i])
        end
        data[#data+1] = instance
    end
    txt:close()
    return data
end
function Dataset:__init(path,image_dir,mean,std,batch_size)
    self.train_data = load_txt(path..'apascal_train.txt',image_dir)
    self.val_data = load_txt(path..'apascal_test.txt',image_dir)
    self.mean = mean
    self.std = std
    self.batch_size = batch_size
end

function Dataset:get_image_attribute(data,index)
    if( #data < index ) then
        print 'index error'
    end

    instance = data[index]
    if( instance.resized == nil ) then
        local im = image.load(instance.fileloc)
        local cropped_image = image.crop(im, instance.crop.lu.x, instance.crop.lu.y, instance.crop.rl.x, instance.crop.rl.y )
        local resized = image.scale(cropped_image,224)
        local preprocessed = resized:add(-self.mean:resize(3,1,1):expandAs(resized):mul(1/self.std))
        instance.resized = torch.Tensor(3,224,224):zero()
        instance.preprocessed = torch.Tensor(3,224,224):zero()

        top_margin = ( 224 - resized:size(2) ) / 2 + 1
        left_margin = ( 224 - resized:size(3) ) / 2 + 1
        instance.resized:sub(1, 3, top_margin, top_margin+resized:size(2)-1 , left_margin, left_margin+resized:size(3)-1):copy(resized)
        instance.preprocessed:sub(1, 3, top_margin, top_margin+resized:size(2)-1 , left_margin, left_margin+resized:size(3)-1):copy(preprocessed)
    end
    return instance.preprocessed, instance.attribute
end

function Dataset:size()
    return #self.train_data, #self.val_data
end
function Dataset:get_samples(train_or_val)
    local inputs = torch.Tensor(self.batch_size, 3, 224, 224):zero()
    local labels = {}
    for i = 1,64 do
        labels[i] = torch.Tensor(self.batch_size):zero()
    end

    local data
    if( train_or_val == 'train' ) then
        data = self.train_data
    else
        data = self.val_data
    end

    for i = 1,self.batch_size do
        local im, attr = self:get_image_attribute(data, math.floor(math.random() * #self.train_data) + 1 )
        inputs:select(1,i):copy( im )
        for label_i = 1,64 do
            labels[label_i][i] = attr[label_i]
        end
    end
    return inputs, labels
end

function Dataset:get_valid_sample_co()
    local co = coroutine.create( function(this)
        for i = 1, #this.val_data, this.batch_size do
            local remain = math.min(this.batch_size, #this.val_data-i)

            local inputs = torch.Tensor(remain, 3, 224, 224):zero()
            local labels = {}
            for label_i = 1,64 do
                labels[label_i] = torch.Tensor(remain):zero()
            end

            for j = 1,remain do
                local im, attr = this:get_image_attribute(this.val_data, i+j-1)
                inputs:select(1,j):copy(im)
                for label_i = 1,64 do
                    labels[label_i][j] = attr[label_i]
                end
            end

            coroutine.yield( inputs, labels )
        end
    end
    )
    return co
end
