json = require 'cjson'
local Logger = torch.class('Logger')

function Logger:__init()
end

function Logger:open(path,append)
    flag = append and 'a' or 'w+'
    self.t_file = io.open(path..'training.log',flag)

   local d = os.date("*t",os.time())
   local e_logname = string.format("%04d%02d%02d-%02d.%02d.log",
      d.year, d.month, d.day, d.hour, d.min )
    self.e_file = io.open(path..e_logname,'w+')
end

function Logger:close()
    self.t_file:close()
    self.e_file:close()
end

function Logger:_encode( type_,t )
    local c = {}

    local d = os.date("*t",os.time())
    c.time = string.format("%04d%02d%02d-%02d:%02d", d.year, d.month, d.day, d.hour, d.min )
    c.type = type_
    c.contents = t

    return json.encode(c)..'\n'
end

function Logger:info( t )
    print(t)
    self.e_file:write( self:_encode('info',t) )
    self.e_file:flush()
end

function Logger:eval( t )
    c = self:_encode( 'eval',t )

    self.e_file:write( c )
    self.e_file:flush()
    self.t_file:write( c )
    self.t_file:flush()
end

function Logger:train( t )
    c = self:_encode( 'train',t )

    self.e_file:write( c )
    self.e_file:flush()
    self.t_file:write( c )
    self.t_file:flush()
end
