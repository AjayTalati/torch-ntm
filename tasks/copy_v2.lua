--[[

  Training a NTM to memorize input.

  The current version seems to work, giving good output after 5000 iterations
  or so. Proper initialization of the read/write weights seems to be crucial
  here.

--]]

require('optim')

require('../')

--require('mobdebug').start()

torch.manualSeed(0)

--[[ variables for training loop, bookeeping, optimizer(rmsprop) ]]

local num_iters = 10000
local print_interval = 25
local min_len = 1
local max_len = 20

--[[ function to generate input data ]]
function generate_sequence(len, bits)
  local seq = torch.zeros(len, bits + 2)
  for i = 1, len do
    seq[{i, {3, bits + 2}}] = torch.rand(bits):round() -- cols 1 & 2 are reserved for the start & end symbols
  end
  return seq
end

-- [[ optimizer config ]]
local rmsprop_state = {
  learningRate = 1e-4,
  momentum = 0.9,
  decay = 0.95
}

-- [[ NTM config ]]
local config = {
  input_dim = 10,
  output_dim = 10,
  mem_rows = 128,
  mem_cols = 20,
  cont_dim = 100
}

--[[ symbols to present to the ntm to define starting and stopping of tasks ]]
local input_dim = config.input_dim
local start_symbol = torch.zeros(input_dim)
local end_symbol = torch.zeros(input_dim)

start_symbol[1] = 1
end_symbol[2] = 1

--[[ forward propagation function ]]
function forward(model, seq, print_flag)
  
  local len = seq:size(1)
  local loss = 0
  local zeros = torch.zeros(input_dim)
  local outputs = torch.Tensor(len, input_dim)
  local criteria = {}

  model:forward(start_symbol)                  -- present start symbol
  for j = 1, len do model:forward(seq[j]) end  -- present inputs
  model:forward(end_symbol)                    -- present end symbol

  -- calculate outputs and loss criteria
  for j = 1, len do
    criteria[j] = nn.BCECriterion()
    outputs[j] = model:forward(zeros) -- what does this do
    loss = loss + criteria[j]:forward(outputs[j], seq[j]) * input_dim
  end

  return outputs, criteria, loss
end

--[[ backward propagation function ]]
function backward(model, seq, outputs, criteria)
  local len = seq:size(1)
  local zeros = torch.zeros(input_dim)

  for j = len, 1, -1 do model:backward( zeros, criteria[j]:backward(outputs[j], seq[j]):mul(input_dim) ) end 

  model:backward(end_symbol, zeros)
  for j = len, 1, -1 do model:backward(seq[j], zeros) end
  model:backward(start_symbol, zeros)

end

--[[ initialize a neural Turing machine model and get it's controllers parameters and gradients ]]
local model = ntm.NTM(config)
local params, grads = model:getParameters()

--[[ training loop ]]
for iter = 1, num_iters do
  
  local feval = function(x)

    grads:zero() -- fill the grads tensor with zeros

    local len = math.floor(torch.random(min_len, max_len))
    local seq = generate_sequence(len, input_dim - 2)

    local outputs, criteria, loss = forward(model, seq) -- forward propagation
    backward(model, seq, outputs, criteria) -- backward propagation
    
    local print_flag = (iter % print_interval == 0)
    if print_flag then print('loss = ' .. loss .. ' .. iter = ' .. iter) end

    grads:clamp(-10, 10) -- clip gradients 
    return loss, grads 
  end

  ntm.rmsprop(feval, params, rmsprop_state)
end

--[[ save the trained model ]]
filename = 'tasks/trained_model_for_copy_task.dat'
torch.save(filename, model)
