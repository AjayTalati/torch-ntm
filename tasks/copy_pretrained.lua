--[[ Load a saved NTM model which has previously been trained with copy.lua, and demo it

To save a model after it's been trained, add these lines 

filename = 'tasks/trained_model_for_copy_task.dat'
torch.save(filename, model)

to the bottom of copy.lua The NTM model will be saved as a binary file in the tasks folder of the torch-ntm project 

USEAGE

For the copy task with a pretrained model:

cd /home/ajay/TorchProjects/ntm-dev
th tasks/copy_with_a_trained_model.lua

]]

require('../') -- load the NTM module

--model = torch.load('/home/ajay/TorchProjects/ntm-dev/tasks/trained_model.dat') -- load the trained model
model = torch.load('tasks/trained_model_for_copy_task.dat') 

--[[ A task specific function -- generate binary vectors ]]
function generate_sequence(len, bits)
  local seq = torch.zeros(len, bits + 2)
  for i = 1, len do
    seq[{i, {3, bits + 2}}] = torch.rand(bits):round()
  end
  return seq
end

--[[ A simple forward propagation function ]]
function forward(model, seq)
  local len = seq:size(1)
  local zeros = torch.zeros(input_dim)
  local outputs = torch.Tensor(len, input_dim)
  model:forward(start_symbol)                  -- present start symbol
  for j = 1, len do                            -- present inputs
    model:forward(seq[j]) 
  end
  model:forward(end_symbol)                    -- present end symbol
  for j = 1, len do
    outputs[j] = model:forward(zeros)          -- calculate the outputs after presenting the inputs 
  end
  return outputs
end

--[[ define the start and stop symbols ]]
input_dim = model.input_dim
start_symbol = torch.zeros(input_dim)
end_symbol = torch.zeros(input_dim)

start_symbol[1] = 1
end_symbol[2] = 1

--[[ A simple copy test ]]
min_len = 1
max_len = 20

len = math.floor(torch.random(min_len, max_len)) -- a random length
seq = generate_sequence(len, input_dim - 2)

outputs = forward(model, seq)

print("target:")
print(seq)
print("outputs:")
print(outputs)

--[[ Produce some graphs - unfortunately these are quite useless]]
master_cell = model.master_cell
graph.dot(master_cell.fg, 'master_cell_forward_graph', 'NTM_master_cell_forward_graph')
graph.dot(master_cell.bg, 'master_cell_backward_graph', 'NTM_master_cell_backward_graph')

initial_module = model.init_module
graph.dot(initial_module.fg, 'initial_module_forward_graph', '/home/ajay/Desktop/NTM_initial_module_forward_graph')
graph.dot(initial_module.bg, 'initial_module_backward_graph', '/home/ajay/Desktop/NTM_initial_module_backward_graph')
