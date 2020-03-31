import torch.nn as nn
from torch.autograd import Variable
import torch

class Conv2DRNNCell(nn.Module):

  def __init__(self, input_dim, hidden_dim, output_dim, kernel_size, padding, bias=True):
    """
    Initialize Conv2DRNN cell.
    
    Parameters
    ----------
    input_dim: int
      Number of channels of input tensor.
    hidden_dim: int
      Number of channels of hidden state.
    output_dim: int
      Number of channels of output tensor.
    kernel_size: (int, int)
      Size of the convolutional kernel.
    padding: (int, int)
      Size of the padding.  
    bias: bool
      Whether or not to add the bias.
    """
    super(Conv2DRNNCell, self).__init__()

    self.input_dim  = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim

    self.kernel_size = kernel_size
    self.padding     = padding
    self.bias        = bias
        
    self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                          out_channels=self.hidden_dim + self.output_dim,
                          kernel_size=self.kernel_size,
                          padding=self.padding,
                          bias=self.bias)
    self.activation = nn.LeakyReLU(0.1)
                          
  def step(self, input_tensor, hidden_state):
    combined = torch.cat([input_tensor, hidden_state], dim=1)  # concatenate along channel axis
        
    combined_conv = self.conv(combined)
    h, o = torch.split(combined_conv, [self.hidden_dim, self.output_dim], dim=1)

    hidden_next = self.activation(h)
    output_tensor = self.activation(o)
        
    return hidden_next, output_tensor

  # Inputs here is [batch_size, num_inputs, input_features, height, width]    
  def forward(self, inputs, hidden=None):
    b, steps, _, h, w = inputs.size()

    if hidden is not None:
      raise NotImplementedError()
    else:
      # Since the init is done in forward. Can send image size here
      hidden = self.init_hidden(b, h, w)

    outputs = []

    for i in range(steps):
      hidden, output = self.step(inputs[:,i,:,:,:], hidden)
      outputs.append(output)
    return hidden, torch.stack(outputs, 1)

  def init_hidden(self, batch_size, height, width):
    return Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda()
            
class Conv2DRNN(nn.Module):
  """
  Builds a set of Conv2DRNNCell layers.
  rnn_hid_dim, hidden_dim, and kernel_size are all tuples/lists of length num_layers.
  
  The idea is that the input tensor is expected to be 5D,
    batch size * num frames * channels * height * width.
  Intermediate hidden layers are
    batch size * num frames * hidden_dim[i] * height * width,
  with hidden states of the RNN layers being
    batch size * num frames * rnn_hid_dim[i] * height * width.
  The output is the last "hidden layer", a tensor of shape
    batch size * num frames * hidden_dim[-1] * height * width
  """
  def __init__(self, input_size, input_dim, rnn_hid_dim, hidden_dim, kernel_size, num_layers,
               bias=True):
    super(Conv2DRNN, self).__init__()

    self._check_kernel_size_consistency(kernel_size)

    # Make sure that both `kernel_size`, `rnn_hid_dim', and `hidden_dim` are lists having len == num_layers
    kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
    rnn_hid_dim = self._extend_for_multilayer(rnn_hid_dim, num_layers)
    hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
    if not len(kernel_size) == len(hidden_dim) == len(rnn_hid_dim) == num_layers:
      raise ValueError('Inconsistent list length.')

    self.height, self.width = input_size

    self.input_dim  = input_dim
    self.hidden_dim = hidden_dim
    self.kernel_size = kernel_size
    self.num_layers = num_layers
    self.bias = bias

    cell_list = []
    for i in range(0, self.num_layers):
      cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

      cell_list.append(Conv2DRNNCell(input_dim=cur_input_dim,
                                     hidden_dim=self.rnn_hid_dim[i],
                                     output_dim=self.hidden_dim[i],
                                     kernel_size=self.kernel_size[i],
                                     bias=self.bias))

    self.cell_list = nn.ModuleList(cell_list)

  def forward(self, input_tensor, hidden_state=None):
    """
        
    Parameters
    ----------
    input_tensor:
      5-D Tensor of shape (b, t, c, h, w)
    hidden_state: todo
            
    Returns
    -------
    last_state_list, layer_output
    """

    layer_output_list = []
    last_state_list   = []

    seq_len = input_tensor.size(1)
    cur_layer_input = input_tensor

    for layer_idx in range(self.num_layers):
      h, c = hidden_state[layer_idx]
      output_inner = []
      for t in range(seq_len):
        h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                         cur_state=[h, c])
        output_inner.append(h)

      layer_output = torch.stack(output_inner, dim=1)
      cur_layer_input = layer_output

      layer_output_list.append(layer_output)
      last_state_list.append([h, c])

    if not self.return_all_layers:
      layer_output_list = layer_output_list[-1:]
      last_state_list   = last_state_list[-1:]

    return layer_output_list, last_state_list

  def _init_hidden(self, batch_size):
    init_states = []
    for i in range(self.num_layers):
      init_states.append(self.cell_list[i].init_hidden(batch_size, self.height, self.width))
    return init_states
        
  @staticmethod
  def _check_kernel_size_consistency(kernel_size):
    if not (isinstance(kernel_size, tuple) or
              (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
      raise ValueError('`kernel_size` must be tuple or list of tuples')  

  @staticmethod
  def _extend_for_multilayer(param, num_layers):
    if not isinstance(param, list):
      param = [param] * num_layers
    return param

