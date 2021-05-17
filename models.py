import numpy as np
import torch
import torch.nn as nn

class MemoryLayer(nn.Module):
    def __init__(self, input_dim, memory_dim, model_dim, mlp_dim, dropout_p):
        super(MemoryLayer, self).__init__()
        
        # Hyperparameters
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.model_dim = model_dim
        self.mlp_dim = mlp_dim
        
        # Parameters
        self.W_q = nn.Linear(input_dim, model_dim)
        self.W_k = nn.Linear(memory_dim, model_dim)
        self.W_v = nn.Linear(memory_dim, model_dim)
        self.lin1 = nn.Linear(model_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        
    def forward(self, x, m):
        # Generate queries, keys, values
        Q = self.W_q(x) # [batch, n_test, model_dim]
        K = self.W_k(m) # [batch, n_memories, model_dim]
        V = self.W_v(m) # [batch, n_memories, model_dim]
        
        # Get attention distributions over memories for each sample
        attn = torch.matmul(Q, K.permute(0,2,1)) # [batch, n_test, n_memories]
        attn = attn/np.sqrt(self.model_dim)
        attn = self.softmax(attn) # [batch, n_test, n_memories]
        
        # Get weighted average of values
        V_bar = torch.matmul(attn, V) # [batch, n_test, model_dim]
        
        # Feedforward
        out = self.lin1(V_bar) # [batch, n_test, mlp_dim]
        out = self.dropout(out)
        out = self.relu(out)   # [batch, n_test, mlp_dim]
        out = self.lin2(out)   # [batch, n_test, input_dim]
        
        return out, attn
        
class EpisodicSystem(nn.Module):
    def __init__(self):
        super(EpisodicSystem, self).__init__()

        # Hyperparameters
        self.n_states = 16    # number of faces in 4x4 grid
        self.axis_dim = 2     # dimension of axis (2d one-hot vectors)
        self.y_dim = 1        # dimension of y (binary)
        self.model_dim = 32   # dimension of Q, K, V
        self.mlp_dim = 64     # dimension of mlp hidden layer
        self.n_layers = 1     # number of layers
        self.dropout_p = 0.0  # dropout probability
        self.input_dim = 2*self.n_states + self.axis_dim # y not given in input
        self.memory_dim = self.input_dim + self.y_dim    # y given in memories
        self.output_dim = 2   # number of choices (binary)
        
        # Memory system
        memory_layers = []
        for l_i in range(self.n_layers):
            layer = MemoryLayer(self.input_dim, self.memory_dim, self.model_dim, 
                                self.mlp_dim, self.dropout_p)
            memory_layers.append(layer)
        self.memory_layers = nn.ModuleList(memory_layers)
        
        # Output
        self.lin1 = nn.Linear(2*self.input_dim, self.mlp_dim)
        self.lin2 = nn.Linear(self.mlp_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_p)
        
    def forward(self, x, m):
        out = x
        # Memory system
        attention = []
        for l_i in range(self.n_layers):
            out, attn = self.memory_layers[l_i](out, m) 
            # out = [batch, n_test, model_dim]
            # attn = [batch, n_test, n_memories]
            attention.append(attn.detach().cpu().numpy())
        
        # MLP
        out = torch.cat([x, out], dim=2) # [batch, n_test, 2*input_dim]
        out = self.lin1(out) # [batch, n_test, mlp_dim]
        out = self.dropout(out) # [batch, n_test, mlp_dim]
        out = self.relu(out) # [batch, n_test, mlp_dim]
        out = self.lin2(out) # [batch, n_test, out_dim]
        
        return out, attention


class CNN(nn.Module):
    def __init__(self, state_dim):
        super(CNN, self).__init__()

        # Hyperparameters
        self.state_dim = state_dim  # size of final embeddings
        self.image_size = 64        # height and width of images
        self.in_channels = 1        # channels in inputs (grey-scaled)
        self.kernel_size = 3        # kernel size of convolutions
        self.padding = 0            # padding in conv layers
        self.stride = 2             # stride of conv layers
        self.pool_kernel = 2        # kernel size of max pooling
        self.pool_stride = 2        # stride of max pooling
        self.out_channels1 = 4      # number of channels in conv1
        self.out_channels2 = 8      # number of channels in conv2
        self.num_layers = 2         # number of conv layers

        # Conv layers
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels1, 
                               self.kernel_size, self.stride, self.padding)
        self.maxpool1 = nn.MaxPool2d(self.pool_kernel, self.pool_stride)

        self.conv2 = nn.Conv2d(self.out_channels1, self.out_channels2, 
                               self.kernel_size, self.stride, self.padding)
        self.maxpool2 = nn.MaxPool2d(self.pool_kernel, self.pool_stride)

        # Linear layer
        self.cnn_out_dim = self.calc_cnn_out_dim()
        self.linear = nn.Linear(self.cnn_out_dim, self.state_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv 1
        x = self.conv1(x)          # [batch, 4, 31, 31]
        x = self.relu(x)           # [batch, 4, 31, 31]
        x = self.maxpool1(x)       # [batch, 4, 15, 15]

        # Conv 2
        x = self.conv2(x)          # [batch, 8, 7, 7]
        x = self.relu(x)           # [batch, 8, 7, 7]
        x = self.maxpool2(x)       # [batch, 8, 3, 3]

        # Linear
        x = x.view(x.shape[0], -1) # [batch, 72]
        x = self.linear(x)         # [batch, 32]
        
        return x
        
    def calc_cnn_out_dim(self):
        w = self.image_size
        h = self.image_size 
        for l in range(self.num_layers):
            new_w = np.floor(((w - self.kernel_size)/self.stride) + 1)
            new_h = np.floor(((h - self.kernel_size)/self.stride) + 1)
            new_w = np.floor(new_w / self.pool_kernel)
            new_h = np.floor(new_h / self.pool_kernel)
            w = new_w
            h = new_h
        return int(w*h*8)


class CorticalSystem(nn.Module):
    def __init__(self, use_images):
        super(CorticalSystem, self).__init__()
        self.use_images = use_images

        # Hyperparameters
        self.n_states = 16
        self.state_dim = 32
        self.mlp_in_dim = 3*self.state_dim # (f1 + f2 + axis)
        self.hidden_dim = 128
        self.output_dim = 2

        # Input embedding (images or one-hot)
        if self.use_images:
            self.face_embedding = CNN(self.state_dim)
        else:
            self.face_embedding = nn.Embedding(self.n_states, self.state_dim)
            nn.init.xavier_normal_(self.face_embedding.weight)
            
        self.axis_embedding = nn.Embedding(2, self.state_dim)
        nn.init.xavier_normal_(self.axis_embedding.weight)

        # MLP
        self.linear1 = nn.Linear(self.mlp_in_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, f1, f2, ax):

        # Embed inputs
        f1_embed = self.face_embedding(f1) # [batch, state_dim]
        f2_embed = self.face_embedding(f2) # [batch, state_dim]
        ax_embed = self.axis_embedding(ax) # [batch, state_dim]
        
        # MLP
        x = torch.cat([f1_embed, f2_embed, ax_embed], dim=1) 
        # x: [batch, 3*state_dim]
        x = self.linear1(x) # [batch, hidden_dim]
        x = self.relu(x)    # [batch, hidden_dim]
        x = self.linear2(x) # [batch, output_dim]

        return x

