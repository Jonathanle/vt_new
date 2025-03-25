"""
TODO:
Create a model that evaluates the risk of a patient with VA, 
Input (
"""
# feelings - without this higher leve conscoius, less executive capacity at the level, conscoiusness is not only associated with a higher level but it feelas thaata there is aa dimension of executiveness that it has alongside being more associated with a lower temporal frame.
# a "contextual" frame???
# seems as though the "narrative" aspect of needing me to have a conscoius is very useful to contextulaize my actions in a larger frame but it seems relevant because in this linguistic space that I create with the narrative, i allow to relate to other elements.

import torch
import torch.nn as nn
import torch.nn.functional as F

class VANet(nn.Module):
    def __init__(self, in_channels=2, hidden_dims=[16, 32, 64], fc_dims=[512, 256], output_dim=1):
        """
        Neural network for processing 3D medical image data with parallel slice processing.
        
        Args:
            in_channels (int): Number of input channels per slice (default: 2)
            hidden_dims (list): List of hidden dimensions for convolutional layers
            fc_dims (list): List of hidden dimensions for fully connected layers
            output_dim (int): Dimension of the output
        """
        """
        TODO: Investigate changes in the model for preventing overfitting in the dataset
        - can i reduce kernel size so that I have more dimensions? 


        - TODO - investigate how model processes the slices



        """



        super(VANet, self).__init__()
        
        # Convolutional layers for processing each slice
        self.conv_layers = nn.ModuleList()
        
        # First convolutional layer
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ))
        
        for i in range(len(hidden_dims) - 1):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ))



        # Calculate the size of flattened features after convolutions
        # After 3 max pooling layers with stride 2, a 128x128 image becomes 16x16
        flattened_size = hidden_dims[-1] * (128 // (2 ** len(hidden_dims))) * (128 // (2 ** len(hidden_dims)))
        
        # Fully connected layers for processing the flattened features
        self.fc_layers = nn.ModuleList()
        
        # First FC layer
        self.fc_layers.append(nn.Sequential(
            nn.Linear(flattened_size, fc_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.5)
        ))
        
        # Additional FC layers
        for i in range(len(fc_dims) - 1):
            self.fc_layers.append(nn.Sequential(
                nn.Linear(fc_dims[i], fc_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.5)
            ))
        
        # Output layer
        self.output_layer = nn.Linear(fc_dims[-1], output_dim)
        
        # Slice aggregation - for combining the features from all slices
        self.slice_aggregation = nn.Sequential(
            nn.Linear(output_dim * 1, output_dim),  # Assuming num_slices=1 for now
            nn.ReLU()
        )
       
            
        self.max_slice = 10
        self.linear_final = nn.Linear(self.max_slice, 1)


        self.sigmoid = torch.sigmoid


    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, num_slices, height, width)
                              or in your case (1, 2, num_slices, 128, 128)
        
        Returns:
            torch.Tensor: Output tensor
        """
        batch_size, channels, num_slices, height, width = x.shape
       
        assert batch_size == 1, "Batch size must be 1 as model training divides the batches"


        # Reshape to process all slices in parallel
        # New shape: (batch_size * num_slices, channels, height, width)
        x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, channels, height, width) # contiguous == prevent new creation; view -- subjectively create new view into accessing tensor  # TODO: document continguous array purpose 

        #TODO after validating, encode the slice number into the slice thorugh maybe the training loop? so that we can contextulaize the encodings.



        # Run 2d Image through the layer 
        for conv_layer in self.conv_layers:
            x_reshaped = conv_layer(x_reshaped)
        
        # Flatten for fully connected layers
        x_flattened = x_reshaped.view(x_reshaped.size(0), -1)
      
        # TODO: pass in an ordinal input later

        # Pass through fully connected layers
        for fc_layer in self.fc_layers:
            x_flattened = fc_layer(x_flattened)
       


        # at this point gives a outptut for each slice an embedded representation


        # Final output for each slice
        slice_outputs = self.output_layer(x_flattened)
        evals = torch.zeros((10, 1), device=slice_outputs.device)

        n_slices = slice_outputs.shape[0]

        # Determine how many slices to actually use (min of n_slices or max_size)
        actual_slices = min(n_slices, self.max_slice)
   
        # Populate the evals tensor with available slice outputs
        evals[:actual_slices, :] = slice_outputs[:actual_slices, :]


        evals = evals.transpose(0, 1)
        logit = self.linear_final(evals)
        

#            
#        # Reshape back to separate the slices
#        # New shape: (batch_size, num_slices, output_dim)
#        slice_outputs = slice_outputs.view(batch_size, num_slices, -1)
#        
#        breakpoint() 
#
#        # Transform the output by weighting all* slices equally (TODO: eval determining later) 
#        # how can i allow ordinal encoding thata is dynamic?
#        # I asaked this becaause the slices are variable, so no fixed weighing system is o
#        # have a network process a "max" number of slices? - we can do some padding?
#
#
#        # no need for ordinal encoding? this is because when i process the slices that the placement in the network, already helps with how it processes.
#        evals = torch.zeros((10, 1))
#
#
#
#
#
#
#        aggregated_output = torch.mean(slice_outputs, dim=1) 
        sigmoid_output = torch.sigmoid(logit) 

        
        return sigmoid_output

# Example usage
def test_vanet():
    # Create a random input tensor of shape (1, 2, num_slices, 128, 128)
    num_slices = 10
    x = torch.randn(1, 2, num_slices, 128, 128)
    
    # Initialize the model
    model = VANet(in_channels=2, output_dim=10)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
   
    return output

# Uncomment to test
# test_vanet()
