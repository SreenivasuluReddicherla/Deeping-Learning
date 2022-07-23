#!/usr/bin/env python
# coding: utf-8

# # Tensor
Tensors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.
# In[20]:


# importing modules
import torch
import numpy as np

Initialization of tensor 
Tensor can be initialiaze in 3 different ways 
    1 . Directly from the data
    2 . From numpy array
    3 . From another tensor
# ## 1 . Directly from the data

# In[2]:


data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)
print(type(data))
print(type(x_data))


# ## 2 . From numpy array

# In[5]:


data1 = [[5,6],[7,8]]
print(data1)
print(type(data1))
np_array = np.array(data1)
print(np_array)
print(type(np_array))
y_data = torch.from_numpy(np_array)
print(y_data)
print(type(y_data))


# ## 3 . Tensor from another tensor

# In[14]:


ones = torch.ones_like(y_data)
print(ones)
zeros = torch.zeros_like(ones)
print(zeros)
rn_val = torch.rand_like(zeros,dtype=float)
print(rn_val)


# ## Shape= ( ) is tuple of tensor dimensions
# ## with constant or random tensors for given shape

# In[32]:


shape=(3,4)
shapes = (3,4,)
r_ten = torch.rand(shape)
print(r_ten)
r_tens = torch.rand(shapes)
print(r_tens)
o_ten = torch.ones(shape)
print(o_ten)
o_tens = torch.ones(shapes)
print(o_tens)
z_ten = torch.zeros(shape)
print(z_ten)
z_tens = torch.ones(shapes)
print(o_tens)


# ## Attributes of a tensor

# In[39]:


new_tensor = torch.rand(3,4)
print(new_tensor)
print(new_tensor.shape)
print(new_tensor.dtype)
print(new_tensor.device)


# ## Operations on tensors

# In[ ]:




