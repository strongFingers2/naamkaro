import streamlit as st
import torch
import torch.nn.functional as F
import pickle
class Linear:
  def __init__(self, fanin, fanout, bias=True):
    self.weight = torch.randn((fanin, fanout), generator = g)
    self.bias = torch.zeros(fanout) if bias else None

  def __call__(self, x):
    self.x = x @ self.weight
    if self.bias is not None:
      self.x += self.bias
    return self.x

  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNormIn1D:
  def __init__(self, dim, eps = 1e-5, momentum = 0.1):
    self.eps = eps
    self.training = True
    #parameters which will be trained with backpropagation
    self.gamma = torch.ones((dim))
    self.beta = torch.zeros((dim))
    #parameters which will not be trained with backpropagation
    #We calculate the running mean and running standard deviation
    self.running_mean = torch.zeros((dim))
    self.running_var = torch.ones((dim))
    self.momentum = momentum

  def __call__(self, x):
    if self.training:
      xmean = x.mean(dim=0, keepdim=True)
      xvar = x.var(dim = 0, keepdim=True)
    else:
      xmean = self.running_mean
      xvar = self.running_var
    self.out = (self.gamma)*(x - xmean) / (torch.sqrt(xvar + self.eps)) + self.beta
    #update the buffer parameters
    if self.training:
      self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*xmean
      self.running_var = (1 - self.momentum)*(self.running_var) + self.momentum*xvar
    return self.out

  def parameters(self):
    return [self.beta, self.gamma]

class tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out

  def parameters(self):
    return []

class Sequential:
  def __init__(self, Layers):
    self.Layers = Layers

  def __call__(self, x):
    for layer in self.Layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.Layers for p in layer.parameters()]

class Embedding:
  def __init__(self, num_embeddings, embedding_dim):
    self.weight = torch.randn((num_embeddings, embedding_dim))

  def __call__(self, IX):
    self.out = self.weight[IX]
    return self.out

  def parameters(self):
    return [self.weight]

class Flatten:
  def __call__(self, x):
    self.out = x.view(x.shape[0], -1)
    return self.out

  def parameters(self):
    return []
alphabet = [ch for ch in "abcdefghijklmnopqrstuvwxyz"]
alphabet = ["."]+alphabet
vocab_size = len(alphabet)
n_emb = 10
n_hidden = 100
block_size = 3
g = torch.Generator().manual_seed(214748345)
C = torch.randn((vocab_size, n_emb), generator = g)
stoi = {s:i for i,s in enumerate(alphabet)}
itos = {i:s for i, s in enumerate(alphabet)}
# Load model directly from full .pkl file
state_dict = torch.load('scratch_model_weights.pt', map_location='cpu')

model = Sequential([
    Embedding(vocab_size, n_emb),
    Flatten(),
    Linear(n_emb*block_size, n_hidden),tanh(),
    Linear(n_hidden, n_hidden), tanh(),
    Linear(n_hidden, vocab_size)])  # same architecture

# Load weights back into the layers
for i, layer in enumerate(model.Layers):
    if hasattr(layer, 'weight'):
        layer.weight = state_dict[f'layer{i}_weight']
    if hasattr(layer, 'bias'):
        layer.bias = state_dict[f'layer{i}_bias']
def generate_next(context):
    out = []
    while True:
        with torch.no_grad():
            x = torch.tensor([context])
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
    return ''.join(itos[i] for i in out)
# Streamlit UI
st.title("🧠 AI Name Generator")
st.write("Enter the starting letters (minimum 4 characters):")

user_input = st.text_input("Prefix", max_chars=20)
CONTEXT_LENGTH = block_size
if user_input:
    user_input = user_input.lower()
    valid_input = [ch for ch in user_input if ch in stoi]

    if len(valid_input) < CONTEXT_LENGTH:
        st.warning(f"Please enter at least {CONTEXT_LENGTH} valid characters from the vocabulary.")
    else:
        context = [stoi[ch] for ch in valid_input[-CONTEXT_LENGTH:]]
        generated = generate_next(context)
        st.success(f"Generated name: `{user_input + generated}`")