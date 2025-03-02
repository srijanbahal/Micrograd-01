import math

# Defining The major Value Object
class Value:
  def __init__(self, data, _children=(), _op='', label=''):
    # Defining All the attributes for the Value Objects
    self.data = data
    self.prev = set(_children)
    self._op = _op # Stores the Opearand
    self._backward = lambda: None # Backward function initializing with None 
    self.label = label
    self.grad = 0.0 # Gradient For the gradient Calcultion (we'll see its use Below)

  def __repr__(self):
    return f"Value(data={self.data})"

  def __add__(self, other):
    if not isinstance(self, Value):
        self = Value(self)  # Convert self to Value if it's an int/float
    if not isinstance(other, Value):
        other = Value(other)  # Convert other to Value if it's an int/float
    # this will check if the other is a instace of Value, and if not it will wrap the other in Value instance
    out = Value(self.data + other.data, (self, other), '+')
    def _backward(): # this function calculate the gradient of the output wrt to this add operation basically just getting the derivative
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out
  def __radd__(self, other):  # This is the new method
    return self + other  # Simply call __add__ to reuse the logic

  def __rmul__(self, other): # other * self => this will just swap self with tother
      return self * other


  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other) # this just converts the other to value object for Multiplication
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out


  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
      self.grad += other * (self.data**(other-1)) * out.grad
    out._backward = _backward

    return out

  def __truediv__(self, other): # self / other
    return self * other**-1

  def __neg__(self): # -self
    return self * -1

  def __sub__(self, other): # self - other
    return self + (-other)

  # Defining Exp Funciton
  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')

    def _backward():
      self.grad += out.grad * out.data
    out._backward = _backward
    return out

  def __neg__(self): ## Returns the Negation of self
    return self * -1

  def __sub__(self, other): # returns the Subtracttion using the addtion
    return self + (-other)


  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out

  def __rtruediv__(self, other): # other / self
        return other * self**-1
  def backward(self):
    ## using Topological Sort
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v.prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed (topo):
      node._backward()

a = Value(2.0, label = 'a')
b = Value(-3.0, label = 'b')
c = Value(10.0, label = 'c')
e = a * b ; e.label = 'e'
d = e + c ; d.label = 'd'
f = Value(-2.0, label = 'f')
L = f*d ; L.label = 'L'
L
d.prev