# implementation file for tensor class
import numpy as np

class tensor:
    def __init__(self, data, _children=(), _op=''):
        if isinstance(data, tensor):
            data = data.data
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=float)             
        elif not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)
        else:
            data = data.astype(float, copy=False)
        self.data = data
        self._op = _op
        self.grad = np.zeros_like(self.data, dtype=float)
        self._prev = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f'tensor(data={self.data}, grad={self.grad} shape={self.data.shape})'  

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad[...] = 0.0

    @staticmethod
    def _coerce(x):                                         
        return x if isinstance(x, tensor) else tensor(x)

    @staticmethod
    def _unbroadcast(grad, shape):
        g = grad
        while g.ndim > len(shape):
            g = g.sum(axis=0)
        for i, (gd, sd) in enumerate(zip(g.shape, shape)):
            if sd == 1 and gd != 1:
                g = g.sum(axis=i, keepdims=True)
        return g

    def __getitem__(self, idx):
        out = tensor(self.data[idx], (self,), 'getitem')
        def _backward():
            g = np.zeros_like(self.data, dtype=float)
            np.add.at(g, np.index_exp[idx], out.grad)       
            self.grad += g
        out._backward = _backward
        return out

    def __add__(self, other):
        other = tensor._coerce(other)
        out = tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad  += tensor._unbroadcast(out.grad, self.shape)
            other.grad += tensor._unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out
    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-tensor._coerce(other))
    def __rsub__(self, other):
        return tensor._coerce(other) + (-self)              

    def __mul__(self, other):
        other = tensor._coerce(other)
        out = tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad  += tensor._unbroadcast(other.data * out.grad, self.shape)
            other.grad += tensor._unbroadcast(self.data  * out.grad, other.shape)
        out._backward = _backward
        return out
    __rmul__ = __mul__

    def __truediv__(self, other):
        return self * (tensor._coerce(other) ** -1)
    def __rtruediv__(self, other):
        return tensor._coerce(other) * (self ** -1)

    def __pow__(self, p):
        if isinstance(p, (int, float)):
            out = tensor(self.data ** p, (self,), '**')
            def _backward():
                self.grad += p * (self.data ** (p - 1.0)) * out.grad
            out._backward = _backward
            return out
        p = tensor._coerce(p)
        out = tensor(self.data ** p.data, (self, p), '**')
        def _backward():
            eps = 1e-12
            self.grad += tensor._unbroadcast(p.data * (self.data ** (p.data - 1.0)) * out.grad, self.shape)
            p.grad    += tensor._unbroadcast(out.data * np.log(np.maximum(self.data, eps)) * out.grad, p.shape)
        out._backward = _backward
        return out

    def tanh(self):
        y = np.tanh(self.data)
        out = tensor(y, (self,), 'tanh')                    
        def _backward():
            self.grad += (1.0 - y*y) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        y = np.exp(self.data)
        out = tensor(y, (self,), 'exp')                     
        def _backward():
            self.grad += y * out.grad
        out._backward = _backward
        return out

    def relu(self):
        y = np.where(self.data > 0, self.data, 0.0)
        out = tensor(y, (self,), 'relu')                    
        def _backward():
            self.grad += (self.data > 0).astype(float) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')
        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                axes = [axis] if isinstance(axis, int) else list(axis)
                for a in axes:
                    g = np.expand_dims(g, axis=a)
            self.grad += np.broadcast_to(g, self.shape)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            count = self.data.size
        else:
            reduced = self.data.sum(axis=axis, keepdims=keepdims)
            count = reduced.size
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    def reshape(self, *shape):
        out = tensor(self.data.reshape(*shape), (self,), 'reshape')
        def _backward():
            self.grad += out.grad.reshape(self.shape)
        out._backward = _backward
        return out

    @property
    def T(self):
        out = tensor(self.data.T, (self,), 'transpose')
        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = tensor._coerce(other)
        A, B = self.data, other.data

        # guard 2D shapes (clear errors early)                         
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(f"matmul supports 2D@2D, got {A.shape} @ {B.shape}")
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"inner dims mismatch: {A.shape} @ {B.shape}")

        Y = A @ B
        out = tensor(Y, (self, other), '@')
        def _backward():
            dY = out.grad
            self.grad  += dY @ B.T
            other.grad += A.T @ dY
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data, dtype=float)
        for v in reversed(topo):
            v._backward()
