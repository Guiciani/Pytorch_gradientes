# Quando treinamos as redes neurais, o algoritmos mais utilizado é o [Back
# Propagation], nesse modelo os parametros (pesos do modelo) são ajustados
# de acordo com o gradiente do loss function.

# A mecanica do Pytorch utiliza o torch.autograd. Esta, suporta os ajustes 
# automaticamente do gradiente, para qualquer grafo computacional.



import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print('Gradient function for z =', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)

# Para otimizar os pesos dos parametros na rede neural, precisamos computar as
# derivadas da nossa função loss respeirando os parametros com o loss.backward()

loss.backward()
print(w.grad)
print(b.grad)

# Desabilitando o rastreamento do gradiente

# Por padrão todos os tensores tem requires_grad=True para rastreamento de onde
# ele partiu/esteve. Em alguns casos, para seguir apenas em frente, é possivel
# desabilitar o rastreamento do gradiente.

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# Também pode-se utilizar o detach() para desabilitar

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

# Alguns objetivos para desabilitar o rastreamento do gradiente

# Marcar alguns parametros na rede neural como Frozen Parameters.
# Aumentar a velocidade computacional quando necessario apenas seguir em frente

# Tensor Gradients e Produto Jacobiano
# 
# Em muitos casos temos uma loss function escalar e precisamos computar o 
# gradiente respeitando alguns parametros. Entretanto,existem alguns casos
# onde o output da função é um tensor arbitrario. Para estes, o Pytorch
# utiliza o Jacobian Product.

# Ao inves de computar a matriz Jacobiana em si, o Pytorch permite
# computar o produto jacobiano V**T

inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)