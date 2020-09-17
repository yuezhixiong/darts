import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from min_norm_solvers import MinNormSolver, gradient_normalizers
import torch.nn.functional as F

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    # loss on train data
    loss = self.model._loss(input, target)
    # w
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    # w_grad + weight_decay * w
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    # eta: learning rate
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled, C):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer, C)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def param_number(self, unrolled_model, C):
    def compute_u(C, is_reduction):
      a = np.array([0, 0, 0, 0, 2*(C**2+9*C), 2*(C**2+25*C), C**2+9*C, C**2+25*C]).reshape(8, 1)
      u = torch.from_numpy(np.repeat(a, 14, axis=1))
      if is_reduction:
        u[3, :] = u[3, :] + torch.Tensor([C**2, C**2, C**2, C**2, 0, C**2, C**2, 0, 0, C**2, C**2, 0, 0, 0])
      return u
    loss = 0
    # u = torch.from_numpy(np.array([0, 0, 0, 0, 2*(C**2+9*C), 2*(C**2+25*C), C**2+9*C, C**2+25*C]))
    C_list = [C, C, 2*C, 2*C, 2*C, 4*C, 4*C, 4*C]
    for i in range(unrolled_model._layers):
      if unrolled_model.cells[i].reduction:
        alpha = F.softmax(unrolled_model.arch_parameters()[1], dim=-1)
        u = compute_u(C_list[i], is_reduction=True)
      else:
        alpha = F.softmax(unrolled_model.arch_parameters()[0], dim=-1)
        u = compute_u(C_list[i], is_reduction=False)
      loss += (2 * torch.mm(alpha.t(), u).sum(dim=1) / torch.from_numpy(np.repeat(range(2, 6), [2, 3, 4, 5]))).sum()
    return loss

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, C):
    grads = {}
    self.optimizer.zero_grad()
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()

    grads['darts'] = []
    for param in unrolled_model.arch_parameters():
      if param.grad is not None:
          grads['darts'].append(Variable(param.grad.data.clone(), requires_grad=False))

    # ---- param loss ----
    self.optimizer.zero_grad()
    param_loss = self.param_number(unrolled_model, C)
    param_loss.backward()
    grads['param'] = []
    for param in unrolled_model.arch_parameters():
      if param.grad is not None:
          grads['param'].append(Variable(param.grad.data.clone(), requires_grad=False))
    # dalpha_param = [v.grad for v in unrolled_model.arch_parameters()]
    # ---- param loss ----

    # ---- MGDA -----
    sol, _ = MinNormSolver.find_min_norm_element([grads[t] for t in grads])
    loss = sol[0] * unrolled_loss + sol[1] * param_loss
    self.optimizer.zero_grad()
    loss.backward()
    # ---- MGDA -----

    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    # a new model with the same alpha
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

