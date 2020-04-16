"""
PyTorch model for the uncertainty estimation.

Copyright 2019
Vincent Fortuin
Microsoft Research Cambridge
"""

from collections import OrderedDict
import numpy as np
from core import *
from torch_backend import *
from model_helpers import *


    
class TorchUncertaintyPair(nn.Module):
    def __init__(self, uncertainty_threshold=1e-2, output_size=10, output_weight=1.,
                 init_scaling=1., gp_weight=1.):
        super().__init__()
        self.learner = Network(net(net_type="large", output_size=output_size, weight=output_weight))
        self.prior = Network(net(net_type="basic", output_size=output_size, weight=output_weight))
        scale_prior(self, init_scaling)
        for param in self.prior.parameters():
            param.requires_grad = False
        self.uncertainty_threshold = uncertainty_threshold
        self.gp_weight = gp_weight
            
    def forward(self, inputs):
        prior_output = self.prior(inputs)['out']
        learner_output = self.learner(inputs)['out']
        diff = (prior_output - learner_output)
        squared_diff = diff * diff
        msd = torch.mean(squared_diff, dim=-1)
        correct = msd < self.uncertainty_threshold
        loss = msd
        if self.gp_weight > 0.:
            inputs_var = torch.autograd.Variable(inputs['input'], requires_grad=True)
            outputs_var = torch.mean(self.learner({"input": inputs_var})['out'])
            grads = torch.autograd.grad(outputs_var, inputs_var, create_graph=True,
                                        retain_graph=True, only_inputs=True)[0]
            grads_norm = grads.norm()
            loss = loss + self.gp_weight * grads_norm
        return {'uncertainties': msd, 'loss': loss, 'correct': correct}
    
    
class TorchUncertaintyEnsemble(nn.Module):
    def __init__(self, ensemble_size=2, uncertainty_threshold=1e-2, output_size=10,
                 output_weight=1., init_scaling=1., gp_weight=0., beta=1.):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.beta = beta
        self.uncertainty_threshold = uncertainty_threshold
        self.ensemble = []
        for i in range(ensemble_size):
            setattr(self, f"pair_{i}", TorchUncertaintyPair(uncertainty_threshold=uncertainty_threshold,
                                                output_size=output_size,
                                                output_weight=output_weight,
                                                init_scaling=init_scaling,
                                                gp_weight=gp_weight))
            self.ensemble.append(getattr(self, f"pair_{i}"))
        
    def forward(self, inputs):
        outputs = [pair(inputs) for pair in self.ensemble]
        losses_stacked = torch.stack([out['loss'] for out in outputs])
        losses_mean = torch.sum(losses_stacked, axis=0)
        uncertainties_stacked = torch.stack([out['uncertainties'] for out in outputs])
        if self.ensemble_size >= 2:
            uncertainties_combined = torch.mean(uncertainties_stacked, axis=0) + self.beta * torch.std(uncertainties_stacked, axis=0)
        else:
            uncertainties_combined = torch.mean(uncertainties_stacked, axis=0)
        correct = uncertainties_combined < self.uncertainty_threshold
        return {'uncertainties': uncertainties_combined, 'loss': losses_mean, 'correct': correct}
    
    
class TorchDeepEnsemble(nn.Module):
    def __init__(self, ensemble_size=2, uncertainty_threshold=1e-2, output_size=10, output_weight=1., adv_eps=0.):
        super().__init__()
        assert ensemble_size >=1, "The ensemble needs at least one member."
        self.ensemble_size = ensemble_size
        self.uncertainty_threshold = uncertainty_threshold
        self.adv_eps = adv_eps
        self.ensemble = []
        for i in range(ensemble_size):
            setattr(self, f"learner_{i}", Network(net(net_type="basic",
                                    output_size=output_size, weight=output_weight)))
            self.ensemble.append(getattr(self, f"learner_{i}"))
        
    def forward(self, inputs):
        outputs = [learner(inputs) for learner in self.ensemble]
        logits_stacked = torch.stack([out['out'] for out in outputs])
        labels_stacked = torch.stack([out['target'] for out in outputs])
        softmax = torch.nn.Softmax(dim=-1)
        log_softmax = torch.nn.LogSoftmax(dim=-1)
        softmax_stacked = softmax(logits_stacked)
        log_softmax_stacked = log_softmax(logits_stacked)
        log_softmax_reshaped = log_softmax_stacked.reshape([-1, log_softmax_stacked.shape[-1]])
        labels_reshaped = labels_stacked.reshape([-1])
        cross_entropy_loss = torch.nn.NLLLoss(reduction='none')
        loss = cross_entropy_loss(log_softmax_reshaped, labels_reshaped)
        softmax_means = torch.mean(softmax_stacked, dim=0)
        softmax_max = torch.max(softmax_means, dim=-1)
        predictions = softmax_max.indices
        correct = (predictions == labels_stacked[0])
        uncertainties = 1. - softmax_max.values
        if self.adv_eps > 0.:
            loss_fun = torch.nn.CrossEntropyLoss()
            inputs_var = torch.autograd.Variable(inputs['input'], requires_grad=True)
            outputs_vars = [learner({"input": inputs_var})['out'] for learner in self.ensemble]
            losses = [loss_fun(out_var, inputs['target']) for out_var in outputs_vars]
            grads = [torch.autograd.grad(loss, inputs_var, only_inputs=True)[0] for loss in losses]
            adv_samples = [inputs['input'] + self.adv_eps * torch.sign(grad) for grad in grads]
            adv_outputs = [learner({"input": adv_sample})['out'] for learner, adv_sample in zip(self.ensemble, adv_samples)]
            loss_fun2 = torch.nn.CrossEntropyLoss(reduction='none')
            adv_losses = [loss_fun2(adv_out, inputs['target']) for adv_out in adv_outputs]
            adv_loss = torch.stack(adv_losses).reshape([-1])
            loss = loss + adv_loss
        output = {'uncertainties': uncertainties, 'loss': loss, 'correct': correct}
        return output
    

class DropoutModel(nn.Module):
    def __init__(self, output_size, weight_regularizer, dropout_regularizer, bootstrap_size=1):
        super().__init__()
        self.bootstrap_size = bootstrap_size
        self.output_size = output_size
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        self.prep = nn.Sequential(nn.Sequential(OrderedDict(conv_bn(3, 64))))
        self.scd_prep = SpatialConcreteDropout(weight_regularizer=self.weight_regularizer,dropout_regularizer=self.dropout_regularizer)
        self.layer1 = nn.Sequential(nn.Sequential(OrderedDict(conv_bn(64,128))), nn.MaxPool2d(2))
        self.scd_layer1 = SpatialConcreteDropout(weight_regularizer=self.weight_regularizer,dropout_regularizer=self.dropout_regularizer)
        self.residual1 = nn.Sequential(OrderedDict(conv_bn(128, 128)))
        self.scd_residual1 = SpatialConcreteDropout(weight_regularizer=self.weight_regularizer,dropout_regularizer=self.dropout_regularizer)
        self.layer2 = nn.Sequential(nn.Sequential(OrderedDict(conv_bn(128, 256))), nn.MaxPool2d(2))
        self.scd_layer2 = SpatialConcreteDropout(weight_regularizer=self.weight_regularizer,dropout_regularizer=self.dropout_regularizer)
        self.layer3 = nn.Sequential(nn.Sequential(OrderedDict(conv_bn(256, 512))), nn.MaxPool2d(2))
        self.scd_layer3 = SpatialConcreteDropout(weight_regularizer=self.weight_regularizer,dropout_regularizer=self.dropout_regularizer)
        self.residual3 = nn.Sequential(OrderedDict(conv_bn(512, 512)))
        self.scd_residual3 = SpatialConcreteDropout(weight_regularizer=self.weight_regularizer,dropout_regularizer=self.dropout_regularizer)
        self.linear = nn.Sequential(nn.MaxPool2d(4), Flatten(), nn.Linear(512, output_size))
        
    def forward(self, inputs):
        x = inputs['input']
        outputs = []
        regularizers = []
        
        for i in range(self.bootstrap_size):
            regularization = torch.empty(6, device=x.device, dtype=x.dtype)

            out, regularization[0] = self.scd_prep(x, self.prep)
            out, regularization[1] = self.scd_layer1(out, self.layer1)
            out_temp, regularization[2] = self.scd_residual1(out, self.residual1)
            out = out + out_temp
            out, regularization[3] = self.scd_layer2(out, self.layer2)
            out, regularization[4] = self.scd_layer3(out, self.layer3)
            out_temp, regularization[5] = self.scd_residual3(out, self.residual3)
            out = out + out_temp
            out = self.linear(out)
            
            outputs.append(out)
            regularizers.append(regularization)
            
        logits_stacked = torch.stack([out for out in outputs])
        regs_stacked = torch.stack([reg for reg in regularizers])
        labels_stacked = torch.stack([inputs['target'] for out in outputs])
        softmax = torch.nn.Softmax(dim=-1)
        log_softmax = torch.nn.LogSoftmax(dim=-1)
        softmax_stacked = softmax(logits_stacked)
        log_softmax_stacked = log_softmax(logits_stacked)
        log_softmax_reshaped = log_softmax_stacked.reshape([-1, log_softmax_stacked.shape[-1]])
        labels_reshaped = labels_stacked.reshape([-1])
        cross_entropy_loss = torch.nn.NLLLoss(reduction='none')
        supervised_loss = torch.sum(cross_entropy_loss(log_softmax_reshaped, labels_reshaped))
        regularizer_loss = torch.sum(regs_stacked)
        loss = ((supervised_loss + regularizer_loss) / labels_reshaped.shape[0]).reshape([-1])
        softmax_means = torch.mean(softmax_stacked, dim=0)
        softmax_max = torch.max(softmax_means, dim=-1)
        predictions = softmax_max.indices
        correct = (predictions == labels_stacked[0])
        uncertainties = 1. - softmax_max.values
                
        output = {'uncertainties': uncertainties, 'loss': loss, 'correct': correct}
        return output