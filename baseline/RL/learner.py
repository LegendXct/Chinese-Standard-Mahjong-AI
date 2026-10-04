from multiprocessing import Process, Event
import time
import numpy as np
import torch
from torch.nn import functional as F
import signal

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import CNNModel

def _npu_available():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, 'npu') and hasattr(torch.npu, 'is_available') and torch.npu.is_available()

class Learner(Process):
    
    def __init__(self, config, replay_buffer):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.stop_event = Event()

    def stop(self):
        self.stop_event.set()
    
    def run(self):
        # keep signal handling in main process only
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda signum, frame: self.stop_event.set())
        # create model pool
        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])
        try:
            # initialize model params
            requested_device = str(self.config.get('device', 'cpu')).strip().lower()
            if requested_device == 'auto':
                if _npu_available():
                    requested_device = 'npu'
                elif torch.cuda.is_available():
                    requested_device = 'cuda'
                else:
                    requested_device = 'cpu'
            if requested_device.startswith('npu') and not _npu_available():
                print('Warning: requested NPU device but NPU is unavailable, using CPU in learner.')
                requested_device = 'cpu'
            if requested_device.startswith('cuda') and not torch.cuda.is_available():
                print('Warning: requested CUDA device but CUDA is unavailable, using CPU in learner.')
                requested_device = 'cpu'
            device = torch.device(requested_device)
            model = CNNModel()
            
            # send to model pool
            model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
            model = model.to(device)
            
            # training
            optimizer = torch.optim.Adam(model.parameters(), lr = self.config['lr'])
            
            # wait for initial samples
            while self.replay_buffer.size() < self.config['min_sample'] and not self.stop_event.is_set():
                time.sleep(0.1)
            
            cur_time = time.time()
            iterations = 0
            while not self.stop_event.is_set():
                # sample batch
                batch = self.replay_buffer.sample(self.config['batch_size'])
                obs = torch.tensor(batch['state']['observation']).to(device)
                mask = torch.tensor(batch['state']['action_mask']).to(device)
                states = {
                    'observation': obs,
                    'action_mask': mask
                }
                actions = torch.tensor(batch['action']).unsqueeze(-1).to(device)
                advs = torch.tensor(batch['adv']).to(device)
                targets = torch.tensor(batch['target']).to(device)
                
                print('Iteration %d, replay buffer in %d out %d' % (iterations, self.replay_buffer.stats['sample_in'], self.replay_buffer.stats['sample_out']))
                
                # calculate PPO loss
                model.train(True) # Batch Norm training mode
                old_logits, _ = model(states)
                old_probs = F.softmax(old_logits, dim = 1).gather(1, actions)
                old_log_probs = torch.log(old_probs + 1e-8).detach()
                for _ in range(self.config['epochs']):
                    logits, values = model(states)
                    action_dist = torch.distributions.Categorical(logits = logits)
                    probs = F.softmax(logits, dim = 1).gather(1, actions)
                    log_probs = torch.log(probs + 1e-8)
                    ratio = torch.exp(log_probs - old_log_probs)
                    surr1 = ratio * advs
                    surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs
                    policy_loss = -torch.mean(torch.min(surr1, surr2))
                    value_loss = torch.mean(F.mse_loss(values.squeeze(-1), targets))
                    entropy_loss = -torch.mean(action_dist.entropy())
                    loss = policy_loss + self.config['value_coeff'] * value_loss + self.config['entropy_coeff'] * entropy_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # push new model (without moving the entire module between CPU/GPU)
                model_pool.push({k: v.detach().cpu() for k, v in model.state_dict().items()})
                
                # save checkpoints
                t = time.time()
                if t - cur_time > self.config['ckpt_save_interval']:
                    path = self.config['ckpt_save_path'] + 'model_%d.pt' % iterations
                    torch.save(model.state_dict(), path)
                    cur_time = t
                iterations += 1
        finally:
            model_pool.close()
