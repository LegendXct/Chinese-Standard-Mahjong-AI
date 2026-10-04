from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner
import signal
import torch
import os
import time

def _npu_available():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, 'npu') and hasattr(torch.npu, 'is_available') and torch.npu.is_available()

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = {
        'replay_buffer_size': 50000,
        'replay_buffer_episode': 400,
        'model_pool_size': 20,
        'model_pool_name': 'model-pool',
        'num_actors': 24,
        'episodes_per_actor': 1000,
        'gamma': 0.98,
        'lambda': 0.95,
        'min_sample': 200,
        'batch_size': 256,
        'epochs': 5,
        'clip': 0.2,
        'lr': 1e-4,
        'value_coeff': 1,
        'entropy_coeff': 0.01,
        'device': 'npu',
        'ckpt_save_interval': 300,
        'ckpt_save_path': os.path.join(base_dir, 'checkpoint') + os.sep
    }

    requested_device = str(config.get('device', 'cpu')).strip().lower()
    if requested_device == 'auto':
        if _npu_available():
            config['device'] = 'npu'
        elif torch.cuda.is_available():
            config['device'] = 'cuda'
        else:
            config['device'] = 'cpu'
    elif requested_device.startswith('npu') and not _npu_available():
        print('Warning: NPU is not available, fallback to CPU.')
        config['device'] = 'cpu'
    elif requested_device.startswith('cuda') and not torch.cuda.is_available():
        print('Warning: CUDA is not available in current PyTorch build, fallback to CPU.')
        config['device'] = 'cpu'
    
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actors.append(actor)
    learner = Learner(config, replay_buffer)

    stop_requested = {'value': False}

    def _request_stop(signum, frame):
        if not stop_requested['value']:
            print('Received signal %d, shutting down...' % signum)
        stop_requested['value'] = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    def _wait_processes(processes, timeout):
        alive = [p for p in processes if p.is_alive()]
        deadline = time.time() + timeout
        while alive and time.time() < deadline:
            for p in alive[:]:
                p.join(timeout = 0.1)
                if not p.is_alive():
                    alive.remove(p)
        return alive

    started_actors = []
    learner_started = False
    try:
        for actor in actors:
            actor.start()
            started_actors.append(actor)
        learner.start()
        learner_started = True

        while True:
            alive_actors = [a for a in started_actors if a.is_alive()]
            if not alive_actors:
                break
            for actor in alive_actors:
                actor.join(timeout = 0.2)
            if stop_requested['value']:
                raise KeyboardInterrupt

        learner.stop()
        while learner.is_alive():
            learner.join(timeout = 1)
            if stop_requested['value']:
                break
    except KeyboardInterrupt:
        print('KeyboardInterrupt, force stopping workers...')
    finally:
        for actor in started_actors:
            actor.stop()

        alive_actors = _wait_processes(started_actors, timeout = 5)
        for actor in alive_actors:
            actor.terminate()
        alive_actors = _wait_processes(alive_actors, timeout = 5)
        for actor in alive_actors:
            print('%s did not exit after terminate(), sending kill()' % actor.name)
            actor.kill()
        _wait_processes(alive_actors, timeout = 3)

        for actor in started_actors:
            try:
                actor.close()
            except Exception:
                pass

        if learner_started:
            if learner.is_alive():
                learner.stop()
            alive_learner = _wait_processes([learner], timeout = 30)
            if alive_learner:
                learner.terminate()
                alive_learner = _wait_processes([learner], timeout = 5)
            if alive_learner:
                print('Learner did not exit after terminate(), sending kill()')
                learner.kill()
                _wait_processes([learner], timeout = 3)
            try:
                learner.close()
            except Exception:
                pass

        replay_buffer.close()
