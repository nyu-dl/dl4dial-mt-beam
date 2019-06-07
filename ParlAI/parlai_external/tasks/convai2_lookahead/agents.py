import torch
from parlai.tasks.convai2.agents import SelfOriginalTeacher
from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.core.worlds import create_task

class LookaheadselfTeacher(SelfOriginalTeacher):
    def __init__(self, opt, shared=None):
        if 'stream' in opt.get('datatype', ''):
            raise RuntimeError('Cannot do convai2:lookaheadself with stream datatype.')
        super().__init__(opt, shared)

    def get(self, episode_idx, entry_idx=0):
        ex = super().get(episode_idx, entry_idx)
        if not ex['episode_done']:
            next_ex = super().get(episode_idx, entry_idx + 1)
            if 'text' in next_ex:
                ex['next_text'] = next_ex['text']
            if 'labels' in next_ex:
                ex['next_labels'] = next_ex['labels']
            if 'eval_labels' in next_ex:
                ex['next_eval_labels'] = next_ex['eval_labels'] 
        return ex
