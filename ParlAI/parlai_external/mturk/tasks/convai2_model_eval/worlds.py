# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.agents import create_agent_from_shared
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
import parlai.mturk.core.mturk_utils as mutils
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.mturk.core.worlds import MTurkOnboardWorld

from joblib import Parallel, delayed
import numpy as np
import os
import pickle
import random
import time


# INSTRUCTIONS
ONBOARD_MSG = '\nWelcome! Below is your persona \
        (you can find it on the left side of the chat)\n \
        When you are ready to start your conversation, \
        click the "I am ready, continue" button below\n'
START_MSG = '\nSuccessfully matched. \
        Now let\'s get to know each other through the chat! \n\
        You need to finish at least <b>{} chat turns</b>, \
        after which you can click the "Done" button to end the chat. \n \
        <b>You can track your character description on the left.</b> \n\
        <span style="color:blue"><b>Please try to speak to the other person \
        as if you are the character assigned.</b></span> \n \
        <span style="color:blue"><b>Do not trivially copy \
        the character descriptions into the message.</b></span>'
CHAT_NOT_DONE_MSG = 'Sorry, we need at least <b>{} more turn(s)</b> to finish. \
       Please send a new message:'
TIMEOUT_MSG = '<b> The other person has timed out. \
        Please click the "Done with this HIT" button below to finish this HIT.\
        </b>'
EXCEED_MIN_TURNS_MSG = '\n {} chat turns finished! \n \
        You can click the "Done" button to end the chat if it\'s your turn \
        or keep chatting.'
UNEXPECTED_DISCONNECTION_MSG = 'The other worker unexpectedly diconnected. \n \
        Please click <span style="color:blue"><b>Done with this HIT</b>\
        </span> button below to finish this HIT.'
CHAT_ENDED_MSG = 'One of you ended the chat. Thanks for your time! \n\
        Please click <span style="color:blue"><b>Done with this HIT</b>\
        </span> button below to finish this HIT.'
WAITING_MSG = 'Please wait while we match you with another worker...'
NAN_MSG = 'The score you entered must be in [1, 2, 3, 4, 5]. Please \
        try again:'
TOO_SHORT_MSG = 'Your message is too short, please make it more than \
        <b><span style="color:red">{} words</span></b>.'
TOO_LONG_MSG = 'Your message is too long, please make it less than \
        <b><span style="color:red">{} words</span></b>.'

# CHOOSING A TOPIC
PICK_TOPIC_MSG = 'To start, please select a topic on the left, then click the \
    \'Pick Topic\' button.'
AFTER_PICK_TOPIC_MSG = 'Thank you for selecting a topic! Now, begin the \
    conversation with your partner about the topic.'
PLEASE_WAIT_MSG = 'Your partner will choose a discussion topic. Click the \
    button below when you are ready to continue.'

# EVALUATION
OTHER_AGENT_FINISHED_MSG = '<b><span style="color:red">This chat is \
    done!</span></b> Please click \
    <span style="color:blue"><b>Done with this HIT</b></span> button below \
    to finish this HIT.'
# Engagingness
ENGAGINGNESS_MSGS = [
    'How much did you enjoy talking to this user?',
    ('How likely would you be to continue talking to this user?')
]
ENGAGINGNESS_CHOICES = ['not at all', 'a little', 'somewhat', 'a lot']

# Fluency
FLUENCY_MSGS = [
    'Did you find the other user easy to understand?',
    'Please select the sentences that you found confusing:'
]
FLUENCY_CHOICES = [
    'I didn\'t understand anything',
    'I understood some of what they said',
    'I mostly understood them',
    'I understood everything'
]

# Consistency
CONSISTENCY_MSGS = [
    'How often did the other user contradict themselves?',
    ('Please select the sentences for which the other user contradicted ' +
     'themselves:')
]
CONSISTENCY_CHOICES = ['not at all', 'a little', 'a lot']

# Persona
PERSONA_MSG = (
    'Which prompt (character) do you think the other user was ' +
    'given for this conversation?  \n 1.<br> {} <br> 2.<br> {}'
)
PERSONA_CHOICES = ['1', '2']


class PersonasGenerator(object):
    def __init__(self, opt):
        self.text_file = self._path(opt)
        self.personas = self.extract_personas()

    def _path(self, opt):
        # Build the data if it doesn't exist.
        persona = opt['persona_type']
        datatype = opt['persona_datatype'].split(':')[0]
        dt = datatype + '_' + persona
        if datatype == 'test':
            return os.path.join(opt['parlai_home'],
                                'parlai_internal/projects/convai2/test_set',
                                dt + '_original_no_cands.txt')
        return os.path.join(opt['datapath'], 'ConvAI2', dt + '_original_no_cands.txt')

    def extract_personas(self):
        personas = []
        with open(self.text_file, 'r') as f:
            lines = f.readlines()

        new_persona = []
        for line in lines:
            if 'persona: ' in line:
                new_persona.append(line.split('persona: ')[1].replace('\n', ''))
            else:
                if new_persona:
                    personas.append(new_persona)
                    new_persona = []

        return personas

    def get_persona(self):
        return random.choice(self.personas)


class PersonaAssignWorld(MTurkOnboardWorld):
    """A world that assigns a persona to an agent."""
    def __init__(self, opt, mturk_agent):
        self.max_persona_time = opt['max_persona_time']
        self.human_eval = opt['human_eval']
        super().__init__(opt, mturk_agent)

    def parley(self):
        personas = self.mturk_agent.personas_generator.get_persona()
        self.mturk_agent.personas = personas
        if not self.human_eval:
            # get model personas
            model_personas = self.mturk_agent.personas_generator.get_persona()
            while model_personas == personas:
                model_personas = \
                    self.mturk_agent.personas_generator.get_persona()
            self.mturk_agent.model_personas = model_personas

        persona_text = ''
        for persona in personas:
            persona_text += '<b><span style="color:blue">' \
                            '{}\n</span></b>'.format(persona.strip())

        self.mturk_agent.observe({
            'id': 'SYSTEM',
            'show_persona': True,
            'text': ONBOARD_MSG + '<br>' + persona_text + '<br>'})

        act = self.mturk_agent.act(timeout=self.max_persona_time)
        timed_out = self.check_timeout(act)
        if timed_out:
            self.episodeDone = True
            return

    def check_timeout(self, act):
        if 'text' in act:
            if ((act['text'] == '[TIMEOUT]') or (act['text'] == '[RETURNED]') or
                    (act['text'] == '[DISCONNECT]')):
                return True
        return False


class ConvAI2Eval(MultiAgentDialogWorld):
    def __init__(self, opt, agents=None, shared=None,
                 range_turn=[3, 5], max_turn=5,
                 max_resp_time=120,
                 model_agent_opt=None,
                 world_tag='',
                 agent_timeout_shutdown=120):

        # TURN CONTROL
        self.turn_idx = 0
        self.range_turn = range_turn
        self.max_turn = max_turn
        self.n_turn = np.random.randint(
            self.range_turn[0],
            self.range_turn[1]
        ) + 1
        self.chat_done = False
        self.other_first = random.choice([True, False])
        self.short_eval = opt.get('short_eval')

        # DATA
        self.start_time = time.time()
        self.dialog = []
        self.mtbeam_log = []
        self.dialog_list = []
        self.engagingness_scores = []
        self.fluency_scores = []
        self.consistency_scores = []
        self.persona_scores = []
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.world_tag = world_tag

        super().__init__(opt, agents, shared)

        # MODEL AGENT SET UP
        if model_agent_opt is not None:
            self.model_agent = create_agent_from_shared(model_agent_opt)
        else:
            # case where we test against a human
            self.model_agent = None

        # TIMEOUT PROTOCOLS
        self.max_resp_time = max_resp_time  # in secs
        self.agent_timeout_shutdown = agent_timeout_shutdown

        # PERSONAS
        self.bot_seen_persona = False
        self.personas = [ag.personas for ag in self.agents]
        if self.model_agent is not None:
            self.eval_agent = self.agents[0]
            self.model_personas = self.agents[0].model_personas
            self.model_persona_text = '\n'.join([
                'your persona: ' + pers for pers in self.model_personas
            ])
        else:
            self.model_personas = None
            for idx in range(len(self.agents)):
                if self.agents[idx].id == 'PERSON_1':
                    self.eval_agent = self.agents[idx]
                    self.other_agent = self.agents[idx - 1]
                    break

    def get_control_msg(self):
        return {'id': 'SYSTEM', 'episode_done': False}

    def get_human_agent_act(self, agent):
        act = agent.act(timeout=self.max_resp_time)
        while self.is_msg_tooshortlong(act, agent):
            act = agent.act(timeout=self.max_resp_time)
        return act

    def format_model_reply(self, text):
        new_text = text.lower()
        switch_list = [
            (' .', '.'),
            (' ,', ','),
            (' ?', '?'),
            (' !', '!'),
            (" '", "'"),
        ]
        for tup in switch_list:
            new_text = new_text.replace(tup[0], tup[1])

        return new_text

    def format_personachat_text(self, text):
        new_text = text.lower()

        switch_list = [("we're", "were"), ("let's", "lets"), ("it's", "its"),
                       ("who's", "whos"), ("you're", "youre"),
                       ("you've", "youve"), ("he'd", "hed"), ("he'll", "hell")]
        for tup in switch_list:
            new_text = new_text.replace(tup[0], tup[1])

        return new_text

    def get_bot_observation(self):
        # TODO: clear bots queue each time so that it observes itself properly
        pass

    def parley(self):
        self.turn_idx += 1
        print(self.world_tag + ' is at turn {}...'.format(self.turn_idx))

        """If at first turn, we need to give each agent their persona"""
        if self.turn_idx == 1:
            for idx, agent in enumerate(self.agents):
                persona_text = ''
                for s in self.personas[idx]:
                    persona_text += '<b><span style="color:blue">' \
                                    '{}\n</span></b>'.format(s.strip())
                control_msg = self.get_control_msg()
                control_msg['persona_text'] = persona_text
                control_msg['text'] = self.get_instruction(
                    tag='start', agent_id=agent.id
                )
                # TODO: check that get instruction actually exists?
                agent.observe(validate(control_msg))
                if idx == 0:
                    time.sleep(3)

        """If we get to the min turns, inform turker that they can end if they
        want.
        """
        if self.turn_idx == self.n_turn + 1:
            for idx, agent in enumerate(self.agents):
                control_msg = self.get_control_msg()
                control_msg['text'] = self.get_instruction(
                    idx,
                    tag='exceed_min_turns'
                )
                control_msg['exceed_min_turns'] = True
                agent.observe(validate(control_msg))

        """Otherwise, we proceed accordingly."""
        # Other agent first
        if self.other_first and self.turn_idx == 1:
            if self.model_agent is not None:
                # Model must observe its persona
                persona_act = {'text': '\n'.join([self.model_persona_text,
                                                  '__SILENCE__']),
                               'episode_done': False}
                self.model_agent.observe(persona_act)
                self.bot_seen_persona = True
                model_act = self.model_agent.act()
                model_act['text'] = self.format_model_reply(model_act['text'])
                model_act['id'] = 'PERSON_2'
                self.dialog.append((1, model_act.get('text')))
                self.mtbeam_log.append(model_act.get('mtbeam_log'))
                self.eval_agent.observe(model_act)
            else:
                act = self.get_human_agent_act(self.other_agent)
                timeout = self.check_timeout(act)
                if timeout:
                    # eval agent early disconnect
                    control_msg = self.get_control_msg()
                    control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
                    self.eval_agent.observe(validate(control_msg))
                    return
                else:
                    self.dialog.append((1, act.get('text')))
                    self.eval_agent.observe(act)

        # Eval agent turn
        act = self.get_human_agent_act(self.eval_agent)
        timeout = self.check_timeout(act)
        if timeout:
            if self.model_agent is None:
                control_msg = self.get_control_msg()
                control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
                self.other_agent.observe(validate(control_msg))
            return

        if act['episode_done']:
            if self.turn_idx >= self.n_turn:
                if not self.other_first:
                    self.dialog_list = [
                        '\n'.join(
                            [self.dialog[i][1], self.dialog[i + 1][1]]
                        ) for i in range(0, len(self.dialog), 2)
                    ]
                else:
                    self.dialog_list = [' \n' + self.dialog[0][1]] + [
                        '\n'.join(
                            [self.dialog[i][1], self.dialog[i + 1][1]]
                        ) for i in range(1, len(self.dialog) - 1, 2)
                    ]
                self.parallel_eval_mode()

                self.chat_done = True
                for ag in self.agents:
                    control_msg = self.get_control_msg()
                    control_msg['text'] = CHAT_ENDED_MSG
                    ag.observe(validate(control_msg))
            return

        self.dialog.append((0, act['text']))

        # Lowercase and format text to match personachat data
        act['text'] = self.format_personachat_text(act['text'])

        if not self.bot_seen_persona and self.model_agent is not None:
            # Add persona for model to observe
            act['text'] = '\n'.join([self.model_persona_text, act['text']])
            self.bot_seen_persona = True
        if self.model_agent is not None:
            self.model_agent.observe(act)
        else:
            self.other_agent.observe(act)

        # Model_agent turn
        if not self.other_first or self.turn_idx < self.n_turn:
            if self.model_agent is not None:
                act = self.model_agent.act()
                act['text'] = self.format_model_reply(act['text'])
                act['id'] = 'PERSON_2'
                # NOTE: your model may or may not need to observe itself here
                # If it does, call model_observes_itself or some other specialized
                # function
            else:
                act = self.get_human_agent_act(self.other_agent)
                timeout = self.check_timeout(act)
                if timeout:
                    # eval agent early disconnect
                    control_msg = self.get_control_msg()
                    control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
                    self.eval_agent.observe(validate(control_msg))
                    return

            self.dialog.append((1, act.get('text')))
            self.mtbeam_log.append(act.get('mtbeam_log'))
            self.eval_agent.observe(act)

    def evaluate_engagingness(self):
        control_msg = self.get_control_msg()
        msg_rng = 1 if self.short_eval else len(ENGAGINGNESS_MSGS)
        for i in range(msg_rng):
            control_msg['text'] = ENGAGINGNESS_MSGS[i]
            control_msg['button_choices'] = '</ROUND>'.join(ENGAGINGNESS_CHOICES)
            self.eval_agent.observe(validate(control_msg))
            act = self.eval_agent.act(timeout=self.max_resp_time)
            timeout = self.check_timeout(act)
            if timeout:
                return False
            act_choice = ENGAGINGNESS_CHOICES.index(act.get('text'))
            self.engagingness_scores.append(act_choice)
        return True

    def evaluate_fluency(self):
        control_msg = self.get_control_msg()
        control_msg['text'] = FLUENCY_MSGS[0]
        control_msg['button_choices'] = '</ROUND>'.join(FLUENCY_CHOICES)
        self.eval_agent.observe(validate(control_msg))
        act = self.eval_agent.act(timeout=self.max_resp_time)
        timeout = self.check_timeout(act)
        if timeout:
            return False
        act_choice = FLUENCY_CHOICES.index(act.get('text'))
        self.fluency_scores.append(act_choice)
        if act_choice != 3:
            control_msg = self.get_control_msg()
            control_msg['text'] = FLUENCY_MSGS[1]
            control_msg['good_rounds'] = True
            control_msg['rounds'] = '</ROUND>'.join(self.dialog_list)
            self.eval_agent.observe(validate(control_msg))
            act = self.eval_agent.act(timeout=self.max_resp_time)
            timeout = self.check_timeout(act)
            if timeout:
                return False
            if 'text' in act:
                self.fluency_scores.append(
                    [int(x) - 1 for x in act['text'].split(',')]
                )
        return True

    def evaluate_consistency(self):
        control_msg = self.get_control_msg()
        control_msg['text'] = CONSISTENCY_MSGS[0]
        control_msg['button_choices'] = '</ROUND>'.join(CONSISTENCY_CHOICES)
        self.eval_agent.observe(validate(control_msg))
        act = self.eval_agent.act(timeout=self.max_resp_time)
        timeout = self.check_timeout(act)
        if timeout:
            return False
        act_choice = CONSISTENCY_CHOICES.index(act.get('text'))
        self.consistency_scores.append(act_choice)
        if act_choice != 0:
            control_msg = self.get_control_msg()
            control_msg['text'] = CONSISTENCY_MSGS[1]
            control_msg['good_rounds'] = True
            control_msg['rounds'] = '</ROUND>'.join(self.dialog_list)
            self.eval_agent.observe(validate(control_msg))
            act = self.eval_agent.act(timeout=self.max_resp_time)
            timeout = self.check_timeout(act)
            if timeout:
                return False
            if 'text' in act:
                self.consistency_scores.append(
                    [int(x) - 1 for x in act['text'].split(',')]
                )
        return True

    def evaluate_persona(self):
        if self.model_agent is not None:
            other_persona = self.model_personas
        else:
            other_persona = self.other_agent.personas
        fake_persona = self.eval_agent.personas_generator.get_persona()
        while fake_persona == other_persona:
            fake_persona = self.eval_agent.personas_generator.get_persona()

        cand_text = []
        for dt in [other_persona, fake_persona]:
            if dt == other_persona:
                is_correct = True
            else:
                is_correct = False
            _text = ''
            for s in dt:
                _text += '<b><span style="color:blue">' + \
                    s.strip() + '</span></b><br>'
            cand_text.append((is_correct, _text))
        random.shuffle(cand_text)

        control_msg = self.get_control_msg()
        control_msg['text'] = PERSONA_MSG.format(
            cand_text[0][1],
            cand_text[1][1]
        )
        control_msg['button_choices'] = '</ROUND>'.join(PERSONA_CHOICES)
        self.eval_agent.observe(validate(control_msg))
        act = self.eval_agent.act(timeout=self.max_resp_time)
        timeout = self.check_timeout(act)
        if timeout:
            return False

        self.persona_scores.append(cand_text[int(act['text']) - 1][0])
        return True

    def parallel_eval_mode(self):
        """Parallel function that shuts one agent down and asks the other
        to do the evaluation if their are two agents. If there is only
        one agent, it performs the evaluation.
        """
        global eval_or_shutdown

        def eval_or_shutdown(agent):
            if self.model_agent is None and agent == self.other_agent:
                control_msg = self.get_control_msg()
                control_msg['text'] = OTHER_AGENT_FINISHED_MSG
                self.other_agent.observe(validate(control_msg))
                # mark eval agent done
                self.eval_agent.mturk_manager.mark_workers_done([self.eval_agent])
                # shutdown other agent
                self.other_agent.shutdown()
            else:
                fin = self.evaluate_engagingness()
                if not fin:
                    return
                if not self.short_eval:
                    fin = self.evaluate_fluency()
                    if not fin:
                        return
                    fin = self.evaluate_consistency()
                    if not fin:
                        return
                fin = self.evaluate_persona()

                return

        Parallel(n_jobs=len(self.agents), backend='threading')(delayed(eval_or_shutdown)(agent) for agent in self.agents)

    def model_observes_itself(self, txt):
        act = {'text': txt, 'episode_done': False}
        self.model_agent.observe(act)

    def episode_done(self):
        return self.chat_done

    def get_instruction(self, agent_id=None, tag='first'):
        if tag == 'start':
            return START_MSG.format(self.n_turn)

        if tag == 'chat_not_done':
            return CHAT_NOT_DONE_MSG.format(self.n_turn + 1 - self.turn_idx)

        if tag == 'timeout':
            return TIMEOUT_MESSAGE

        if tag == 'exceed_min_turns':
            return EXCEED_MIN_TURNS_MSG.format(self.n_turn)

    def save_data(self):
        convo_finished = True
        bad_workers = []
        if not self.opt['is_sandbox']:
            if (self.opt.get('unique_workers') and
                    self.opt.get('unique_qualif_id')):
                # assign qualification to evaluating agent only
                qual = self.opt['unique_qualif_id']
                mutils.give_worker_qualification(
                    self.eval_agent.worker_id,
                    qual,
                    value=None,
                    is_sandbox=False
                )
        if (self.dialog == [] or self.persona_scores == []):
            convo_finished = False

        data_path = self.opt['save_data_path']
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if convo_finished:
            filename = os.path.join(
                data_path, '{}_{}_{}.pkl'.format(
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type
                )
            )
        else:
            filename = os.path.join(
                data_path,
                '{}_{}_{}_incomplete.pkl'.format(
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type
                )
            )
        print(
            self.world_tag,
            ': Data successfully saved at {}.'.format(filename)
        )
        pickle.dump({'dialog': self.dialog,
                     'mtbeam_log': self.mtbeam_log,
                     'dialog_list': self.dialog_list,
                     'other_first': self.other_first,
                     'total_time': time.time() - self.start_time,
                     'workers': [ag.worker_id for ag in self.agents],
                     'hit_id': [ag.hit_id for ag in self.agents],
                     'assignment_id': [ag.assignment_id for ag in self.agents],
                     'human_personas': [ag.personas for ag in self.agents],
                     'model_personas': self.model_personas,
                     'bad_workers': bad_workers,
                     'n_turn': self.n_turn,
                     'engagingness': self.engagingness_scores,
                     'fluency': self.fluency_scores,
                     'consistency': self.consistency_scores,
                     'persona': self.persona_scores}, open(filename, 'wb'))

    def is_msg_tooshortlong(self, act, ag, th_min=3, th_max=20):
        if act['episode_done']:
            return False

        control_msg = self.get_control_msg()

        msg_len = len(act['text'].split(' '))
        if msg_len < th_min:
            control_msg['text'] = TOO_SHORT_MSG.format(th_min)
            ag.observe(validate(control_msg))
            return True
        if msg_len > th_max:
            control_msg['text'] = TOO_LONG_MSG.format(th_max)
            ag.observe(validate(control_msg))
            return True
        return False

    def reset_random(self):
        self.n_turn = np.random.randint(
            self.range_turn[0],
            self.range_turn[1]
        ) + 1

    def check_timeout(self, act):
        if act is None:
            self.chat_done = True
            return True
        if ((act['text'] == '[TIMEOUT]') or (act['text'] == '[RETURNED]') or
                (act['text'] == '[DISCONNECT]')):
            control_msg = self.get_control_msg()
            control_msg['episode_done'] = True
            control_msg['text'] = self.get_instruction(
                agent_id=act['id'],
                tag='timeout'
            )
            for ag in self.agents:
                if ag.id != act['id']:
                    ag.observe(validate(control_msg))
            self.chat_done = True
            return True
        else:
            return False

    def shutdown(self):
        # only need to shut down evaluating agent
        # if more than one agent, other agent shut down previously
        self.eval_agent.shutdown()
