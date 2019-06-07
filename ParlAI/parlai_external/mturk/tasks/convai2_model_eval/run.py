# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.mturk.core.mturk_manager import MTurkManager
import parlai.mturk.core.mturk_utils as mturk_utils

#from parlai_external.mturk.mtdont import MTDONT_LIST
from worlds import ConvAI2Eval, PersonasGenerator, PersonaAssignWorld
from task_config import task_config
import model_configs as mcf

import gc
import datetime
import json
import logging
import os
import sys
import time

MASTER_QUALIF = {
    'QualificationTypeId': '2F1QJWKUDD8XADTFD2Q0G6UTO95ALH',
    'Comparator': 'Exists',
    'RequiredToPreview': True
}


def main():
    """This task consists of an MTurk agent evaluating a ConvAI2 model.
    """
    start_time = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M')
    argparser = ParlaiParser(False, add_model_args=True)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument('-mt', '--max-turns', default=10, type=int,
                           help='maximal number of chat turns')
    argparser.add_argument('--max-resp-time', default=240,
                           type=int,
                           help='time limit for entering a dialog message')
    argparser.add_argument('--max-choice-time', type=int,
                           default=300, help='time limit for turker'
                           'choosing the topic')
    argparser.add_argument('--ag-shutdown-time', default=120,
                           type=int,
                           help='time limit for entering a dialog message')
    argparser.add_argument('-rt', '--range-turn', default='3,5',
                           help='sample range of number of turns')
    argparser.add_argument('--human-eval', type='bool', default=False,
                           help='human vs human eval, no models involved')
    argparser.add_argument('--auto-approve-delay', type=int,
                           default=3600 * 24 * 2,
                           help='how long to wait for auto approval')
    argparser.add_argument('--only-masters', type='bool', default=False,
                           help='Set to true to use only master turks for '
                                'this test eval')
    argparser.add_argument('--unique-workers', type='bool', default=False,
                           help='Each worker must be unique')
    argparser.add_argument('--create-model-qualif', type='bool', default=True,
                           help='Create model qualif so unique eval between'
                                'models.')
    argparser.add_argument('--limit-workers', type=int, default=5,
                           help='max HITs a worker can complete')
    argparser.add_argument('--mturk-log', type=str,
                           default='/home/kulikov/projects/mtbeam/human-eval/mturklogs/{}.log'.format(start_time))
    argparser.add_argument('--short-eval', type='bool', default=True,
                           help='Only ask engagingness question and persona'
                                'question.')
    # persona specific arguments
    argparser.add_argument('--persona-type', type=str, default='self',
                           choices=['self', 'other', 'none'])
    argparser.add_argument('--persona-datatype', type=str, default='test',
                           choices=['train', 'test', 'valid'])
    argparser.add_argument('--max-persona-time', type=int, default=360,
                           help='max time to view persona')
    argparser.add_argument('--model-config', type=str, default=None,
                           help='model config from standalone file')

    def inject_override(opt, override_dict):
        opt['override'] = override_dict
        for k, v in override_dict.items():
            opt[k] = v

    def get_logger(opt):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        fmt = logging.Formatter(
            '%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)
        if 'mturk_log' in opt:
            logfile = logging.FileHandler(opt['mturk_log'], 'a')
            logfile.setFormatter(fmt)
            logger.addHandler(logfile)
        logger.info('COMMAND: %s' % ' '.join(sys.argv))
        logger.info('-' * 100)
        logger.info('CONFIG:\n%s' % json.dumps(opt, indent=4, sort_keys=True))

        return logger


    PREV_MODEL_QUALS = []

    start_opt = argparser.parse_args()

    # Get model config from config file
    config = getattr(mcf, start_opt['model_config'])
    
    #config = mcf.mtbeam1_noiter_noexpand_model_config

    inject_override(start_opt, config)

    if not start_opt.get('human_eval'):
        bot = create_agent(start_opt)
        shared_bot_params = bot.share()
    else:
        shared_bot_params = None

    if not start_opt['human_eval']:
        get_logger(bot.opt)
    else:
        get_logger(start_opt)

    if start_opt['human_eval']:
        folder_name = 'human_eval-{}'.format(start_time)
    else:
        folder_name = '{}-{}'.format(start_opt['model_config'], start_time)

    start_opt['task'] = os.path.basename(
        os.path.dirname(os.path.abspath(__file__)))
    start_opt['save_data_path'] = os.path.join(
        os.getcwd(),
        'data',
        'convAI2_eval',
        folder_name
    )
    start_opt.update(task_config)

    if not start_opt.get('human_eval'):
        mturk_agent_ids = ['PERSON_1']
    else:
        mturk_agent_ids = ['PERSON_1', 'PERSON_2']

    # QUALIFICATION STUFF
    if start_opt['limit_workers'] > 0:
        start_opt['unique_qual_name'] = config['chat_qual_max']
        start_opt['max_hits_per_worker'] = start_opt['limit_workers']

    mturk_manager = MTurkManager(
        opt=start_opt,
        mturk_agent_ids=mturk_agent_ids
    )

    personas_generator = PersonasGenerator(start_opt)

    directory_path = os.path.dirname(os.path.abspath(__file__))

    mturk_manager.setup_server(task_directory_path=directory_path)

    try:
        mturk_manager.start_new_run()
        agent_qualifications = []
        if not start_opt['is_sandbox']:
            # assign qualifications
            if start_opt['only_masters']:
                agent_qualifications.append(MASTER_QUALIF)
            if start_opt['unique_workers']:
                qual_name = 'ChatEval'
                qual_desc = (
                    'Qualification to ensure each worker completes a maximum '
                    'of one of these chat/eval HITs'
                )
                qualification_id = \
                    mturk_utils.find_or_create_qualification(qual_name, qual_desc,
                                                             False)
                print('Created qualification: ', qualification_id)
                UNIQUE_QUALIF = {
                    'QualificationTypeId': qualification_id,
                    'Comparator': 'DoesNotExist',
                    'RequiredToPreview': True
                }
                start_opt['unique_qualif_id'] = qualification_id
                agent_qualifications.append(UNIQUE_QUALIF)
            elif start_opt['create_model_qualif']:
                qual_name = config['chat_qual_name']
                qual_desc = (
                    'Qualification to ensure workers complete only a certain'
                    'number of these HITs'
                )
                qualification_id = \
                    mturk_utils.find_or_create_qualification(qual_name, qual_desc,
                                                             False)
                print('Created qualification: ', qualification_id)
                start_opt['unique_qualif_id'] = qualification_id
                for qual_name in PREV_MODEL_QUALS:
                    qualification_id = \
                        mturk_utils.find_or_create_qualification(
                            qual_name,
                            qual_desc,
                            False
                        )
                    QUALIF = {
                        'QualificationTypeId': qualification_id,
                        'Comparator': 'DoesNotExist',
                        'RequiredToPreview': True
                    }
                    agent_qualifications.append(QUALIF)

        mturk_manager.create_hits(qualifications=agent_qualifications)

        #with open('/home/kulikov/code/mt-beam-parlai/mturk/mtdont.txt', 'r') as f:
        #    MTDONT_LIST = [i.rstrip() for i in f.readlines()]

        #if not start_opt['is_sandbox']:
        #    # ADD BLOCKED WORKERS HERE
        #    blocked_worker_list = MTDONT_LIST
        #    for w in blocked_worker_list:
        #        try:
        #            print('Soft Blocking {}\n'.format(w))
        #            mturk_manager.soft_block_worker(w)
        #        except:
        #            print('Did not soft block worker:', w)
        #        time.sleep(0.1)

        def run_onboard(worker):
            worker.personas_generator = personas_generator
            world = PersonaAssignWorld(start_opt, worker)
            world.parley()
            world.shutdown()

        def check_worker_eligibility(worker):
            return True

        def assign_worker_roles(workers):
            for index, worker in enumerate(workers):
                worker.id = mturk_agent_ids[index % len(mturk_agent_ids)]

        mturk_manager.set_onboard_function(onboard_function=run_onboard)
        mturk_manager.ready_to_accept_workers()

        def run_conversation(mturk_manager, opt, workers):
            conv_idx = mturk_manager.conversation_index
            world = ConvAI2Eval(
                opt=start_opt,
                agents=workers,
                range_turn=[int(s)
                            for s in start_opt['range_turn'].split(',')],
                max_turn=start_opt['max_turns'],
                max_resp_time=start_opt['max_resp_time'],
                model_agent_opt=shared_bot_params,
                world_tag='conversation t_{}'.format(conv_idx),
                agent_timeout_shutdown=opt['ag_shutdown_time'],
            )
            world.reset_random()
            while not world.episode_done():
                world.parley()
            world.save_data()

            world.shutdown()
            gc.collect()

        mturk_manager.start_task(
            eligibility_function=check_worker_eligibility,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation
        )

    except BaseException:
        raise
    finally:
        mturk_manager.expire_all_unassigned_hits()
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()
