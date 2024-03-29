# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

task_config = {}

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Chat with and rate a fellow conversationalist!'


"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config['hit_description'] = (
    'Play a character and chat with another user. At the end of the chat, ' +
    'you will be asked several questions to evaluate the other user.'
)


"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'chat,dialog'


"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config['task_description'] = \
    """
<br>
<b><h2>Task Description</h2></b>
<br>
<b>In this task, you will chat with another user playing the part of a given
character.</b>. For example, your given character could be:
<br>
<br>
I am a vegetarian. I like swimming. My father used to work for Ford.
My favorite band is Maroon5. I got a new job last month, which is about
advertising design.
<br>
<br>
Chat with the other user <b><span style="color:red">naturally</span></b> and
try to get to know each other, i.e. both ask questions and answer questions
of your chat partner while sticking to your given character.
<br>
<br>
<b>If you complete the task, you will receive $0.90</b>. It may take up to
48 hours to review the HITs, so please allow that much time to pass before
payment. After completion, you may be assigned a qualification that prevents you
from working on more if you have completed enough of these HITs.
<br>
<br>
After a given number of turns, you may be asked a few questions in order to
evaluate your partner.
<br>
<br>
<b>If your partner answers poorly, change topic.</b> Do not linger on their
poor response. Instead, mention this during the evaluation section.
<br>
<br>
<h4><span style="color:blue"><b>Close Window/Timeout/Return HIT</b></span></h4>
Once the conversation has started, close window/timeout or return HIT during
the chat will result in <b>HIT EXPIRED</b> to you and NO reward paid.
<br>
<br>
<h4><span style="color:blue"><b>Important Notice</b></span></h4>
1. <b>Be aware the conversations you have will be made public, so act as you
would e.g. on a public social network like Twitter.</b>
<br>
2. Please do not send long messages: messages cannot exceed 30 words.
<br>
3. Please do not reference the task or MTurk itself during the conversation,
but speak naturally to the other person.
<br>
4. Please do not send any message that could make others uncomfortable,
including any level of discrimination, racism, sexism and offensive
religious/politics comments, otherwise the submission will be rejected.
<br>
<br>
Note: the user you are chatting with may be a human or a bot.
"""
