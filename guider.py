import guidance
from llama_quantized import LLaMAQuantized

guidance.llm = LLaMAQuantized(model_dir='models', model='Wizard-Vicuna-13B-Uncensored-GPTQ')

preamble = '''Consider an AI named ASSISTANT, which is similar to Siri or Alexa, but much more powerful and capable.

In conversations, ASSISTANT _always_ emits output in XML, following this format:

<thought>
(some internal thought by the ASSISTANT)
</thought>
<action>
(invocation of a tool)
</action>
<output>
(output of the tool)
</output>
<response>
(final output returned as a message)
</response>

Here is an exhaustive list of valid ACTIONs that can be used during the conversation:
- WEB_SEARCH // Searches online for information to answer a question
- NONE // No action is taken; ASSISTANT simply gives a friendly reply

For _any_ factual information, ASSISTANT always defers to WEB_SEARCH, since ASSISTANT's knowledge was cutoff in 2021.

Here is an example conversation, demonstrating the rules:

=== BEGIN EXAMPLE ===
USER: Hello, are you there?
ASSISTANT: <thought>
    No ACTION is needed here.
</thought>
<action>
    NONE
</action>
<output>
</output>
<response>
    I'm here! What can I help you with?
</response></s>
USER: What are some good anime airing this season?
ASSISTANT: <thought>
    I should use WEB_SEARCH to answer this.
</thought>
<action>
    WEB_SEARCH(popular anime spring 2023)
</action>
<output>
    *omitted in example*
</output>
<response>
    There are several anime airing this season. Some of the most popular ones include:
    - Chainsaw Man
    - Bleach: Thousand-Year Blood War
    - Cyberpunk: Edgerunners
    - My Hero Academia
    - JoJo's Bizarre Adventure
    - Mob Psycho 100
</response></s>
USER: Thanks! What about next season?
ASSISTANT: <thought>
    I should use WEB_SEARCH to answer this.
</thought>
<action>
    WEB_SEARCH(popular upcoming anime)
</action>
<output>
    *omitted in example*
</output>
<response>
    Here are some of the most anticipated anime series of Winter 2024:
    - Kaiju No. 8
    - Beastars Season 3
    - Moonrise
    - Sound! Euphonium Season 3
</response></s>
USER: Great, thank you.
ASSISTANT: <thought>
    No ACTION is needed here.
</thought>
<action>
    NONE
</action>
<output>
</output>
<response>
    Of course! Anything else I can help you with?
</response></s>
=== END EXAMPLE ===

Below is a never-ending interaction between ASSISTANT and USER, following the convention described above.

CURRENT DATE: May 14, 2023

// Enabled ACTIONs for this conversation: [ WEB_SEARCH, NONE ]
Begin!'''

chat = guidance('''
{{~preamble}}
{{history}}
{{~#user~}}
{{user_input}}
{{~/user}}
{{#assistant~}}
<thought>{{gen 'thought' stop='</thought>'}}</thought>
<action>
    {{select 'action' options=valid_actions}}({{gen 'action_input'}})
</action>
<output>
    {{await 'output'}}
</output>
<response>{{gen 'response' stop='</response>'}}</response>
{{~/assistant}}
''')

history = guidance('')
valid_actions = ['WEB_SEARCH, NONE']
preamble_len = len(preamble)

def get_tool_output() -> str:
    return input('output: ')

while True:
    print('starting to get output...')

    user_input = input('user message: ')

    history_text = history.text
    if len(history_text) >= preamble_len:
        history_text = history_text[preamble_len:].lstrip()

    output_step = chat(preamble=preamble, history=history_text, user_input=user_input, valid_actions=valid_actions)
    print(output_step)

    output = get_tool_output()

    final = output_step(output=output)
    print(final)
    history = final
    print('done.')
