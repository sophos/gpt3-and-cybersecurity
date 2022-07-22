# Command analyzer

Invoke the following command to translate a command line into a natural language description.

```
python command_analyzer.py --cmd="command line" --tags=="comma seperated tags"
```

<br>
Invoke the following command to evaluate our back-translation approaches.

Unzip the cmd_data/results_data_cmd_tag_and_gold_reference_desc.json.zip file with a password which is this repository name.

```
python command_analyzer.py --run_type=evaluate_approaches --path_output_json="cmd_data/results_data_cmd_tag_and_gold_reference_desc.json" --path_input_json="cmd_data/data_cmd_tag_and_gold_reference_desc.json"
```

<br>
Demo examples are available in the [notebook](https://github.com/sophos-ai/gpt3-cybersecurity/blob/master/command_analyzer/command_demo.ipynb). 

# Dataset
cmd_data folder provides a dataset which includes command lines, tags and reference descriptions.
