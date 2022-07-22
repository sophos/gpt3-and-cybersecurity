# Spam detector

Invoke the following command to identify a message as spam or ham.

```
python spam_detector.py --message="test message"
```

The above command generates a prompt using two in-context samples from [a training dataset](./spam_data/experiment_1/train_2.csv). The default training file can be changed with its --path_train_data option.

<br>

Invoke the following command to evaluate spam detection approaches which include traditional ML models and novel GPT-3 models.

```
python spam_detector.py --run_type=evaluate_approaches --path_data_folder=spam_data --num_experiments=5
```

<br>
Demo examples are available in the [notebook](https://github.com/sophos-ai/gpt3-cybersecurity/blob/master/spam_detector/spam_demo.ipynb).

# Spam dataset
spam_data folder provides training and test datasets which were randomly sampled from a spam datast, The [spam data](./spam_data/SMSSpamCollection) is from [UCI SMS Spam collection data set](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).
