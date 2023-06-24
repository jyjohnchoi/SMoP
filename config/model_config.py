class Config:
    def __init__(self):
        self.models = {'gpt3': 'gpt3',
                       'gpt2-large': 'gpt2-large',
                       
                       'roberta-base': 'roberta-base',
                       'roberta-large': 'roberta-large',
                       
                       't5-small': 't5-small', 
                       't5-base': 't5-base', 
                       't5-large': 't5-large', 
                       't5-3b': 't5-3b', 
                       't5-11b': 't5-11b', 

                       't5-v1_1-small': 'google/t5-v1_1-small', 
                       't5-v1_1-base': 'google/t5-v1_1-base', 
                       't5-v1_1-large': 'google/t5-v1_1-large', 
                       't5-v1_1-3b': 'google/t5-v1_1-xl', 
                       't5-v1_1-11b': 'google/t5-v1_1-xxl', 

                       't5-lm-small': 'google/t5-small-lm-adapt',
                       't5-lm-base': 'google/t5-base-lm-adapt',
                       't5-lm-large': 'google/t5-large-lm-adapt',
                       't5-lm-3b': 'google/t5-xl-lm-adapt',
                       't5-lm-11b': 'google/t5-xxl-lm-adapt',

                       't0-3b': 'bigscience/T0_3B', 
                       't0-11b': 'bigscience/T0pp', 

                       'uqa-t5-small': 'allenai/unifiedqa-t5-small', 
                       'uqa-t5-base': 'allenai/unifiedqa-t5-base', 
                       'uqa-t5-large': 'allenai/unifiedqa-t5-large', 
                       'uqa-t5-3b': 'allenai/unifiedqa-t5-3b', 
                       'uqa-t5-11b': 'allenai/unifiedqa-t5-11b',

                       'pegasus-large': 'google/pegasus-large',
                       'unicorn-11b': 'models/unicorn-11b',
                    }