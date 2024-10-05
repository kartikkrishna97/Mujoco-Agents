
'''
This file contains the configs used for Model creation and training. You need to give your best hyperparameters and the configs you used to get the best results for 
every environment and experiment.  These configs will be automatically loaded and used to create and train your model in our servers.
'''
#You can add extra keys or modify to the values of the existing keys in bottom level of the dictionary.
#DONT CHANGE THE OVERALL STRUCTURE OF THE DICTIONARY. 

configs = {
    

    'Hopper-v4': {
        "imitation":{
            #You can add or change the keys here
              "hyperparameters": {
                'hidden_size': 32,
                'n_layers': None,
                'batch_size': 512, 
                'max length': 400,
                'num trajectories': 50,
            },
            "num_iteration": 200,

        },

        "RL":{
            #You can add or change the keys here
               "hyperparameters": {
                'hidden_size': None,
                'n_layers': None,
                'batch_size': None, 
                
            },
            "num_iteration": 100000,
        },
    },
    
    
    'Ant-v4': {
        "imitation":{
            #You can add or change the keys here
              "hyperparameters": {
                'hidden_size': 128,
                'n_layers': None,
                'batch_size': 2048,
                'max length': 500,
                'num trajectories': 50, 
            },
            "num_iteration": 150,

        },
        
        "SB3":{
            #You can add or change the keys here
                "hyperparameters": {
                    "total_timesteps": None,
                },            
        },
    },
    
    'PandaPush-v3': {
        "SB3":{
            #You can add or change the keys here
                "hyperparameters": {
                    "total_timesteps": None,
                },            
        },
    },

}