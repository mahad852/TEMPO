import torch
import torch.nn as nn

from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils.rev_in import RevIn
from peft import get_peft_model, LoraConfig, TaskType

from models.TEMPO import print_trainable_parameters, TEMPO

criterion = nn.MSELoss()

class ECG_TEMPO(nn.Module):
    
    def __init__(self, configs, device):
        super(TEMPO, self).__init__()
        self.is_gpt = configs.is_gpt
        self.pretrain = configs.pretrain
        
        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2_ecg_features = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model  
                self.gpt2_raw_data = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model  
            else:
                print("------------------no pretrain------------------")
                self.gpt2_ecg_features = GPT2Model(GPT2Config())
                self.gpt2_raw_data = GPT2Model(GPT2Config())
                

            self.gpt2_ecg_features.h = self.gpt2_ecg_features.h[:configs.gpt_layers]
           
            self.prompt = configs.prompt

            # 
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.gpt2_ecg_features_token = self.tokenizer(text="Predict arrythmia rhythm type given the following features extracted from the ecg data: ['VentricularRate', ' AtrialRate', ' QRSDuration', ' QTInterval', ' QTCorrected', ' RAxis', ' TAxis', ' QRSCount', ' QOnset', ' QOffset', ' TOffset'] in order", 
                                                          return_tensors="pt").to(device)
            self.gpt2_raw_data_token = self.tokenizer(text="Predict arrythmia rhythm type given the ECG lead 2 signal data", return_tensors="pt").to(device)

            self.token_len_ecg_features = len(self.gpt2_ecg_features_token['input_ids'][0])
            self.token_len_raw_data = len(self.gpt2_raw_data_token['input_ids'][0])

        self.in_layer_ecg_features = nn.Linear(1, configs.d_model)
        self.in_layer_raw_data = nn.Linear(1, configs.d_model)

        if configs.prompt == 1:
            # print((configs.d_model+9) * self.patch_num)
            self.use_token = configs.use_token
            if self.use_token == 1: # if use prompt token's representation as the forecasting's information
                    self.out_layer_ecg_features = nn.Linear(configs.d_model * (11 + self.token_len_ecg_features), configs.pred_len)
                    self.out_layer_raw_data = nn.Linear(configs.d_model * (5000 + self.token_len_raw_data), configs.pred_len)
            else:
                self.out_layer_ecg_features = nn.Linear(configs.d_model * 11, configs.pred_len)
                self.out_layer_raw_data = nn.Linear(configs.d_model * 5000, configs.pred_len)

            self.prompt_layer_ecg_features = nn.Linear(configs.d_model, configs.d_model)
            self.prompt_layer_raw_data = nn.Linear(configs.d_model, configs.d_model)

            for layer in (self.prompt_layer_ecg_features, self.prompt_layer_raw_data):
                layer.to(device=device)
                layer.train()
        else:
            self.out_layer_ecg_features = nn.Linear(configs.d_model * 11, configs.pred_len)
            self.out_layer_raw_data = nn.Linear(configs.d_model * 5000, configs.pred_len)

        
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2_ecg_features.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        config = LoraConfig(
            # task_type=TaskType.CAUSAL_LM, # causal language model
            r=16,
            lora_alpha=16,
            # target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="lora_only",               # bias, set to only lora layers to train
            # modules_to_save=["classifier"],
        )
         
        self.gpt2_ecg_features = get_peft_model(self.gpt2_ecg_features, config)
        print_trainable_parameters(self.gpt2_ecg_features)


        for layer in (self.gpt2_ecg_features, self.in_layer_ecg_features, self.out_layer_ecg_features, \
                      self.in_layer_raw_data, self.out_layer_raw_data, self.in_layer_noise, self.out_layer_noise):
            layer.to(device=device)
            layer.train()

        # for layer in (self.map_trend, self.map_season, self.map_resid):
        #     layer.to(device=device)
        #     layer.train()
        
        self.classifier = nn.Linear(configs.pred_len, 10)
        
        self.cnt = 0

        self.num_nodes = configs.num_nodes
        self.rev_in_ecg_features = RevIn(num_features=self.num_nodes).to(device)
        self.rev_in_raw_data = RevIn(num_features=self.num_nodes).to(device)
        


    def get_emb(self, x, tokens=None, type = 'ecg_features'):
        if tokens is None:
            x = self.gpt2_ecg_features(inputs_embeds =x).last_hidden_state if type == 'ecg_features' else self.gpt2_raw_data(inputs_embeds =x).last_hidden_state
        else:
            [a,b,c] = x.shape
            if type == 'ecg_features': 
                prompt_x = self.gpt2_ecg_features.wte(tokens)
                prompt_x = prompt_x.repeat(a,1,1)
                prompt_x = self.prompt_layer_ecg_features(prompt_x)                

            elif type == 'raw_data':
                prompt_x = self.gpt2_raw_data.wte(tokens)
                prompt_x = prompt_x.repeat(a,1,1)
                prompt_x = self.prompt_layer_raw_data(prompt_x)
            
            x = torch.cat((prompt_x, x), dim=1)
                
        return x


    def forward(self, x: torch.Tensor, ecg_features: torch.Tensor, itr, test=False):
        B, L, M = x.shape # 4, 512, 1

       
        x = self.rev_in_ecg_features(x, 'norm')

        original_x = x

        ecg_features = ecg_features.unsqueeze(2)
        x = x.unsqueeze(2)

        ecg_features = self.in_layer_ecg_features(ecg_features) # 4, 64, 768
        if self.is_gpt and self.prompt == 1:
            ecg_features = self.get_emb(ecg_features, self.gpt2_ecg_features_token['input_ids'], 'ecg_features')
        else:
            ecg_features = self.get_emb(ecg_features)

        x = self.in_layer_raw_data(x) # 4, 64, 768
        if self.is_gpt and self.prompt == 1:
            x = self.get_emb(x, self.gpt2_raw_data_token['input_ids'], 'raw_data')
        else:
            x = self.get_emb(x)

        x_all = torch.cat((ecg_features, x), dim=1)

        x = self.gpt2_ecg_features(inputs_embeds =x_all).last_hidden_state 
        
        if self.prompt == 1:
            ecg_features  = x[:, :self.token_len_ecg_features + 11, :]  
            x  = x[:, self.token_len_ecg_features + 11:, :]  
            
            if self.use_token == 0:
                ecg_features = ecg_features[:, self.token_len_ecg_features:, :]
                x = x[:, self.token_len_raw_data:, :]
            
        
        ecg_features = self.out_layer_ecg_features(ecg_features.reshape((x.shape[0], -1)))
        x = self.out_layer_raw_data(x.reshape(((x.shape[0], -1))))        

        outputs = ecg_features + x
        outputs = self.rev_in_trend(outputs, 'denorm')

        return self.classifier(outputs)