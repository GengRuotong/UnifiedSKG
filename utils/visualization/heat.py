import torch

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def compute_correlation_matrix(self, input_ids):        
    _, seq_len = input_ids.size()
    hidden = self.model.base_model(input_ids).last_hidden_state
    #print (hidden)
    norm_hidden = hidden / hidden.norm(dim=2, keepdim=True)
    correlation_matrix = torch.matmul(norm_hidden, norm_hidden.transpose(1,2)).view(seq_len, seq_len)
    return correlation_matrix.detach().numpy()

# to produce similarity matrix heatmap
def save_token_similarity_map(self, input_ids, save_name):
        input_ids = torch.LongTensor(input_ids).view(1, -1)
        correlation_matrix = self.compute_correlation_matrix(input_ids)
        df = pd.DataFrame(correlation_matrix)
        df.to_string(index=False)
        df.style.hide_index()
        df.style.hide_index()
        sns.heatmap(df, cmap="Blues", xticklabels=False, yticklabels=False)
        plt.savefig(save_name, format='png', dpi=500, bbox_inches = 'tight')
        plt.show()