import torch

def build_mask_matrix(loss_matrix, valid_len_list):
    '''
        prefix_len: the length of prefix that we do not want to compute CL loss for.

        (1) if a sequence of length 4 contains zero padding token (i.e., the valid length is 4),
            then the loss padding matrix looks like
                 [0., 1., 1., 1.],
                 [1., 0., 1., 1.],
                 [1., 1., 0., 1.],
                 [1., 1., 1., 0.]

        (2) if a sequence of length 4 contains 1 padding token (i.e., the valid length is 3),
            then the loss padding matrix looks like
                 [0., 1., 1., 0.],
                 [1., 0., 1., 0.],
                 [1., 1., 0., 0.],
                 [0., 0., 0., 0.]
    '''

    bsz, seqlen, perspect_num, _ = loss_matrix.shape
    base_mask = torch.ones(perspect_num, perspect_num) - torch.eye(perspect_num, perspect_num)
    base_mask = base_mask.type(torch.FloatTensor)
    assert bsz == len(valid_len_list)
    valid_loss_list, base_mask_list = [], []
    for i in range(bsz):
        valid_len = int(valid_len_list[i])
        if valid_len > seqlen:
            valid_len = seqlen
        valid_loss_matrix = loss_matrix[i][ :valid_len, :, :] 
        base_mask_matrix = torch.stack([base_mask]*valid_len, dim=0)
        valid_loss_list.append(valid_loss_matrix)
        base_mask_list.append(base_mask_matrix)
    all_valid_loss_matrix = torch.cat(valid_loss_list, dim=0)
    all_base_mask_matrix = torch.cat(base_mask_list, dim=0)
    return all_valid_loss_matrix, all_base_mask_matrix
    

def contrastive_loss(margin, perspect_tensor, input_ids, pad_token_id):
        cosine_matrix = torch.matmul(perspect_tensor, perspect_tensor.transpose(2,3)) 
        bsz, seqlen, perspect_num, _ = cosine_matrix.size()
        gold_score = torch.diagonal(cosine_matrix, offset=0, dim1=2, dim2=3)
        gold_score = torch.unsqueeze(gold_score, dim=-1)
        assert gold_score.size() == torch.Size([bsz, seqlen, perspect_num, 1])
        difference_matrix = gold_score - cosine_matrix
        assert difference_matrix.size() == torch.Size([bsz, seqlen, perspect_num, perspect_num])
        loss_matrix = margin - difference_matrix # bsz x seqlen x perspect_num x perspect_num
        loss_matrix = torch.nn.functional.relu(loss_matrix)

        # calculate average loss_matrix
        ### input mask
        input_mask = torch.ones_like(input_ids).type(torch.FloatTensor)
        if loss_matrix.is_cuda:
            input_mask = input_mask.cuda(loss_matrix.get_device())
        input_mask = input_mask.masked_fill(input_ids.eq(pad_token_id), 0.0)

        if loss_matrix.is_cuda:
            input_mask = input_mask.cuda(loss_matrix.get_device())

        valid_len_list = torch.sum(input_mask, dim = -1).tolist()
        
        all_valid_loss_matrix, all_base_mask_matrix = build_mask_matrix(loss_matrix, valid_len_list)
        
        if cosine_matrix.is_cuda:
            all_base_mask_matrix = all_base_mask_matrix.cuda(cosine_matrix.get_device())
        if loss_matrix.is_cuda:
            all_valid_loss_matrix =  all_valid_loss_matrix.cuda(loss_matrix.get_device())
        masked_loss_matrix = torch.matmul(all_valid_loss_matrix, all_base_mask_matrix)
    
        cl_loss = torch.sum(masked_loss_matrix) / torch.sum(all_base_mask_matrix)
        return cl_loss