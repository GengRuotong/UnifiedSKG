'''
maicai
train: 16659 test: 2082 valid: 2083 domain average len: 241.2994777597695
total 20824
maoyanyanchu
train: 8695 test: 1086 valid: 1088 domain average len: 204.63645773433007
total 10869
taxi-yonghu
train: 25006 test: 3125 valid: 3127 domain average len: 299.4909621690794
total 31258
waimai
train: 28636 test: 3579 valid: 3580 domain average len: 255.90951948596174
total 35795
youxuan
train: 22965 test: 2870 valid: 2872 domain average len: 189.92471151752667
total 28707
total average len: 244.97640274222496
'''
import json

file_name = ['maicai', 'maoyanyanchu', 'taxi-yonghu', 'waimai', 'youxuan']
base_path = 'data/sample_datas_wo_prefix/id_record/' 
data_base_path = 'data/sample_datas_wo_prefix/'
train_domain_sum = 0
train_data_len = 0
average_list = [241.2994777597695, 204.63645773433007, 299.4909621690794, 255.90951948596174, 189.92471151752667]
for i in range(5):
    input_path = base_path + file_name[i] + '.json'
    data_path = data_base_path + 'mt_' + file_name[i] + '/train.json'
    with open(input_path, 'r') as f:
        f_out = json.loads(f.read())
        print(f_out['file_name'])
        sum1 = len(f_out['train_ids'])
        # sum2 = len(f_out['test_ids'])
        # sum3 = len(f_out['valid_ids'])
        # sum = sum1 + sum2 + sum3
        train_domain_sum += sum1
        # print('train:', sum1, 'test:', sum2, 'valid:', sum3)
        # print("total", sum)
    with open(data_path, 'r') as f_data:
        f_data_out = f_data.readlines()
        data_sum = 0
        above_average_sum = 0
        for j in range(len(f_data_out)):
            f_text = json.loads(f_data_out[j])['text']
            data_sum += len(f_text)
            if len(f_text) > average_list[i]:
                above_average_sum += 1
        train_data_len += data_sum
        average_input_len = data_sum / sum1
        above_average_ratio = above_average_sum / sum1
        print("domain average:", average_input_len)
        print("above_average_ratio:", above_average_ratio)
print("total average:", train_data_len / train_domain_sum)
print('finish!')
