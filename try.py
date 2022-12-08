# task_names_list = [self.args.task_id2task_name[task_id.item()] for task_id in kwargs.pop("task_ids")]
load_multiple_prefix_module_weights_from = [('maicai_0','checkpoint1'),('youxuan_1', 'checkpoint2')]
if load_multiple_prefix_module_weights_from:
    for task_name, module_weight_location in load_multiple_prefix_module_weights_from:
        print(task_name)
        print(module_weight_location)
    task_id2task_name = sorted(['_'.join(task_name.split('_')[:-1]) for task_name, module_weight_location in load_multiple_prefix_module_weights_from])
    print(task_id2task_name)
        
    task_name2task_id = {task_name: task_id for task_id, task_name in enumerate(task_id2task_name)}
    print(task_name2task_id)