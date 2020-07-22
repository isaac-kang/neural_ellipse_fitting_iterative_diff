import json
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
from os.path import abspath
from os.path import basename


path = 'imagenet_224x224_mobilenet_v1_per-axis-post-training-integer_dm_1.0.json'
model = load_json(path)
ops = [op['builtin_code'] for op in model['operator_codes']]

print('list all operators: \n%s' % ops)


tensors = model['subgraphs'][0]['tensors']
print()
print('total number of tensors: %s' % len(tensors))
print()

for tensor in tensors:
    print('name: %s' % tensor['name'])
    print('shape: %s' % tensor['shape'])
    if 'type' in tensor:
        print('type: %s' % tensor['type'])
    if 'quantization' in tensor:
        quant_params = tensor['quantization']
        if 'scale' in quant_params:
            scale = quant_params['scale']
            print(scale)
            if len(scale) > 1:
                print('(per-axis)')
            else:
                print('(per-tensor)')
            print('\t#scale: %d ' % len(scale))
        if 'zero_point' in quant_params:
            zero_point = quant_params['zero_point']
            print(zero_point)
            print('\t#zero_point: %d' % len(zero_point))
    print()