from pathlib import Path

import onnx
from onnx.tools.update_model_dims import update_inputs_outputs_dims


root = Path(__file__).resolve().parent.parent
source = root / 'work' / 'centerface-source.onnx'
destination = root / 'public' / 'centerface.onnx'
model = onnx.load(source)

input_dims = {
    node.name: [dimension.dim_value for dimension in node.type.tensor_type.shape.dim]
    for node in model.graph.input
}
output_dims = {
    node.name: [dimension.dim_value for dimension in node.type.tensor_type.shape.dim]
    for node in model.graph.output
}
input_dims['input.1'] = ['B', 3, 'H', 'W']
output_dims.update({
    '537': ['B', 1, 'h', 'w'],
    '538': ['B', 2, 'h', 'w'],
    '539': ['B', 2, 'h', 'w'],
    '540': ['B', 10, 'h', 'w'],
})

dynamic_model = update_inputs_outputs_dims(model, input_dims, output_dims)
onnx.checker.check_model(dynamic_model)
onnx.save(dynamic_model, destination)
print(f'Saved dynamic CenterFace model to {destination}')
