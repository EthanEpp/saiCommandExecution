import onnx
from onnx import helper
from onnx import TensorProto

# Load the ONNX model
model_path = "/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/bert_layer.onnx"
onnx_model = onnx.load(model_path)

# Check for any tensors with data type bool
bool_tensors = []
for value_info in onnx_model.graph.value_info:
    if value_info.type.tensor_type.elem_type == onnx.TensorProto.BOOL:
        bool_tensors.append(value_info.name)

print("Boolean tensors in the model:", bool_tensors)

# model_path = '/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/bert_layer.onnx'

# # Load the ONNX model
# onnx_model = onnx.load(model_path)

# # List of nodes to modify
# nodes_to_modify = []

# # Check for bool outputs and collect nodes to modify
# for node in onnx_model.graph.node:
#     for output in node.output:
#         for value_info in onnx_model.graph.value_info:
#             if value_info.name == output and value_info.type.tensor_type.elem_type == onnx.TensorProto.BOOL:
#                 nodes_to_modify.append((node, output))

# # Add Cast nodes for each bool output
# for node, output in nodes_to_modify:
#     cast_output_name = output + "_cast"
#     cast_node = helper.make_node(
#         "Cast",
#         inputs=[output],
#         outputs=[cast_output_name],
#         to=TensorProto.INT32,  # Cast bool to int32
#         name=node.name + "_cast"
#     )
#     onnx_model.graph.node.append(cast_node)

#     # Update references in subsequent nodes
#     for next_node in onnx_model.graph.node:
#         for idx, input_name in enumerate(next_node.input):
#             if input_name == output:
#                 next_node.input[idx] = cast_output_name

# # Save the modified ONNX model
# modified_model_path = "/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/modified_bert_layer.onnx"
# onnx.save(onnx_model, modified_model_path)
# print(f"Modified ONNX model saved to {modified_model_path}")
