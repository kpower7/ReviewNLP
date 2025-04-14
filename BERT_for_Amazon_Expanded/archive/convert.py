from transformers import AutoConfig

tf_config = AutoConfig.from_pretrained("/home/gridsan/kpower/BERT_for_Amazon_Expanded/bert_pretrained", from_tf=True)
torch_config = AutoConfig.from_pretrained("/home/gridsan/kpower/BERT_for_Amazon_Expanded/bert_pytorch_converted")

print(tf_config)  # Compare with:
print(torch_config)
