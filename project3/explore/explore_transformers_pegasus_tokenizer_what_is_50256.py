from transformers import  AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('google/pegasus-cnn_dailymail', use_fast=False)
print(tokenizer.decode(50256))
print(tokenizer.special_tokens_map)
print(tokenizer.special_tokens_map_extended)
# check the ids of the special tokens
print(tokenizer.pad_token_id)
print(tokenizer.eos_token_id)
print(tokenizer.unk_token_id)
print(tokenizer.mask_token_id)
"""
checkups
{'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>', 'mask_token': '<mask_2>', 'additional_special_tokens': ['<mask_1>', '<unk_2>', '<unk_3>', '<unk_4>', '<unk_5>', '<unk_6>', '<unk_7>', '<unk_8>', '<unk_9>', '<unk_10>', '<unk_11>', '<unk_12>', '<unk_13>', '<unk_14>', '<unk_15>', '<unk_16>', '<unk_17>', '<unk_18>', '<unk_19>', '<unk_20>', '<unk_21>', '<unk_22>', '<unk_23>', '<unk_24>', '<unk_25>', '<unk_26>', '<unk_27>', '<unk_28>', '<unk_29>', '<unk_30>', '<unk_31>', '<unk_32>', '<unk_33>', '<unk_34>', '<unk_35>', '<unk_36>', '<unk_37>', '<unk_38>', '<unk_39>', '<unk_40>', '<unk_41>', '<unk_42>', '<unk_43>', '<unk_44>', '<unk_45>', '<unk_46>', '<unk_47>', '<unk_48>', '<unk_49>', '<unk_50>', '<unk_51>', '<unk_52>', '<unk_53>', '<unk_54>', '<unk_55>', '<unk_56>', '<unk_57>', '<unk_58>', '<unk_59>', '<unk_60>', '<unk_61>', '<unk_62>', '<unk_63>', '<unk_64>', '<unk_65>', '<unk_66>', '<unk_67>', '<unk_68>', '<unk_69>', '<unk_70>', '<unk_71>', '<unk_72>', '<unk_73>', '<unk_74>', '<unk_75>', '<unk_76>', '<unk_77>', '<unk_78>', '<unk_79>', '<unk_80>', '<unk_81>', '<unk_82>', '<unk_83>', '<unk_84>', '<unk_85>', '<unk_86>', '<unk_87>', '<unk_88>', '<unk_89>', '<unk_90>', '<unk_91>', '<unk_92>', '<unk_93>', '<unk_94>', '<unk_95>', '<unk_96>', '<unk_97>', '<unk_98>', '<unk_99>', '<unk_100>', '<unk_101>', '<unk_102>']}
pad: 0
eos: 1
unk: 105
mask: 3
"""