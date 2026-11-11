"""Use frozen official loops and audited kwargs for each native bank kind."""
import importlib
from types import SimpleNamespace
from pathlib import Path
from experiments.root_gqa_bank_qualification_v1_20261026 import worker as u
from experiments.root_native_trained_eval_qual_v3_20261026 import worker as loader
from experiments.root_aces_scienceqa_start_v1_20261025 import audit as science
from experiments.session6_native_start_execution_v6_20261024 import kernel
from experiments.session6_native_start_execution_v6_20261024.audit import tensor_digest
from experiments.session6_official_eval_v2_20261024.journal import Journal

def expected_science(row,loaded,store):
    import torch
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images,tokenizer_image_token
    tokenizer,model,processor,_=loaded;conv=conv_templates['vicuna_v1'].copy()
    conv.append_message(conv.roles[0],science.prompt_text(row));conv.append_message(conv.roles[1],None);prompt=conv.get_prompt()
    tokens=tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0)
    rgb=store.rgb(row['native_input']['image'])
    try:
        pixels=process_images([rgb],processor,model.config)[0].unsqueeze(0).to(torch.float16)
        return dict(prompt_utf8_sha256=u.sha(prompt.encode()),input_ids_sha256=tensor_digest(tokens),input_token_count=tokens.shape[-1],
            pixel_values_sha256=tensor_digest(pixels),image_size=list(rgb.size),raw_prompt=science.prompt_text(row))
    finally:rgb.close()

def validate_entry(bank,entry,row,function):
    (science.validate_entry if bank=='ScienceQA_image' else loader.audit.validate_entry)(entry,row,function)

class AuditedJournal(Journal):
    def append(self,key,raw,record):
        validate_entry(self.roster[key]['bank'],dict(semantic_key=key,raw=raw,audit=record),self.roster[key],self.function)
        return super().append(key,raw,record)

def native(bank,cfg,rows,journal,loaded,store,current,deadline):
    if bank!='ScienceQA_image':return loader.native(cfg,rows,journal,loaded,store,current,deadline)
    missing=journal.missing();recorder=science.Recorder(missing,loaded,{},journal.function,current)
    args=SimpleNamespace(conv_mode='vicuna_v1',temperature=0,single_pred_prompt=True,max_new_tokens=1024,
        num_chunks=1,chunk_idx=0,model_base=None,model_path=str(Path(cfg['output'])/'TRAINED-Stage1-PEFT-native-c121'),
        image_folder=str(store.virtual),question_file=str(Path(cfg['output'])/'scienceqa-native-source-view.json'),
        answers_file=str(journal.directory/'native-answers.jsonl'),answer_prompter=False)
    previous=science.expected;science.expected=lambda row,value,images:expected_science(row,value,store)
    try:
        with store.installed(),recorder.installed():
            return kernel.evaluate('scienceqa',args,rows,journal,loaded,deadline,
                {kernel.MODULES['scienceqa']:loader.official.c.desc(loader.official.c.LLAVA/'llava/eval/model_vqa_science.py')},
                recorder.take,stop_requested=lambda:False)
    finally:science.expected=previous
