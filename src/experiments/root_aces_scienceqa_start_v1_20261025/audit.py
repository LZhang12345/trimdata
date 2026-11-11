from contextlib import contextmanager
import hashlib
from pathlib import Path
from . import common as c
from experiments.session6_native_start_execution_v6_20261024.audit import tensor_digest
SUFFIX="\nAnswer with the option's letter from the given choices directly."

def prompt_text(row):
 value=row['native_input']['conversations'][0];c.require(value['from']=='human' and len(row['native_input']['conversations'])==1,'Answer-free firsthuman only')
 return '<image>\n'+value['value'].replace('<image>','').strip()+SUFFIX

def expected(row,loaded,images):
 import torch
 from PIL import Image
 from llava.conversation import conv_templates
 from llava.constants import IMAGE_TOKEN_INDEX
 from llava.mm_utils import process_images,tokenizer_image_token
 alias=images['images'][row['image_reference']];ref=alias['local'];path=Path(ref['path']);root=Path(images['image_root'])
 c.require(path==root/row['native_input']['image'] and '..' not in path.parts and path.is_relative_to(root) and not any(p.is_symlink() for p in (path,*path.parents)),'Exact approved native image locator')
 c.require(c.desc(ref['path'])==ref,'Live encoded ScienceQA image changed')
 tokenizer,model,processor,_=loaded;conv=conv_templates['vicuna_v1'].copy();conv.append_message(conv.roles[0],prompt_text(row));conv.append_message(conv.roles[1],None);prompt=conv.get_prompt()
 tokens=tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0)
 with Image.open(ref['path']) as im:
  rgb=im.convert('RGB')
  try:c.require(hashlib.sha256(rgb.tobytes()).hexdigest()==alias['rgb_sha256'] and list(rgb.size)==[alias['width'],alias['height']],'Live dimensionedRGB changed')
  finally:rgb.close()
  pixels=process_images([im],processor,model.config)[0].unsqueeze(0).to(torch.float16)
  return dict(prompt_utf8_sha256=hashlib.sha256(prompt.encode()).hexdigest(),input_ids_sha256=tensor_digest(tokens),input_token_count=tokens.shape[-1],pixel_values_sha256=tensor_digest(pixels),image_size=list(im.size),raw_prompt=prompt_text(row))

class Recorder:
 def __init__(self,rows,loaded,images,fn,actual):self.rows=rows;self.loaded=loaded;self.images=images;self.fn=fn;self.actual=actual;self.position=0;self.pending=None
 def take(self):
  c.require(self.pending is not None,'No real ScienceQA generation tocommit');v=self.pending;self.pending=None;self.position+=1;return v
 @contextmanager
 def installed(self):
  model=self.loaded[1];original=model.generate
  def generate(*args,**kw):
   c.require(self.pending is None and self.position<len(self.rows) and len(args)==1,'Native ordered generation event')
   c.require(set(kw)=={'images','image_sizes','do_sample','temperature','max_new_tokens','use_cache'} and kw['do_sample'] is False and kw['temperature']==0 and kw['max_new_tokens']==1024 and kw['use_cache'] is True,'Exact scientific native kwargs; beams/top_p deliberately inherited/absent')
   c.require(model.generation_config.num_beams==1,'Actual inheritedgreedy beam')
   row=self.rows[self.position];proof=expected(row,self.loaded,self.images)
   c.require(tensor_digest(args[0])==proof['input_ids_sha256'] and tensor_digest(kw['images'])==proof['pixel_values_sha256'] and [list(x) for x in kw['image_sizes']]==[proof['image_size']],'Actual prompt/token/processedpixel mismatch')
   out=original(*args,**kw);ids=out.detach().cpu().tolist();eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
   c.require(len(ids)==1 and ids[0] and (ids[0][-1] in eos or len(ids[0])>=1024),'Actual native EOS/lengthtermination')
   self.pending=dict(function_sha256=c.key(self.fn),actual=self.actual,native_generation=dict(semantic_key=row['semantic_key'],**proof,output_ids=ids[0],decoded=self.loaded[0].batch_decode(out,skip_special_tokens=True)[0].strip(),explicit_kwargs={k:v for k,v in kw.items() if k not in ('images','image_sizes')},image_reference=row['image_reference']))
   return out
  model.generate=generate
  try:yield;c.require(self.pending is None,'Generated answer was not durable')
  finally:model.generate=original

def validate_entry(entry,row,fn):
 raw=entry['raw'];audit=entry['audit'];g=audit['native_generation']
 c.require(entry['semantic_key']==g['semantic_key']==row['semantic_key'] and audit['function_sha256']==c.key(fn) and type(raw['question_id']) is type(row['native_id']) and raw['question_id']==row['native_id'] and raw['prompt']==prompt_text(row) and raw['text']==g['decoded'] and g['image_reference']==row['image_reference'],'Native raw/audit/typedQA/prompt differs')
 c.require(audit['generation_config_binding']==dict(model_generation_config_sha256=c.key(fn['model_generation_config']),native_num_beams_keyword_present=False,effective_num_beams=1),'Native inherited generationconfig differs')
