"""Exact ScienceQA public test input support, skipping all answer/solution fields."""
import io,json,hashlib
from pathlib import PurePosixPath
from experiments.root_official_protected_inputs_v1_20261024.roles_projection_v2 import Reader
from experiments.root_official_input_qualification_v1_20261024 import formats as f
from experiments.root_official_protected_inputs_v1_20261024 import parsers as p
require=f.require

def scalar(r):
 r.ws();return r.string(True) if r.peek()==34 else r.atom(True)
def array(r,callback):
 r.expect('[');r.ws()
 if r.peek()==93:r.take();return
 i=0
 while True:
  callback(i);i+=1;r.ws();token=r.take()
  if token==93:return
  require(token==44,'Malformed authorized input array')
def end(r):r.ws();require(r.peek() is None,'Trailing original input JSON')

def splits(raw):
 r=Reader(io.BytesIO(raw),len(raw));result={}
 def field(k):
  if k=='test':
   value=[];array(r,lambda _:value.append(p.identity(scalar(r))));result[k]=value
  else:r.skip_value()
 r.object(field);end(r);ids=result.get('test',[])
 require(ids and len(ids)==len(set(ids)),'Exact unique official ScienceQA test PID membership required')
 return ids

def problems(raw,testids):
 r=Reader(io.BytesIO(raw),len(raw));wanted=set(testids);result={}
 def problem(pid):
  if pid not in wanted:r.skip_value();return
  v={}
  def field(k):
   if k in ('image','question','hint','split'):v[k]=scalar(r)
   elif k=='choices':
    choices=[];array(r,lambda _:choices.append(scalar(r)));v[k]=choices
   else:r.skip_value()
  r.object(field);result[pid]=v
 r.object(problem);end(r)
 require(set(result)==wanted and all(v.get('split')=='test' for v in result.values()),'Every declared official test PID and original split required')
 return result

def native(raw):
 r=Reader(io.BytesIO(raw),len(raw));result=[]
 def row(i):
  v={}
  def field(k):
   if k in ('id','image'):v[k]=scalar(r)
   elif k=='conversations':
    def turn(n):
     if n:r.skip_value();return
     q={}
     def first(k):
      if k in ('from','value'):q[k]=scalar(r)
      else:r.skip_value()
     r.object(first);require(q.get('from')=='human' and isinstance(q.get('value'),str),'Exact c121 first human question required');v['human']=q['value']
    array(r,turn)
   else:r.skip_value()
  r.object(field);require('human' in v and 'id' in v,'Native CQM-A input identity missing')
  v['id']=p.identity(v['id']);v['source_row']=i;result.append(v)
 array(r,row);end(r)
 require(result and len({x['id'] for x in result})==len(result),'Native CQM-A test PIDs must be unique')
 return result

def graph(native_raw,split_raw,problem_raw,index):
 testids=splits(split_raw);original=problems(problem_raw,testids);allrows=native(native_raw)
 require({x['id'] for x in allrows}==set(testids),'Full native CQM-A original test membership differs')
 expected={pid:v for pid,v in original.items() if v.get('image') is not None}
 rows=[];mapping={};original_hashes={}
 for value in allrows:
  pid=value['id'];official=original[pid];has_image='image' in value
  require(has_image==(pid in expected),'Native/official image-only membership differs')
  if not has_image:continue
  filename=official['image'];require(isinstance(filename,str) and PurePosixPath(filename).name==filename and filename not in ('','.','..'),'Literal official image filename required')
  canonical=pid+'/'+filename
  require(value['image']==canonical,'Native c121 test image reference differs from exact original PID/filename')
  # Pinned sqa.sh supplies images/test as the loader root; original archive includes test/.
  candidates=[x for x in ('test/'+canonical,) if x in index and not index[x]['directory']]
  require(len(candidates)==1,'Original ScienceQA test archive member absent or ambiguous')
  prompt=value['human'].replace('<image>','').strip() # exact c121 loader line44
  require(prompt,'Native CQM-A question input empty')
  rows.append(dict(id=pid,image=value['image'],question_sha256=hashlib.sha256(prompt.encode()).hexdigest(),original_row_index=value['source_row']))
  mapping[value['image']]=candidates[0]
  original_hashes[pid]=f.key({k:official.get(k) for k in ('question','hint','choices','image','split')})
 require({x['id'] for x in rows}==set(expected) and rows,'Complete official image-only test support required')
 return rows,mapping,dict(official_test_count=len(testids),native_all_test_count=len(allrows),image_only_count=len(rows),original_public_input_hashes=original_hashes,native_prompt_rule='c121_first_human_value.replace_image_token_then_strip',source_image_root='c121_sqa.sh:images/test',literal_native_image_rule='<PID>/<official_image_filename>',literal_archive_member_rule='test/<PID>/<official_image_filename>',answers_solutions_or_assistant_values_decoded=False,no_original_native_prompt_format_equivalence_claimed=True)
