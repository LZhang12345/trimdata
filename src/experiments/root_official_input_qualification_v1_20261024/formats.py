"""Input-only adapters; unchanged source bytes, no answers/prediction outputs."""
import base64,csv,hashlib,io,json
from collections import Counter
from pathlib import PurePosixPath
from experiments.root_official_protected_inputs_v1_20261024 import parsers as old

METRICS=('VQAv2','GQA','VizWiz','ScienceQA_image','TextVQA','POPE','MME_perception','MME_cognition',
 'MMBench_EN','MMBench_CN','SEED_image','LLaVA_Bench_in_the_Wild','MM_Vet','MMMU_validation','NoCaps_validation')
MEMBERS=('MME/llava_mme.jsonl','pope/llava_pope_test.jsonl','scienceqa/llava_test_CQM-A.json',
 'seed_bench/llava-seed-bench.jsonl','textvqa/llava_textvqa_val_v051_ocr.jsonl',
 'vizwiz/llava_test.jsonl','vqav2/llava_vqav2_mscoco_test-dev2015.jsonl','mm-vet/llava-mm-vet.jsonl')
MMV='mm-vet/llava-mm-vet.jsonl'

def require(v,msg):
 if not v:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def key(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode())
def safe(name):
 p=PurePosixPath(name);require(not p.is_absolute() and '..' not in p.parts and '\\' not in name,'Unsafe member/reference');return name

def census_json(raw,array=False):
 rows=old.json_load(raw) if array else [old.json_load(line) for line in raw.splitlines()]
 require(isinstance(rows,list) and rows and all(isinstance(r,dict) for r in rows),'Expected nonempty source records')
 keys=Counter(tuple(sorted(r)) for r in rows)
 id_field='id' if array else 'question_id'
 ids=[str(r[id_field]) for r in rows]
 # Values/text/answers are intentionally absent from this raw-format report.
 return rows,dict(records=len(rows),unique_IDs=len(set(ids)),duplicate_IDs=len(ids)-len(set(ids)),
  schema_counts=[dict(keys=list(k),records=v) for k,v in sorted(keys.items())],
  records_with_image=sum('image' in r for r in rows),ordered_IDs_sha256=key(ids),
  raw_question_or_answer_emitted=False)

def census_tsv(raw):
 old_limit=csv.field_size_limit();baseline=None
 try:
  csv.field_size_limit(131072)
  try:old.mmbench_tsv(raw,{'qa_access':'authorized'});baseline='PASS_OLD_STRICT_INLINE_PARSER'
  except Exception as e:baseline=type(e).__name__
  csv.field_size_limit(16<<20)
  reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')),delimiter='\t')
  require(reader.fieldnames and len(reader.fieldnames)==len(set(reader.fieldnames)) and
   {'index','image','question','hint','A','B','C','D'}<=set(reader.fieldnames),'MMBench exact columns absent')
  rows=list(reader);require(rows and all(None not in r and None not in r.values() for r in rows),'Malformed TSV')
 finally:csv.field_size_limit(old_limit)
 ids=[old.identity(r['index']) for r in rows]
 short=[r['index'] for r in rows if 0<len(r['image'])<=64]
 return rows,dict(records=len(rows),unique_IDs=len(set(ids)),columns=reader.fieldnames,
  old_strict_parser_outcome=baseline,short_image_fields=len(short),short_image_IDs=short,
  maximum_encoded_field_bytes=max(len(r['image']) for r in rows),
  ordered_IDs_sha256=key(ids),CSV_field_limit_used_for_census=16<<20)

# pandas.read_table's empty/NA values become float NaN before c121 is_none.
NA={'','nan','none','NaN','NA','N/A','n/a','NULL','null','#N/A','#NA','<NA>','-NaN','-nan','1.#IND','1.#QNAN','-1.#IND','-1.#QNAN'}
def none(v):return v is None or v in NA or (isinstance(v,str) and v.lower() in ('nan','none'))
def c121_prompt(row):
 require(isinstance(row['question'],str) and row['question'].strip(),'Question field invalid')
 text=row['question']
 if not none(row['hint']):text=row['hint']+'\n'+text
 options=[]
 for name in 'ABCD':
  if none(row[name]):break # Exact c121 get_options stops at the first missing option.
  options.append(row[name])
 require(1<=len(options)<=4,'Unexpected empty multiple-choice support')
 for name,option in zip('ABCD',options):text+='\n'+name+'. '+option
 return text,len(options)

def repair_mmbench(rows,census):
 require(len(rows)==census['records'] and len(rows)==census['unique_IDs'],'Missing/duplicate TSV index')
 mapping={old.identity(r['index']):r['image'] for r in rows};records=[];blobs={};aliases={}
 for row in rows:
  ident=old.identity(row['index']);encoded=row['image'];source_id=ident
  if 0<len(encoded)<=64:
   require(census['short_image_fields']>0 and encoded in mapping,'Unresolved image alias')
   source_id=encoded;encoded=mapping[encoded]
   require(len(encoded)>64,'Cycle/chained/nonimage alias; no heuristic resolution')
   aliases[ident]=source_id
  require(len(encoded)>64,'Invalid original inline image')
  if source_id not in blobs:
   blob=base64.b64decode(encoded,validate=True);require(blob,'Empty original image');blobs[source_id]=blob
  prompt,count=c121_prompt(row)
  records.append(dict(id=ident,image='tsv-index/'+ident,question_sha256=sha(prompt.encode()),
   options=count,image_storage_key=source_id,encoded_sha256=sha(blobs[source_id]),encoded_bytes=len(blobs[source_id])))
 return records,blobs,dict(alias_resolution=aliases,source_representation_adaptation=bool(aliases),
  question_and_options_order_preserved=True,c121_single_prediction_suffix_not_executed=True)

def native_jsonl(raw):
 rows,info=census_json(raw);parsed=old.author_jsonl(raw,{'qa_access':'authorized'})
 require(len(parsed)==len(rows) and len({r['id'] for r in parsed})==len(parsed),'Duplicate/incomplete native support')
 return [dict(id=r['id'],image=r['image'],question_sha256=sha(r['question'].encode())) for r in parsed],info

def qualify_mmvet(raw,annotation,images):
 rows,info=native_jsonl(raw);meta=old.json_load(annotation)
 require(isinstance(meta,dict) and meta,'Original MM-Vet membership object absent')
 wanted={'v1_'+r['id'] for r in rows}
 require(wanted==set(meta),'Native c121 converter v1_<question_id> membership differs')
 result=[]
 for row in rows:
  upstream=meta['v1_'+row['id']]
  require(isinstance(upstream,dict) and upstream.get('imagename')==row['image'],'Native/original image reference differs')
  require(isinstance(upstream.get('question'),str) and sha(upstream['question'].encode())==row['question_sha256'],
   'Native/original question inputs differ')
  require(row['image'] in images,'Missing/ambiguous original MM-Vet encoded image')
  result.append(dict(row,image_storage_key=images[row['image']]))
 return result,info
