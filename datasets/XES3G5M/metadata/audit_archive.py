from pathlib import Path
import csv, sys, io, json, gzip, tarfile, hashlib, time, codecs
from collections import Counter
csv.field_size_limit(sys.maxsize)
root=Path('datasets/XES3G5M'); archive=root/'raw/XES3G5M.tar.gz'
stats={'archive':archive.name,'published_preprocessed_data':True,'files':{},'metadata':{}}
members=[]; user_sets={}; item_sets={}; concept_sets={}; metadata_id_sets={}
targets={'XES3G5M/question_level/train_valid_sequences_quelevel.csv','XES3G5M/question_level/test_quelevel.csv'}
start=time.monotonic()
with gzip.open(archive,'rb') as gz:
    with tarfile.open(fileobj=gz,mode='r|') as tf:
        for m in tf:
            members.append({'name':m.name,'bytes':m.size,'type':'file' if m.isfile() else 'directory' if m.isdir() else str(m.type)})
            if m.name in targets:
                print('Auditing',m.name,flush=True)
                f=codecs.getreader('utf-8-sig')(tf.extractfile(m))
                reader=csv.DictReader(f)
                nrows=0; total=0; invalid=0; selected=0; multi=0; missing_concepts=0; correct=Counter(); users=set(); items=set(); concepts=set(); folds=Counter(); sizes=Counter(); timestamp_min=None; timestamp_max=None
                time_inversions=0;time_ties=0;early_2010=0;early_2018=0;early_users=set();tie_users=set()
                for row in reader:
                    nrows+=1; users.add(row['uid']); folds[row['fold']]+=1
                    q=row['questions'].split(','); c=row['concepts'].split(','); r=row['responses'].split(','); ts=row['timestamps'].split(',')
                    mask=row.get('selectmasks'); masks=mask.split(',') if mask else None
                    if not (len(q)==len(c)==len(r)==len(ts) and (masks is None or len(masks)==len(q))):raise RuntimeError('Inconsistent sequence lengths')
                    n=0;prev_time=None
                    for i,(qi,ci,ri,ti) in enumerate(zip(q,c,r,ts)):
                        if qi=='-1':continue
                        if ri not in ('0','1'):invalid+=1;continue
                        n+=1;total+=1;correct[ri]+=1;items.add(qi);tags=[tag for tag in ci.split('_') if tag and tag!='-1'];concepts.update(tags);multi+=len(tags)>1;missing_concepts+=not tags
                        selected+=masks is None or masks[i]=='1'
                        t=int(ti);timestamp_min=t if timestamp_min is None else min(timestamp_min,t);timestamp_max=t if timestamp_max is None else max(timestamp_max,t)
                        if t<1262304000000:early_2010+=1
                        if t<1514764800000:early_2018+=1;early_users.add(row['uid'])
                        if prev_time is not None:
                            time_inversions+=t<prev_time
                            if t==prev_time:time_ties+=1;tie_users.add(row['uid'])
                        prev_time=t
                    sizes[n]+=1
                f.close()
                stats['files'][m.name]={'rows':nrows,'num_users':len(users),'num_questions':len(items),'num_leaf_concepts':len(concepts),'valid_nonpadding_response_tokens':total,'correct_counts':dict(correct),'correct_rate':correct['1']/total,'invalid_nonpadding_responses':invalid,'missing_concepts_for_valid_question_tokens':missing_concepts,'selected_by_published_mask_tokens':selected if 'selectmasks' in reader.fieldnames else None,'multi_concept_tokens':multi,'multi_concept_token_fraction':multi/total,'fold_row_counts':dict(folds),'sequence_length_distribution':dict(sorted(sizes.items())),'timestamp_min_ms':timestamp_min,'timestamp_max_ms':timestamp_max,'columns':reader.fieldnames}
                user_sets[m.name]=users;item_sets[m.name]=items;concept_sets[m.name]=concepts
                stats['files'][m.name]['timestamp_audit']={'adjacent_inversions_within_published_row':time_inversions,'adjacent_ties_within_published_row':time_ties,'users_with_adjacent_ties':len(tie_users),'tokens_before_2010_utc':early_2010,'tokens_before_2018_utc':early_2018,'users_with_tokens_before_2018_utc':len(early_users),'note':'Ties and inversions are checked within supplied rows, not across training chunks. Preserve the author sequence order; do not break ties by question ID.'}
            elif m.name in {'XES3G5M/metadata/questions.json','XES3G5M/metadata/kc_routes_map.json'}:
                data=json.load(tf.extractfile(m));stats['metadata'][m.name]={'entries':len(data),'top_level_type':type(data).__name__};metadata_id_sets[m.name]=set(map(str,data)) if isinstance(data,dict) else set()
                if 'questions.json' in m.name:
                    stats['metadata'][m.name]['question_type_counts']=dict(Counter(str(x.get('type')) for x in data.values()))
                    stats['metadata'][m.name]['question_fields']=sorted(next(iter(data.values())).keys())
            if len(members)%2000==0:print('Members checked',len(members),flush=True)
    # Consume the gzip trailer too: tar EOF alone is insufficient for a gzip CRC check.
    while gz.read(8*1024*1024):pass
stats['gzip_crc_integrity']='passed';stats['tar_members']=len(members);stats['uncompressed_member_bytes']=sum(m['bytes'] for m in members if m['type']=='file')
keys=list(user_sets)
if len(keys)!=2:raise RuntimeError(f'Expected two question-level files, got {keys}')
stats['question_level_totals']={'unique_users':len(set.union(*user_sets.values())),'train_test_user_overlap':len(set.intersection(*user_sets.values())),'unique_questions':len(set.union(*item_sets.values())),'unique_leaf_concepts':len(set.union(*concept_sets.values())),'valid_nonpadding_response_tokens':sum(x['valid_nonpadding_response_tokens'] for x in stats['files'].values()),'note':'Counts describe supplied question-level sequences after author truncation/padding. They do not reconstruct raw 5,549,635 original interactions; training and test files follow different truncation formats.'}
stats['question_level_totals']['correct_rate']=sum(x['correct_counts'].get('1',0) for x in stats['files'].values())/stats['question_level_totals']['valid_nonpadding_response_tokens']
stats['metadata_join_coverage']={'missing_question_ids':sorted(set.union(*item_sets.values())-metadata_id_sets['XES3G5M/metadata/questions.json']),'missing_leaf_concept_ids':sorted(set.union(*concept_sets.values())-metadata_id_sets['XES3G5M/metadata/kc_routes_map.json'])}
(root/'metadata/stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
(root/'metadata/archive_members.json').write_text(json.dumps(members,ensure_ascii=False,indent=2)+'\n')
manifest=json.loads((root/'metadata/download_manifest.json').read_text());manifest['gzip_crc_integrity']='passed';manifest['tar_members']=len(members);manifest['uncompressed_member_bytes']=stats['uncompressed_member_bytes'];(root/'metadata/download_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'totals':stats['question_level_totals'],'archive_members':len(members),'uncompressed_bytes':stats['uncompressed_member_bytes'],'metadata':stats['metadata'],'seconds':round(time.monotonic()-start,1)},ensure_ascii=False,indent=2),flush=True)
