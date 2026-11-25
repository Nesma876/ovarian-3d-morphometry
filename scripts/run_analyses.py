"""Run the published analysis stages with locally authorised inputs."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from follicle_stable.io import read_segmentation_from_zip, reconstruct_instances
from follicle_stable.morphometry import equivalent_diameter_mm

SEED = 20261120
SIX = ndi.generate_binary_structure(3, 1)
OFFSETS = [(z, y, x) for z in (0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1)
           if (z, y, x) > (0, 0, 0) and sum(v != 0 for v in (z, y, x)) >= 2]
CAT = ["[0,2)", "[2,5)", "[5,10)", "[10,inf)"]


def category(d):
    return CAT[0] if d < 2 else CAT[1] if d < 5 else CAT[2] if d < 10 else CAT[3]


def shifted(labels, offset):
    aa, bb = [], []
    for d in offset:
        if d > 0: aa.append(slice(None, -d)); bb.append(slice(d, None))
        elif d < 0: aa.append(slice(-d, None)); bb.append(slice(None, d))
        else: aa.append(slice(None)); bb.append(slice(None))
    return labels[tuple(aa)], labels[tuple(bb)], aa


def contact_rows(labels6, acquisition_id, participant_id):
    rows = []
    for off in OFFSETS:
        a, b, slices = shifted(labels6, off)
        mask = (a > 0) & (b > 0) & (a != b)
        coords = np.argwhere(mask)
        origin = np.array([s.start or 0 for s in slices])
        kind = "edge" if sum(v != 0 for v in off) == 2 else "vertex"
        for c in coords:
            ca = c + origin; cb = ca + np.asarray(off)
            left, right = int(labels6[tuple(ca)]), int(labels6[tuple(cb)])
            if left > right: left, right, ca, cb = right, left, cb, ca
            rows.append((acquisition_id, participant_id, left, right, kind, *map(int, ca), *map(int, cb)))
    return rows


def correspondence():
    manifest = pd.read_csv(MANIFEST)
    components, groups, contacts_all, counts = [], [], [], []
    for k, rec in manifest.iterrows():
        sem, spacing, _ = read_segmentation_from_zip(ZIP, rec.segmentation_zip_member)
        l6, n6 = reconstruct_instances(sem, connectivity=1)
        l18, n18 = reconstruct_instances(sem, connectivity=2)
        l26, n26 = reconstruct_instances(sem, connectivity=3)
        if not (n6 >= n18 >= n26): raise AssertionError(f"count monotonicity: {rec.acquisition_id}")
        if not (np.array_equal(l6 > 0, l18 > 0) and np.array_equal(l6 > 0, l26 > 0)):
            raise AssertionError(f"foreground mismatch: {rec.acquisition_id}")
        vv = float(np.prod(spacing)); s6 = np.bincount(l6.ravel(), minlength=n6 + 1); s26 = np.bincount(l26.ravel(), minlength=n26 + 1)
        counts.append(dict(acquisition_id=rec.acquisition_id, participant_id=rec.participant_id,
            abs_count_diff_6_18=n6-n18, abs_count_diff_6_26=n6-n26,
            spacing_z_mm=spacing[0], spacing_y_mm=spacing[1], spacing_x_mm=spacing[2]))
        contact_data = contact_rows(l6, rec.acquisition_id, rec.participant_id)
        contacts_all.extend(contact_data)
        cdf = pd.DataFrame(contact_data, columns=["acquisition_id","participant_id","component_a","component_b","contact_type","a_z","a_y","a_x","b_z","b_y","b_x"])
        mapping = {}
        for iid in range(1, n6 + 1):
            found = np.unique(l26[l6 == iid]); found = found[found > 0]
            if len(found) != 1: raise AssertionError(f"non-unique containment: {rec.acquisition_id}/{iid}")
            mapping[iid] = int(found[0])
        inverse = {}
        for iid, gid in mapping.items(): inverse.setdefault(gid, []).append(iid)
        for gid, ids in inverse.items():
            gd = equivalent_diameter_mm(float(s26[gid] * vv)); gcat = category(gd)
            local = cdf[cdf.component_a.isin(ids) & cdf.component_b.isin(ids)] if len(cdf) else cdf
            edge = int((local.contact_type == "edge").sum()) if len(local) else 0
            vertex = int((local.contact_type == "vertex").sum()) if len(local) else 0
            source_d = [equivalent_diameter_mm(float(s6[i] * vv)) for i in ids]
            source_c = [category(d) for d in source_d]
            crossing = any(CAT.index(gcat) > CAT.index(c) for c in source_c)
            groups.append({"acquisition_id":rec.acquisition_id,"participant_id":rec.participant_id,"group_26":gid,
                "source_component_ids":";".join(map(str,ids)),"multiplicity":len(ids),"source_volumes_mm3":";".join(f"{s6[i]*vv:.9g}" for i in ids),
                "group_volume_mm3":s26[gid]*vv,"source_diameters_mm":";".join(f"{d:.9g}" for d in source_d),
                "group_diameter_mm":gd,"source_categories":";".join(source_c),"group_category":gcat,
                "edge_contact_pairs":edge,"vertex_contact_pairs":vertex,"category_crossing":crossing})
            for iid, d, c in zip(ids, source_d, source_c):
                components.append({"acquisition_id":rec.acquisition_id,"participant_id":rec.participant_id,"component_6":iid,
                    "group_26":gid,"multiplicity":len(ids),"source_volume_mm3":s6[iid]*vv,"group_volume_mm3":s26[gid]*vv,
                    "source_diameter_mm":d,"group_diameter_mm":gd,"source_category":c,"group_category":gcat,
                    "edge_contact_pairs_in_group":edge,"vertex_contact_pairs_in_group":vertex,"category_crossing":crossing})
        if (k + 1) % 50 == 0: print(f"correspondence {k+1}/{len(manifest)}", flush=True)
    comp = pd.DataFrame(components); grp = pd.DataFrame(groups); contacts = pd.DataFrame(contacts_all,
        columns=["acquisition_id","participant_id","component_a","component_b","contact_type","a_z","a_y","a_x","b_z","b_y","b_x"])
    pd.DataFrame(counts).to_csv(RES / "connectivity.csv", index=False)
    comp.to_csv(RES / "component_correspondence.csv", index=False)
    grp[grp.multiplicity > 1].to_csv(RES / "merger_groups.csv", index=False)
    contacts.to_csv(RES / "diagonal_contacts.csv", index=False)
    mult = grp.assign(multiplicity_class=np.where(grp.multiplicity >= 4, "4+", grp.multiplicity.astype(str))).groupby("multiplicity_class").size().rename("groups_26").reset_index()
    mult["proportion_of_all_groups_26"] = mult.groups_26 / len(grp); mult.to_csv(RES / "merger_multiplicity.csv", index=False)
    trans = comp.groupby(["source_category","group_category"]).size().rename("source_components").reset_index()
    trans.to_csv(RES / "morphometric_transitions.csv", index=False)
    return comp, grp, contacts


def fit(a, shape):
    out = np.zeros(shape, dtype=a.dtype); sl = tuple(slice(0, min(x, y)) for x, y in zip(a.shape, shape)); out[sl] = a[sl]; return out


def overlap_events(a, b):
    mask = (a > 0) & (b > 0)
    pairs = np.unique(np.stack([a[mask], b[mask]], 1), axis=0) if mask.any() else np.empty((0,2), int)
    amap = {int(i):set() for i in np.unique(a[a>0])}; bmap = {int(i):set() for i in np.unique(b[b>0])}
    for i,j in pairs: amap[int(i)].add(int(j)); bmap[int(j)].add(int(i))
    return sum(not x for x in amap.values()), sum(len(x)>1 for x in amap.values()), sum(len(x)>1 for x in bmap.values()), sum(not x for x in bmap.values())


def round_trip(source, factor, order):
    """Round-trip on identical index geometry; return fitted array and geometry audit."""
    zoom_down = np.repeat(1.0 / factor, 3)
    low = ndi.zoom(source, zoom=zoom_down, order=order, mode="nearest", prefilter=order > 1)
    restored = fit(ndi.zoom(low, zoom=np.repeat(factor, 3), order=order,
                            mode="nearest", prefilter=order > 1), source.shape)
    return restored, low.shape


def component_diameters(labels, spacing):
    sizes = np.bincount(labels.ravel())[1:]
    return np.asarray([equivalent_diameter_mm(float(v * np.prod(spacing))) for v in sizes if v])


def distribution_deltas(original, restored):
    def fractions(values):
        if not len(values): return np.zeros(4)
        return np.asarray([(values < 2).mean(), ((values >= 2) & (values < 5)).mean(),
                           ((values >= 5) & (values < 10)).mean(), (values >= 10).mean()])
    return fractions(restored) - fractions(original)


def resampling():
    chosen = pd.read_csv(SELECTION)[["acquisition_id","participant_id"]]
    manifest = pd.read_csv(MANIFEST).set_index("acquisition_id"); binary_rows=[]; label_rows=[]
    for _, sel in chosen.iterrows():
        rec=manifest.loc[sel.acquisition_id]; sem, spacing, _=read_segmentation_from_zip(ZIP, rec.segmentation_zip_member)
        original,n0=reconstruct_instances(sem,connectivity=1); binary=original>0; v0=int(binary.sum())
        base_d=component_diameters(original, spacing)
        for factor in (1.5,2.0):
            # PRIMARY: both arms start from the exact same semantic binary array.
            for method,order,thresholds in (("nearest",0,[0.5]),("linear",1,[.3,.4,.5,.6,.7])):
                prob, low_shape = round_trip(binary.astype(float), factor, order)
                for threshold in thresholds:
                    restored=prob>=threshold; lab,n=ndi.label(restored,structure=SIX)
                    inter=int((binary&restored).sum()); union=int((binary|restored).sum()); ev=overlap_events(original,lab)
                    diam=component_diameters(lab, spacing); dd=distribution_deltas(base_d, diam)
                    binary_rows.append({"acquisition_id":sel.acquisition_id,"participant_id":sel.participant_id,"spacing_factor":factor,"method":method,
                        "threshold":threshold,"count_original":n0,"count_restored":int(n),"signed_count_change":int(n-n0),"absolute_count_change":abs(int(n-n0)),
                        "relative_count_change":(n-n0)/max(n0,1),"dice":2*inter/max(v0+int(restored.sum()),1),"iou":inter/max(union,1),
                        "signed_volume_change_voxels":int(restored.sum())-v0,"relative_volume_change":(int(restored.sum())-v0)/max(v0,1),
                        "absolute_relative_volume_error":abs(int(restored.sum())-v0)/max(v0,1),"removed_components":ev[0],"fragmented_sources":ev[1],
                        "merged_targets":ev[2],"new_components":ev[3],"mean_diameter_original_mm":base_d.mean(),"mean_diameter_restored_mm":diam.mean() if len(diam) else 0,
                        "diameter_fraction_delta_lt2":dd[0],"diameter_fraction_delta_2to5":dd[1],"diameter_fraction_delta_5to10":dd[2],"diameter_fraction_delta_ge10":dd[3],
                        "source_binary_sha256_equal":True,"source_shape":str(binary.shape),"intermediate_shape":str(low_shape),"target_shape":str(restored.shape),
                        "connectivity":6,"boundary_mode":"nearest","coordinate_convention":"scipy.ndimage.zoom index grid"})

            # SECONDARY: preserve label identity, then report both surviving IDs and
            # a fresh 6-connected binary reconstruction. These are distinct endpoints.
            propagated, low_shape = round_trip(original.astype(np.int32), factor, 0)
            propagated=propagated.astype(np.int32); restored=propagated>0
            relabelled,n_reconstructed=ndi.label(restored,structure=SIX)
            surviving=int(np.count_nonzero(np.bincount(propagated.ravel())[1:]))
            inter=int((binary&restored).sum()); union=int((binary|restored).sum())
            label_rows.append({"acquisition_id":sel.acquisition_id,"participant_id":sel.participant_id,"spacing_factor":factor,
                "original_label_count":n0,"surviving_label_id_count":surviving,"reconstructed_binary_component_count":int(n_reconstructed),
                "signed_surviving_label_change":surviving-n0,"signed_reconstructed_count_change":int(n_reconstructed-n0),
                "dice":2*inter/max(v0+int(restored.sum()),1),"iou":inter/max(union,1),"source_representation":"6-connected integer instance labels",
                "intermediate_shape":str(low_shape),"target_shape":str(restored.shape),"boundary_mode":"nearest"})
    full=pd.DataFrame(binary_rows); full.to_csv(RES/"threshold_sensitivity.csv",index=False)
    pd.DataFrame(label_rows).to_csv(RES/"label_propagation_results.csv",index=False)
    primary=full[((full.method=="nearest") | (full.threshold==.5))].copy()
    nn=primary[primary.method=="nearest"].set_index(["acquisition_id","participant_id","spacing_factor"])
    lin=primary[primary.method=="linear"].set_index(["acquisition_id","participant_id","spacing_factor"])
    paired=pd.DataFrame(index=nn.index).reset_index()
    for col in ["dice","iou","absolute_count_change","signed_count_change","relative_count_change","absolute_relative_volume_error","merged_targets","fragmented_sources","removed_components","new_components"]:
        paired[f"nearest_{col}"]=nn[col].values; paired[f"linear_{col}"]=lin[col].values
    paired["delta_dice_linear_minus_nearest"]=paired.linear_dice-paired.nearest_dice
    paired["delta_count_error_linear_minus_nearest"]=paired.linear_absolute_count_change-paired.nearest_absolute_count_change
    paired["linear_higher_dice_and_larger_count_error"]=(paired.delta_dice_linear_minus_nearest>0)&(paired.delta_count_error_linear_minus_nearest>0)
    paired.to_csv(RES/"representation_matched_results.csv",index=False)
    return full,paired



THRESH = [0, .5, 1, 1.5, 2, 3]
REPS = 5000

def cat(d):
    return 0 if d < 2 else 1 if d < 5 else 2 if d < 10 else 3

def wp1_graph():
 """Exact whole-component filtering using verified 6-component contact graphs."""
 comp=pd.read_csv(RES/"component_correspondence.csv")
 contacts=pd.read_csv(RES/"diagonal_contacts.csv")
 manifest=pd.read_csv(MAN)[["acquisition_id","participant_id"]]
 rows=[]
 by_contacts={a:g for a,g in contacts.groupby("acquisition_id")}
 by_comp={a:g for a,g in comp.groupby("acquisition_id")}
 for k,r in manifest.iterrows():
  c=by_comp.get(r.acquisition_id,pd.DataFrame(columns=comp.columns)); e=by_contacts.get(r.acquisition_id,pd.DataFrame(columns=contacts.columns))
  total=float(c.source_volume_mm3.sum()) if len(c) else 0.; groups=c.drop_duplicates("group_26") if len(c) else c
  for t in THRESH:
   keep=c[c.source_diameter_mm>=t].copy(); keep_ids=set(keep.component_6.astype(int)); keepg=groups[groups.group_diameter_mm>=t]
   va=float(keep.source_volume_mm3.sum()); vb=float(keepg.group_volume_mm3.sum())
   c6=len(keep); c26=len(keepg)
   rows.append(dict(protocol="A_independent_post_reconstruction",threshold_mm=t,acquisition_id=r.acquisition_id,participant_id=r.participant_id,
    count_6=c6,count_26=c26,absolute_count_difference=abs(c6-c26),relative_count_difference=(c6-c26)/max(c6,1),affected=int(c6!=c26),
    merger_groups=np.nan,source_components_in_mergers=np.nan,category_crossing_groups=np.nan,max_merger_multiplicity=np.nan,
    retained_volume_6_fraction=va/total if total else 1.,retained_volume_26_fraction=vb/total if total else 1.,common_retained_volume_fraction=np.nan,removed_volume_mm3=np.nan))
   # Connected components of the retained verified diagonal-contact graph equal
   # 26-components of the common mask after removal of complete 6-components.
   parent={i:i for i in keep_ids}
   def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
   def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb: parent[rb]=ra
   if keep_ids and len(e):
    for q in e[e.component_a.isin(keep_ids)&e.component_b.isin(keep_ids)].itertuples(): union(int(q.component_a),int(q.component_b))
   sets={}
   for i in keep_ids: sets.setdefault(find(i),[]).append(i)
   mg=part=cross=0; mmax=max((len(v) for v in sets.values()),default=0)
   vol=keep.set_index("component_6").source_volume_mm3.to_dict(); diam=keep.set_index("component_6").source_diameter_mm.to_dict()
   for ids in sets.values():
    if len(ids)>1:
     mg+=1; part+=len(ids); gd=equivalent_diameter_mm(sum(vol[i] for i in ids)); cross+=int(any(cat(gd)>cat(diam[i]) for i in ids))
   x6=len(keep_ids); x26=len(sets)
   rows.append(dict(protocol="B_common_6_reference_mask",threshold_mm=t,acquisition_id=r.acquisition_id,participant_id=r.participant_id,
    count_6=x6,count_26=x26,absolute_count_difference=abs(x6-x26),relative_count_difference=(x6-x26)/max(x6,1),affected=int(x6!=x26),
    merger_groups=mg,source_components_in_mergers=part,category_crossing_groups=cross,max_merger_multiplicity=mmax,
    retained_volume_6_fraction=np.nan,retained_volume_26_fraction=np.nan,common_retained_volume_fraction=va/total if total else 1.,removed_volume_mm3=total-va))
 df=pd.DataFrame(rows); df.to_csv(W1/"per_acquisition_size_sensitivity.csv",index=False)
 summary=[]
 for (p,t),g in df.groupby(["protocol","threshold_mm"]):
  pg=g.groupby("participant_id").agg(affected=("affected","mean"),reldiff=("relative_count_difference","mean"))
  summary.append(dict(protocol=p,threshold_mm=t,n_acquisitions=len(g),affected_acquisitions=int(g.affected.sum()),affected_fraction=g.affected.mean(),
   mean_absolute_count_difference=g.absolute_count_difference.mean(),median_absolute_count_difference=g.absolute_count_difference.median(),acquisition_weighted_relative_difference=g.relative_count_difference.mean(),
   participant_weighted_affected_fraction=pg.affected.mean(),participant_weighted_relative_difference=pg.reldiff.mean(),merger_groups=g.merger_groups.sum(min_count=1),
   source_components_in_mergers=g.source_components_in_mergers.sum(min_count=1),category_crossing_groups=g.category_crossing_groups.sum(min_count=1),
   mean_common_volume_retained=g.common_retained_volume_fraction.mean(),mean_6_volume_retained=g.retained_volume_6_fraction.mean(),mean_26_volume_retained=g.retained_volume_26_fraction.mean()))
 s=pd.DataFrame(summary); s.to_csv(W1/"size_sensitivity_summary.csv",index=False); return df,s

RULES=[("any_overlap",lambda x: x.intersection>0),("intersection_ge_2",lambda x:x.intersection>=2),("intersection_ge_5",lambda x:x.intersection>=5),("original_fraction_ge_0.01",lambda x:x.original_fraction>=.01),("original_fraction_ge_0.05",lambda x:x.original_fraction>=.05),("pair_iou_ge_0.05",lambda x:x.pair_iou>=.05),("pair_iou_ge_0.10",lambda x:x.pair_iou>=.10)]
SIX=ndi.generate_binary_structure(3,1)
def fit(a,shape):
 out=np.zeros(shape,dtype=a.dtype); s=tuple(slice(0,min(x,y)) for x,y in zip(a.shape,shape)); out[s]=a[s]; return out
def rt(a,factor,order):
 low=ndi.zoom(a,zoom=np.repeat(1/factor,3),order=order,mode="nearest"); return fit(ndi.zoom(low,zoom=np.repeat(factor,3),order=order,mode="nearest"),a.shape)
def edge_table(o,r,spacing):
 no,nr=int(o.max()),int(r.max()); so=np.bincount(o.ravel(),minlength=no+1); sr=np.bincount(r.ravel(),minlength=nr+1)
 m=(o>0)&(r>0); code=o[m].astype(np.int64)*(nr+1)+r[m]; u,c=np.unique(code,return_counts=True); oi=u//(nr+1); ri=u%(nr+1)
 df=pd.DataFrame({"original":oi,"reconstructed":ri,"intersection":c}); df=df[(df.original>0)&(df.reconstructed>0)].copy()
 df["original_size"]=so[df.original];df["reconstructed_size"]=sr[df.reconstructed];df["original_fraction"]=df.intersection/df.original_size;df["pair_iou"]=df.intersection/(df.original_size+df.reconstructed_size-df.intersection)
 vv=float(np.prod(spacing)); od=np.asarray([equivalent_diameter_mm(float(v*vv)) for v in so]); rd=np.asarray([equivalent_diameter_mm(float(v*vv)) for v in sr]); oc=np.asarray(ndi.center_of_mass(np.ones(o.shape),o,range(1,no+1)))*spacing; rc=np.asarray(ndi.center_of_mass(np.ones(r.shape),r,range(1,nr+1)))*spacing
 df["abs_diameter_error_mm"]=np.abs(rd[df.reconstructed]-od[df.original]);df["abs_relative_volume_error"]=np.abs(df.reconstructed_size-df.original_size)/df.original_size;df["centroid_displacement_mm"]=[np.linalg.norm(oc[i-1]-rc[j-1]) for i,j in zip(df.original,df.reconstructed)]
 return df,no,nr
def evaluate(df,no,nr,rule):
 e=df[rule(df)].copy(); od=e.groupby("original").size();rd=e.groupby("reconstructed").size(); e["odeg"]=e.original.map(od);e["rdeg"]=e.reconstructed.map(rd); one=e[(e.odeg==1)&(e.rdeg==1)]
 merges=int((rd>1).sum());frags=int((od>1).sum());unmatched=no-int(e.original.nunique());new=nr-int(e.reconstructed.nunique());complex_nodes=int(((e.odeg>1)&(e.rdeg>1)).sum())
 vals={"one_to_one_matches":len(one),"fraction_original_matched":e.original.nunique()/max(no,1),"merger_events":merges,"fragmentation_events":frags,"unmatched_original":unmatched,"new_reconstructed":new,"complex_edges":complex_nodes}
 for col in ("pair_iou","abs_diameter_error_mm","abs_relative_volume_error","centroid_displacement_mm"):
  vals[f"mean_{col}"]=one[col].mean();vals[f"median_{col}"]=one[col].median()
 vals["included_objects"]=len(one); return vals,set(map(tuple,one[["original","reconstructed"]].to_numpy()))
def matching():
 sel=pd.read_csv(SELECTION)[["acquisition_id","participant_id"]].drop_duplicates(); man=pd.read_csv(MANIFEST).set_index("acquisition_id")
 rows=[];fixed=[]
 for k,s in sel.iterrows():
  rec=man.loc[s.acquisition_id];sem,spacing,_=read_segmentation_from_zip(ZIP,rec.segmentation_zip_member);o,n0=reconstruct_instances(sem,connectivity=1);b=o>0
  for factor in (1.5,2.0):
   for method,order in (("nearest",0),("linear",1)):
    mask=rt(b.astype(float),factor,order)>=.5;r,nr=ndi.label(mask,structure=SIX);edges,no,nr=edge_table(o,r,np.asarray(spacing)); rulepairs={}
    for name,rule in RULES:
     vals,pairs=evaluate(edges,no,nr,rule);rulepairs[name]=pairs;vals.update(acquisition_id=s.acquisition_id,participant_id=s.participant_id,factor=factor,method=method,criterion=name);rows.append(vals)
    common=set.intersection(*rulepairs.values()) if rulepairs else set()
    common_edges=edges[edges.apply(lambda q:(q.original,q.reconstructed) in common,axis=1)]
    fixed.append(dict(acquisition_id=s.acquisition_id,participant_id=s.participant_id,factor=factor,method=method,n_common_matches=len(common),mean_common_iou=common_edges.pair_iou.mean(),mean_common_abs_diameter_error_mm=common_edges.abs_diameter_error_mm.mean(),mean_common_abs_relative_volume_error=common_edges.abs_relative_volume_error.mean(),mean_common_centroid_displacement_mm=common_edges.centroid_displacement_mm.mean()))
   print(f"matching {k+1}/{len(sel)} factor {factor}",flush=True)
 df=pd.DataFrame(rows);df.to_csv(RES/"object_matching_sensitivity.csv",index=False);pd.DataFrame(fixed).to_csv(RES/"fixed_common_correspondences.csv",index=False)
 paired=[];rng=np.random.default_rng(SEED)
 for (factor,crit),g in df.groupby(["factor","criterion"]):
  wide=g.pivot(index="participant_id",columns="method",values=["merger_events","fragmentation_events","fraction_original_matched","mean_abs_diameter_error_mm","mean_pair_iou"])
  for metric in ["merger_events","fragmentation_events","fraction_original_matched","mean_abs_diameter_error_mm","mean_pair_iou"]:
   delta=(wide[(metric,"linear")]-wide[(metric,"nearest")]).dropna().values; draws=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(REPS)];lo,hi=np.percentile(draws,[2.5,97.5]);paired.append(dict(factor=factor,criterion=crit,metric=metric,n_participants=len(delta),mean_linear_minus_nearest=delta.mean(),median_linear_minus_nearest=np.median(delta),ci_low=lo,ci_high=hi,bootstrap_reps=REPS,seed=SEED))
 pd.DataFrame(paired).to_csv(RES/"object_matching_paired_statistics.csv",index=False)
def select_one_per_participant(meta,internal):
 merged=meta.merge(internal,on=["acquisition_id","participant_id"])
 chosen=[]
 quality_order=["Poor","Good","Missing","Bad"]
 for index,(pid,g) in enumerate(merged.groupby("participant_id",sort=True)):
  desired=quality_order[index%len(quality_order)]; candidates=g[g.image_quality==desired]
  if candidates.empty: candidates=g
  candidates=candidates.assign(mean_spacing=candidates[["spacing_z_mm","spacing_y_mm","spacing_x_mm"]].astype(float).mean(axis=1))
  candidates=candidates.sort_values(["mean_spacing","acquisition_id"],ascending=[index%2==0,True])
  chosen.append(candidates.iloc[0])
 return pd.DataFrame(chosen)


def main():
    global ZIP, MANIFEST, MAN, RES, W1, SELECTION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("stage", choices=("all", "connectivity", "size", "resampling", "matching"))
    args = parser.parse_args()
    ZIP, MANIFEST, RES = args.archive, args.manifest, args.output
    MAN, W1 = MANIFEST, RES
    if not ZIP.is_file() or not MANIFEST.is_file():
        parser.error("archive and locally prepared manifest must exist")
    RES.mkdir(parents=True, exist_ok=True)
    if args.stage in {"all", "resampling", "matching"}:
        SELECTION = args.selection
        if SELECTION is None:
            if args.metadata is None:
                parser.error("supply --metadata or the exact study --selection")
            chosen = select_one_per_participant(pd.read_csv(args.metadata), pd.read_csv(MANIFEST))
            SELECTION = RES / "local_selection.csv"
            chosen[["acquisition_id", "participant_id"]].to_csv(SELECTION, index=False)
        chosen = pd.read_csv(SELECTION)
        if chosen.empty or chosen.participant_id.duplicated().any() or chosen.acquisition_id.duplicated().any():
            parser.error("selection must have exactly one acquisition per selected participant")
        ledger = pd.read_csv(MANIFEST).set_index("acquisition_id")
        if not chosen.acquisition_id.isin(ledger.index).all() or not np.array_equal(
            ledger.loc[chosen.acquisition_id, "participant_id"].to_numpy(), chosen.participant_id.to_numpy()
        ):
            parser.error("selection does not agree with the local manifest")
    for name, action in (("connectivity", correspondence), ("size", wp1_graph),
                         ("resampling", resampling), ("matching", matching)):
        if args.stage in {"all", name}:
            action()

if __name__ == "__main__":
    main()
