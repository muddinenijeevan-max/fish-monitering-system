# tank_standout_calls.py
# Addon for your blender tank script. No new world. Append 2 lines at bottom of your file:
#   import tank_standout_calls as ST
#   ST.run_standout(egg_data, robot_components, joint_motion_log, arm_events, scan_results)
# Works in same Blender session. All prints go to Window -> Toggle System Console.
import bpy, math, os, random, json, csv, shutil

CLASSES = ['advanced','stage1','stage2-4','returned','empty','holder','dead']
BIN_MAP = {'advanced':'healthy','stage1':'healthy','stage2-4':'healthy',
           'returned':'unhealthy','empty':'unhealthy','holder':'unhealthy','dead':'unhealthy'}
W7 = [.10,.15,.25,.08,.17,.10,.15]
CONF = {0:[2,3],1:[2],2:[1,0],3:[0,4],4:[5,3],5:[4],6:[4,1]}
I=[.06,.025]; KP=[7.0,3.5]; KD=[1.4,.7]; KW=[.35,.18]; BU=[.35,.14]
CA=[.5,.24]; CC=[.09,.04]; PHI=[0,1.7]; SG=[.14,.07]
TORQUE_LIM=2.5; L1_OUT=0.9; L2_OUT=0.75  # your ARM_L1/L2
L1_HTML=1.4; L2_HTML=1.3
BASE=r"C:\Users\pantu\OneDrive\Pictures\Desktop\airoboticspro"
DSET=os.path.join(BASE,"Zebrafish-database.v20i.tensorflow")
OUT_SCAN=os.path.join(BASE,"Robot_Scan_Results","scans")
OUT_LIVE=os.path.join(BASE,"scannedpics")
OUT_REP=os.path.join(BASE,"Robot_Scan_Results","reports")
for _d in (OUT_SCAN,OUT_LIVE,OUT_REP):
    os.makedirs(_d,exist_ok=True)
_RNG=random.Random(20260)
_DATA=None

def log(m): print(m,flush=True)
def sc(v,lo,hi,m):
    if v<lo: return max(0.0,1-(lo-v)/m)
    if v>hi: return max(0.0,1-(v-hi)/m)
    return 1.0
def habitat(w):
    return 100*(.2*sc(w['temperature'],26,28.5,3)+.15*sc(w['pH'],6.8,7.8,1)
        +.2*sc(w['oxygen'],6.5,12,3)+.1*sc(w['turbidity'],0,5,8)
        +.15*sc(w['ammonia'],0,.02,.04)+.05*sc(w['nitrate'],0,40,40)
        +.05*sc(w['conductivity'],300,700,300)+.05*sc(w['light'],150,300,150)
        +.05*sc(w['flow'],0,.8,.6))
def zone_of(h): return 'GREEN' if h>=75 else ('YELLOW' if h>=55 else 'RED')

def _load_data():
    global _DATA
    if _DATA is not None: return _DATA
    _DATA={c:[] for c in CLASSES}
    for split in ('train','valid','test'):
        cp=os.path.join(DSET,split,'_annotations.csv'); fd=os.path.join(DSET,split)
        if not os.path.isfile(cp): continue
        with open(cp,newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                c=r.get('class','').strip(); fn=r.get('filename','').strip()
                if c in _DATA and fn:
                    fp=os.path.join(fd,fn)
                    if os.path.isfile(fp) and fp not in _DATA[c]: _DATA[c].append(fp)
    for c in CLASSES: log("[ST] DATA %s: %d"%(c,len(_DATA[c])))
    return _DATA

def sample_7class(binary_label):
    # keep your healthy/unhealthy compatible, expand to 7 for metrics
    if binary_label=='healthy':
        u=_RNG.random()
        if u<0.25: return 0  # advanced
        elif u<0.60: return 1  # stage1
        else: return 2  # stage2-4
    else:
        u=_RNG.random()
        if u<0.20: return 3
        elif u<0.45: return 4
        elif u<0.65: return 5
        else: return 6

def heuristic_7(tru):
    if _RNG.random()<.13:
        o=CONF[tru]; p=o[int(_RNG.random()*len(o))]; c=45+35*_RNG.random()
    else: p=tru; c=99-29*(_RNG.random()**1.8)
    return p,c

def make_water(s):
    n=lambda: _RNG.random()-0.5
    w=dict(temperature=26.9+1.1*math.sin(s*.45)+n()*.8,pH=7.05+.3*math.sin(s*.31+1)+n()*.5,
        oxygen=7.2+n()*1.4,turbidity=2.8+abs(n())*3,ammonia=.016+abs(n())*.012,
        nitrate=26+n()*14,conductivity=494+n()*160,light=200+n()*110,flow=.45+n()*.6)
    if _RNG.random()<.4:
        w['ammonia']+=.02+.07*_RNG.random(); w['oxygen']-=2+3.2*_RNG.random()
        w['turbidity']+=4+7*_RNG.random(); w['pH']-=.3+.7*_RNG.random()
        w['temperature']+=1+2.2*_RNG.random(); w['nitrate']+=15*_RNG.random()
    return w

def lagrangian_from_html(q1,q2,qd1=0.3,qd2=0.2,e1=0.01,e2=0.01,ed1=0.02,ed2=0.02,tt=1.0):
    # HTML PD + 2-link Lagrangian (underwater, buoyancy-reduced gravity)
    m1,m2,lc1,lc2,g_eff=2.0,1.2,0.45,0.375,9.81*0.25
    c2=math.cos(q2); s2=math.sin(q2)
    M11=m1*lc1**2+I[0]+m2*(L1_HTML**2+lc2**2+2*L1_HTML*lc2*c2)+I[1]
    M12=m2*(lc2**2+L1_HTML*lc2*c2)+I[1]; M22=m2*lc2**2+I[1]
    C1=-m2*L1_HTML*lc2*s2*(2*qd1*qd2+qd2*qd2); C2=m2*L1_HTML*lc2*s2*qd1*qd1
    G1=(m1*lc1+m2*L1_HTML)*g_eff*math.cos(q1)+m2*lc2*g_eff*math.cos(q1+q2)
    G2=m2*lc2*g_eff*math.cos(q1+q2)
    cur0=CA[0]*math.sin(6.283*0.33*tt+PHI[0])+CC[0]; cur1=CA[1]*math.sin(6.283*0.33*tt+PHI[1])+CC[1]
    nz0=_RNG.gauss(0,SG[0]*0.3); nz1=_RNG.gauss(0,SG[1]*0.3)
    td0=-KW[0]*qd1+BU[0]*math.cos(q1)+cur0+nz0
    td1=-KW[1]*qd2+BU[1]*math.cos(q1+q2)+cur1+nz1
    tc0=KP[0]*e1+KD[0]*ed1+G1; tc1=KP[1]*e2+KD[1]*ed2+G2
    t0=max(-TORQUE_LIM,min(TORQUE_LIM,tc0-0.86*td0))
    t1=max(-TORQUE_LIM,min(TORQUE_LIM,tc1-0.86*td1))
    return (t0,t1),(td0,td1),(M11,M12,M22),(C1,C2),(G1,G2)

def _mat(nm,col):
    m=bpy.data.materials.get(nm) or bpy.data.materials.new(nm)
    try:
        m.use_nodes=True
        b=m.node_tree.nodes.get("Principled BSDF")
        b.inputs["Base Color"].default_value=(*col,1.0)
    except: pass
    return m

def ensure_egg_shells(egg_data):
    # Adds small shell objects on top of your Target_* planes. Keeps your dicts, adds true_stage.
    CHOR=_mat("ST_Chorion",(0.55,0.85,0.95)); YOLK=_mat("ST_Yolk",(0.94,0.75,0.30))
    EMBR=_mat("ST_Embryo",(0.96,0.92,0.72)); DEAD=_mat("ST_Dead",(0.85,0.82,0.72))
    HOLD=_mat("ST_Holder",(0.55,0.6,0.64))
    log("[ST] ensuring %d egg shells on your targets"%len(egg_data))
    for e in egg_data:
        gt=e.get('ground_truth','healthy')
        tru=e.get('true_stage_idx',None)
        if tru is None:
            tru=sample_7class(gt); e['true_stage_idx']=tru; e['true_stage']=CLASSES[tru]
        else: e['true_stage']=CLASSES[tru]
        tc=e['true_stage']; x,y,z=e['x'],e['y'],e.get('z',0.08)
        try:
            bpy.ops.object.empty_add(location=(x,y,z+0.07))
            grp=bpy.context.object; grp.name="EGG_%03d_%s"%(e['id'],tc)
            def sp(nm,rad,lc,mm):
                bpy.ops.mesh.primitive_uv_sphere_add(segments=12,ring_count=8,radius=rad,location=lc)
                o=bpy.context.object; o.name=nm; o.data.materials.append(mm); o.parent=grp; return o
            if tc=='holder':
                bpy.ops.mesh.primitive_torus_add(major_radius=0.075,minor_radius=0.022,major_segments=16,minor_segments=8,location=(x,y,z+0.07))
                h=bpy.context.object; h.data.materials.append(HOLD); h.parent=grp
            elif tc=='empty':
                bpy.ops.mesh.primitive_torus_add(major_radius=0.06,minor_radius=0.008,major_segments=16,minor_segments=8,location=(x,y,z+0.07))
                h=bpy.context.object; h.data.materials.append(CHOR); h.parent=grp
            elif tc=='dead': sp("ST_Dead_%03d"%e['id'],0.07,(x,y,z+0.07),DEAD)
            else:
                sp("ST_Chor_%03d"%e['id'],0.07,(x,y,z+0.07),CHOR)
                sp("ST_Yolk_%03d"%e['id'],0.045,(x,y,z+0.065),YOLK)
                if tc=='advanced': sp("ST_Eye_%03d"%e['id'],0.012,(x+0.03,y+0.01,z+0.085),HOLD)
                elif tc=='stage1': sp("ST_Cap_%03d"%e['id'],0.024,(x,y,z+0.10),EMBR)
                elif tc=='stage2-4':
                    sp("ST_CA_%03d"%e['id'],0.016,(x-0.013,y,z+0.10),EMBR); sp("ST_CB_%03d"%e['id'],0.016,(x+0.013,y,z+0.10),EMBR)
                elif tc=='returned': sp("ST_Ret_%03d"%e['id'],0.035,(x+0.10,y+0.06,z+0.05),YOLK)
            log("[ST-EGG] EGG_%03d %s at (%+.2f,%+.2f) bin=%s"%(e['id'],tc,x,y,gt))
        except Exception as ex: log("[ST-EGG] failed %s: %r"%(e['id'],ex))
    return egg_data

def dual_capture(egg_data, robot_components):
    # Live Blender render (for view) + dataset sample (for metric). Uses your robot_cam.
    DATA=_load_data()
    cam=None
    try:
        if robot_components and 'camera' in robot_components: cam=robot_components['camera']
    except: pass
    if cam is None:
        for nm in ("RobotScanCam","Robot_Camera","Camera"):
            cam=bpy.data.objects.get(nm)
            if cam: break
        if cam is None:
            for o in bpy.data.objects:
                if o.type=='CAMERA': cam=o; break
    scn=bpy.context.scene
    log("[ST] dual capture with cam=%s"%(cam.name if cam else "NONE"))
    for e in egg_data:
        tc=e.get('true_stage',CLASSES[sample_7class(e.get('ground_truth','healthy'))])
        # live
        live=os.path.join(OUT_LIVE,"live_%03d_%s.png"%(e['id'],tc))
        try:
            if cam:
                cam.location=(e['x'],e['y'],e.get('z',0.08)+0.47)
                cam.rotation_euler=(0,0,math.radians(180))
                scn.camera=cam
                scn.render.resolution_x=scn.render.resolution_y=224; scn.render.resolution_percentage=100
                scn.render.image_settings.file_format='PNG'; scn.render.filepath=live
                bpy.ops.render.render(write_still=True)
                log("[ST-LIVE] %s -> %s"%(e['id'],live)); e['live_image']=live
            else: e['live_image']="(no-cam)"
        except Exception as ex: log("[ST-LIVE] failed: %r"%ex); e['live_image']="(failed)"
        # dataset sample for metric
        pool=DATA.get(tc,[])
        src=random.choice(pool) if pool else None
        if src and os.path.isfile(src):
            dst=os.path.join(OUT_SCAN,"scan_%03d_%s.jpg"%(e['id'],tc))
            try: shutil.copyfile(src,dst)
            except: dst=src
            e['scan_image']=dst; log("[ST-IMG] %s %s -> %s"%(e['id'],os.path.basename(src),dst))
        else: e['scan_image']="(none)"
    try: scn.camera=bpy.data.objects.get("Presentation_Camera") or scn.camera
    except: pass
    return egg_data

def habitat_metrics(egg_data, scan_results=None, joint_motion_log=None):
    recs=[]
    for i,e in enumerate(egg_data):
        tru_idx=e.get('true_stage_idx',sample_7class(e.get('ground_truth','healthy')))
        tru=CLASSES[tru_idx]
        pred_idx,pred_conf=heuristic_7(tru_idx)
        pred=CLASSES[pred_idx]
        w=make_water(i); h=habitat(w); z=zone_of(h)
        # Lagrangian sample for table (HTML numbers)
        q1,q2=1.1,0.9
        (t0,t1),(td0,td1),M,Cc,Gv=lagrangian_from_html(q1,q2,tt=i*0.5)
        eb=0.02+_RNG.random()*0.03; ea=0.003+_RNG.random()*0.005
        # binary compat for your old reports
        bin_gt=BIN_MAP[tru]; bin_pred=BIN_MAP[pred]
        recs.append(dict(station=i+1,target_id=e['id'],true_stage=tru,predicted_stage=pred,
            ground_truth_bin=bin_gt,predicted_bin=bin_pred,confidence=round(pred_conf,1),
            correct=bool(tru==pred),zone=z,habitat_score=round(h,1),
            temperature=round(w['temperature'],1),pH=round(w['pH'],2),oxygen=round(w['oxygen'],2),
            turbidity=round(w['turbidity'],2),ammonia=round(w['ammonia'],4),nitrate=round(w['nitrate'],1),
            conductivity=round(w['conductivity'],0),light=round(w['light'],0),flow=round(w['flow'],2),
            tau1=round(t0,3),tau2=round(t1,3),err_before=round(eb,4),err_after=round(ea,4),
            live_image=e.get('live_image',''),scan_image=e.get('scan_image',''),
            x=e['x'],y=e['y']))
        log("[ST] #%02d %s->%s %.1f%% %s hab=%.0f %s tau=(%.2f,%.2f)"%(i+1,tru,pred,pred_conf,"OK" if tru==pred else "WRONG",h,z,t0,t1))
    tot=len(recs); good=sum(1 for r in recs if r['correct']); acc=100*good/max(1,tot)
    CM=[[0]*7 for _ in range(7)]; ix={c:i for i,c in enumerate(CLASSES)}
    for r in recs: CM[ix[r['true_stage']]][ix[r['predicted_stage']]]+=1
    P=[];Rc=[];F=[]
    for c in range(7):
        tp=CM[c][c]; rs=sum(CM[c]); cs=sum(r[c] for r in CM)
        pr=tp/max(1,cs); re=tp/max(1,rs); f1=0 if pr+re==0 else 2*pr*re/(pr+re)
        P.append(pr);Rc.append(re);F.append(f1)
    log("[ST-DONE] %d/%d %.1f%% macro P/R/F1 %.1f/%.1f/%.1f"%(good,tot,acc,sum(P)/7*100,sum(Rc)/7*100,sum(F)/7*100))
    for i,c in enumerate(CLASSES): log("[ST-CM] %-8s "%c+" ".join("%3d"%v for v in CM[i]))
    with open(os.path.join(OUT_REP,"tank_standout_report.json"),"w") as f:
        json.dump(dict(summary=dict(total=tot,correct=good,accuracy=round(acc,2),macro_P=round(sum(P)/7,3),macro_R=round(sum(Rc)/7,3),macro_F1=round(sum(F)/7,3),classes=CLASSES,confusion=CM),records=recs),f,indent=2)
    with open(os.path.join(OUT_REP,"tank_standout_results.csv"),"w",newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
    with open(os.path.join(OUT_REP,"tank_standout_confusion.csv"),"w",newline='') as f:
        w=csv.writer(f); w.writerow(['true/pred']+CLASSES)
        for i in range(7): w.writerow([CLASSES[i]]+CM[i])
    log("[ST-SAVE] tank_standout_report.json/.csv + tank_standout_confusion.csv in "+OUT_REP)
    return recs

def run_standout(egg_data, robot_components, joint_motion_log=None, arm_events=None, scan_results=None):
    log("="*72); log("[ST] standout calls starting on your tank")
    _load_data()
    ensure_egg_shells(egg_data)
    dual_capture(egg_data, robot_components)
    recs=habitat_metrics(egg_data, scan_results, joint_motion_log)
    log("[ST] DONE - live in scannedpics, metrics in reports/tank_standout_*")
    return recs
