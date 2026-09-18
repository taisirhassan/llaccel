#!/usr/bin/env python3
"""Run pinned ORFS synthesis Tcl with native host Yosys, then cached-netlist ORFS.

The environment is captured from a live ORFS synthesis process, filtered to
ORFS-defined variables. All flow scripts/platform files are copied unchanged.
The native tool versions and immutable inputs are recorded separately from the
Docker physical toolchain. --prepare-only runs canonicalization, not mapping.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'openroad/orfs@sha256:d4598d07ce4dbdbed3c1baaa98860f983c305d1141ce3fdc4a31da788c32db61'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def call(command, log, env=None):
    print(f'Running {log.name}', flush=True)
    with log.open('w') as stream:
        stream.write(json.dumps(list(map(str, command)))+'\n');stream.flush()
        subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n');temporary.replace(path)


def write_if_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def detail_route_profile(text, cores, reuse_pin_access):
    marker = 'source_step_tcl PRE DETAIL_ROUTE'
    route = '  log_cmd detailed_route {*}$all_args'
    if text.count(marker) != 1 or text.count(route) != 1:
        raise ValueError('pinned ORFS detailed route structure changed')
    if cores is not None:
        if cores <= 0:
            raise ValueError('detailed route cores must be positive')
        text = text.replace(marker, marker + f'\nset_thread_count {cores}')
    if reuse_pin_access:
        text = text.replace(route, '''  # pin_access from global routing persisted preferred access points in ODB.
  if {[lsearch -exact $all_args -no_pin_access] < 0} {
    lappend all_args -no_pin_access
  }
  puts "LLACCEL detailed route reuses database pin access; connectivity and DRC checks remain enabled"
''' + route)
    return text


def metrics_without_power(text):
    """Keep pinned ORFS timing/ERC/area reports while omitting optional power."""
    start = '  report_puts "$when report_power"'
    end = '  # TODO these only work to stdout'
    checkpoint = '  puts "Report metrics stage $stage, $when..."'
    if text.count(start) != 1 or text.count(end) != 1 or text.count(checkpoint) != 1:
        raise ValueError('pinned ORFS metrics structure changed')
    a, b = text.index(start), text.index(end)
    if a >= b:
        raise ValueError('pinned ORFS power block order changed')
    text = text[:a] + '  report_puts "$when power omitted: vectorless activity calculation disabled"\n\n' + text[b:]
    return text.replace(checkpoint, '''  # Preserve completed CTS geometry before potentially expensive reports.
  if { $stage == 4 } {
    orfs_write_db $::env(RESULTS_DIR)/4_cts_before_metrics.odb
    orfs_write_sdc $::env(RESULTS_DIR)/4_cts_before_metrics.sdc
  }
''' + checkpoint)


def global_route_with_checkpoint(text, checkpoint_key):
    """Save raw route segments with the database before repair can exhaust RAM."""
    marker = '''  if { ![do_global_route $res_aware $use_cugr] } {
    return
  }
'''
    if text.count(marker) != 1:
        raise ValueError('pinned ORFS global route structure changed')
    if not re_full_sha256(checkpoint_key):
        raise ValueError('checkpoint key must be a SHA-256 digest')
    return text.replace(marker, marker + '''
  file delete -force $::env(RESULTS_DIR)/5_before_repair.ready
  orfs_write_db $::env(RESULTS_DIR)/5_before_repair.odb
  orfs_write_sdc $::env(RESULTS_DIR)/5_before_repair.sdc
  write_global_route_segments $::env(RESULTS_DIR)/5_before_repair.segments
  set checkpoint_file [open $::env(RESULTS_DIR)/5_before_repair.ready w]
  puts $checkpoint_file "''' + checkpoint_key + '''"
  close $checkpoint_file
''')


def re_full_sha256(value):
    return len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def global_route_resume(text):
    load = 'load_design 4_cts.odb 4_cts.sdc'
    route = '''  log_cmd pin_access {*}$additional_args

  if { ![do_global_route $res_aware $use_cugr] } {
    return
  }'''
    if text.count(load) != 1 or text.count(route) != 1:
        raise ValueError('pinned ORFS route resume structure changed')
    return text.replace(load, 'load_design 5_before_repair.odb 5_before_repair.sdc').replace(
        route, '  log_cmd read_global_route_segments $::env(RESULTS_DIR)/5_before_repair.segments')


def verify_route_checkpoint(record_path, memory_dir, mapped, sdc):
    record = json.loads(record_path.read_text())
    key = record['checkpoint_key']
    checkpoint = record['profile']['global_route_checkpoint']
    if not re_full_sha256(key) or checkpoint['key'] != key or (memory_dir/'5_before_repair.ready').read_text().strip() != key:
        raise ValueError('route checkpoint completion key mismatch')
    computed_key = hashlib.sha256(json.dumps(checkpoint['inputs'], sort_keys=True).encode()).hexdigest()
    if computed_key != key:
        raise ValueError('route checkpoint changed provenance')
    for label, path in [('mapped_sha256', mapped), ('sdc_sha256', sdc), ('cts_sha256', memory_dir/'4_cts.odb')]:
        if checkpoint['inputs'][label] != digest(path):
            raise ValueError(f'route checkpoint changed upstream input: {label}')
    recorded_files = {(ROOT/Path(name)).resolve(): value for name, value in record['files'].items()}
    for suffix in ['odb', 'sdc', 'segments']:
        path = memory_dir/f'5_before_repair.{suffix}'
        expected = recorded_files.get(path.resolve(), {})
        if expected.get('sha256') != digest(path) or expected.get('bytes') != path.stat().st_size:
            raise ValueError(f'route checkpoint changed artifact: {suffix}')
    return key


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot',required=True)
    ap.add_argument('--variant',choices=['v1','v2'],default='v1')
    ap.add_argument('--container',default='cool_hertz',help='live pinned-image ORFS synthesis environment source')
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--single-pass-abc',action='store_true',help='standard-cell mapping without optional SAT choices/repeated LUT rewriting')
    ap.add_argument('--skip-adder-extraction',action='store_true',help='skip expensive optional FA recognition; ABC maps ordinary logic gates')
    ap.add_argument('--reuse-mapped',action='store_true',help='validate and reuse a prior native netlist for physical-stage retry')
    ap.add_argument('--repair-max-iterations',type=int,help='explicit diagnostic cap per physical timing-repair call; clock unchanged')
    ap.add_argument('--skip-physical-lec',action='store_true',help='disable optional physical Kepler equivalence check after tool incompatibility; no formal proof claimed')
    ap.add_argument('--skip-vectorless-power',action='store_true',help='retain timing/ERC/area reports but omit expensive default-activity power estimates')
    ap.add_argument('--physical-cores',type=int,default=3,help='threads for the current Docker physical invocation; reused checkpoints retain their prior settings')
    ap.add_argument('--checkpoint-global-route',action='store_true',help='save ODB, SDC and raw route segments before post-route repair')
    ap.add_argument('--resume-global-route-checkpoint',type=Path,help='resume a hash-verified pre-repair checkpoint manifest')
    ap.add_argument('--skip-post-route-repair',action='store_true',help='omit optional incremental sizing/timing repair after global route; diagnostic flow, not timing closure')
    ap.add_argument('--detail-route-cores',type=int,help='separate memory-conscious thread count for detailed routing')
    ap.add_argument('--reuse-pin-access',action='store_true',help='reuse pin access persisted by completed global routing; retain detailed-route connectivity/DRC checks')
    ap.add_argument('--physical',action='store_true',help='run Docker ORFS finish using the native mapped netlist')
    args=ap.parse_args()
    if args.physical_cores <= 0:
        ap.error('--physical-cores must be positive')
    if args.detail_route_cores is not None and args.detail_route_cores <= 0:
        ap.error('--detail-route-cores must be positive')
    if args.repair_max_iterations is not None and args.repair_max_iterations <= 0:
        ap.error("--repair-max-iterations must be positive")
    if not all(c.isalnum() or c in '_-' for c in args.snapshot):ap.error('invalid snapshot name')
    snapshot=ROOT/'build/synth'/args.snapshot
    netlist=snapshot/f'llaccel_core_{args.variant}.v'
    if not netlist.is_file():ap.error(f'missing immutable input {netlist}')
    run_suffix=('-no-fa' if args.skip_adder_extraction else '')+('-single-pass' if args.single_pass_abc else '')
    run_variant=args.variant+run_suffix
    work=ROOT/'build/native-orfs'/args.snapshot/run_variant
    work.mkdir(parents=True,exist_ok=True)
    previous=json.loads((work/'manifest.json').read_text()) if (work/'manifest.json').is_file() else {}
    assets=ROOT/'build/native-orfs/assets'
    assets.mkdir(parents=True,exist_ok=True)
    captured=assets/'synth-environment.json'
    if not captured.is_file():
        inspect=json.loads(subprocess.check_output(['docker','inspect',args.container]))[0]
        if inspect['Config']['Image'] != IMAGE:
            raise ValueError('environment source is not the pinned ORFS image')
        for relative in ['scripts','platforms/nangate45','platforms/common']:
            target=assets/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                subprocess.run(['docker','cp',f'{args.container}:/OpenROAD-flow-scripts/flow/{relative}',str(target)],check=True)
        capture=r'''
import json,pathlib
variables=json.loads(pathlib.Path('/OpenROAD-flow-scripts/flow/scripts/variables.json').read_text())
allowed=set(variables)|{'SCRIPTS_DIR','UTILS_DIR','FLOW_HOME','PLATFORM_DIR','RESULTS_DIR','REPORTS_DIR','OBJECTS_DIR','LOG_DIR','PYTHON_EXE','YOSYS_EXE','SDC_FILE_CLOCK_PERIOD'}
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  command=(proc/'cmdline').read_bytes().split(b'\0')
  if not any(x.endswith(b'/yosys') for x in command) or b'-c' not in command:continue
  values=dict(item.decode().split('=',1) for item in (proc/'environ').read_bytes().split(b'\0') if b'=' in item)
  print(json.dumps({k:v for k,v in values.items() if k in allowed}));break
 except (OSError,ValueError):continue
else:raise RuntimeError('no active ORFS Yosys process found')
'''
        raw=subprocess.check_output(['docker','exec',args.container,'python3','-c',capture],text=True)
        save(captured,json.loads(raw))
        save(assets/'source.json',{'image':IMAGE,'container_image_id':inspect['Image']})
    # Common arithmetic maps are referenced by the platform's synth arguments.
    if not (assets/'platforms/common').exists():
        subprocess.run(['docker','cp',f'{args.container}:/OpenROAD-flow-scripts/flow/platforms/common',str(assets/'platforms/common')],check=True)
    env_values=json.loads(captured.read_text())
    for key,value in env_values.items():
        env_values[key]=value.replace('/OpenROAD-flow-scripts/flow',str(assets)).replace('/work/',str(ROOT)+'/')
    for key,subdir in [('RESULTS_DIR','results'),('REPORTS_DIR','reports'),('OBJECTS_DIR','objects'),('LOG_DIR','logs')]:
        path=work/subdir;path.mkdir(exist_ok=True);env_values[key]=str(path)
    env_values.update({'VERILOG_FILES':str(netlist),'SDC_FILE':str(snapshot/'constraint.sdc'),
        'SCRIPTS_DIR':str(assets/'scripts'),'PLATFORM_DIR':str(assets/'platforms/nangate45'),
        'FLOW_HOME':str(assets),'PYTHON_EXE':sys.executable,'YOSYS_EXE':shutil.which('yosys') or '',
        'DESIGN_NAME':'llaccel_core','DESIGN_NICKNAME':f'llaccel_core_{args.variant}','SYNTH_NETLIST_FILES':'','SDC_FILE_CLOCK_PERIOD':str(work/'clock_period.txt')})
    if args.skip_adder_extraction:
        env_values['ADDER_MAP_FILE']=''
    # Read the exact simple clock constraint used by repository snapshots.
    import re
    match=re.search(r'^set clk_period\s+([0-9.]+)\s*$',(snapshot/'constraint.sdc').read_text(),re.M)
    if not match:raise ValueError('snapshot clock period cannot be established')
    (work/'clock_period.txt').write_text(match.group(1)+'\n')
    environment=os.environ.copy();environment.update(env_values)
    environment['PATH']=str(Path(sys.executable).parent)+os.pathsep+environment['PATH']
    driver=assets/'scripts/synth.tcl'
    driver_changes=[]
    if args.single_pass_abc:
        abc_script=work/'abc_single_pass.script'
        abc_script.write_text('&get -n\n&st\n&nf\n&put\nbuffer -c\ntopo\nstime -c\nupsize -c\ndnsize -c\n')
        env_values['LLACCEL_ABC_SCRIPT']=str(abc_script)
        environment['LLACCEL_ABC_SCRIPT']=str(abc_script)
        original=driver.read_text()
        needle='  log_cmd abc {*}$abc_args'
        if original.count(needle)!=1:raise ValueError('ORFS ABC driver structure changed')
        replacement='  set abc_args [lreplace $abc_args 0 1 -script $::env(LLACCEL_ABC_SCRIPT)]\n  write_rtlil $::env(RESULTS_DIR)/preabc.rtlil\n'+needle
        driver=work/'synth_single_pass.tcl'
        driver.write_text(original.replace(needle,replacement))
        driver_changes=['Override only ABC script path; retain ORFS synthesis/library mapping/checks',
                        'Save pre-ABC RTLIL checkpoint',
                        'Skip SAT choice sweeping and five repeated LUT rewrite/remap sequences; retain ordinary standard-cell mapping and buffer/sizing']
    save(work/'environment.json',env_values)
    manifest={'status':'RUNNING','snapshot':args.snapshot,'variant':args.variant,'image':IMAGE,
        'full_adder_extraction':not args.skip_adder_extraction,
        'single_pass_abc':args.single_pass_abc, 'driver_changes':driver_changes,
        'driver_sha256':digest(driver),
        'abc_script_sha256':digest(work/'abc_single_pass.script') if args.single_pass_abc else digest(assets/'scripts/abc_speed.script'),
        'environment_sha256':digest(work/'environment.json'),
        'native_yosys':subprocess.check_output(['yosys','-V'],text=True).strip(),
        'native_yosys_sha256':digest(Path(shutil.which('yosys')).resolve()),
        'native_abc_sha256':digest(Path(shutil.which('yosys-abc')).resolve()),
        'inputs':{netlist.name:digest(netlist),'constraint.sdc':digest(snapshot/'constraint.sdc')},
        'flow_files':{str(p.relative_to(assets)):digest(p) for p in assets.rglob('*') if p.is_file()}}
    mapped=work/'results/1_2_yosys.v'
    if args.reuse_mapped:
        for key in ['inputs','flow_files','environment_sha256','native_yosys_sha256','native_abc_sha256','driver_sha256','abc_script_sha256']:
            if previous.get(key)!=manifest[key]:raise ValueError(f'cannot reuse changed native synthesis {key}')
        if not mapped.is_file() or previous.get('mapped_sha256')!=digest(mapped):
            raise ValueError('cannot reuse missing or changed mapped netlist')
    save(work/'manifest.json',manifest)
    try:
        if not args.reuse_mapped:
            call(['yosys','-T','-l',work/'logs/canonicalize.log','-c',assets/'scripts/synth_canonicalize.tcl'],work/'canonicalize-console.log',environment)
            if args.prepare_only:
                manifest['status']='PREPARED';save(work/'manifest.json',manifest);return
            call(['yosys','-T','-l',work/'logs/synth.log','-c',driver],work/'synth-console.log',environment)
            mapped=work/'results/1_2_yosys.v'
        if not mapped.is_file():raise ValueError('native synthesis produced no mapped netlist')
        manifest.update(status='SYNTHESIZED',mapped_sha256=digest(mapped))
        save(work/'manifest.json',manifest)
        if args.physical:
            physical=ROOT/'build/orfs'/(f'{args.snapshot}-native'+run_suffix)
            physical.mkdir(parents=True,exist_ok=True)
            memory_dir=physical/'results/nangate45'/f'llaccel_core_{args.variant}'/'base'
            memory_dir.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(work/'results/mem.json',memory_dir/'mem.json')
            cached=' '.join('/work/'+str(p.relative_to(ROOT)) for p in (work/'reports').glob('synth_*.txt'))
            hook=ROOT/'synth/orfs/bounded_repair.tcl'
            hook_args=''
            if args.repair_max_iterations is not None:
                hook_path='/work/'+str(hook.relative_to(ROOT))
                hook_args=' LLACCEL_REPAIR_MAX_ITERATIONS='+str(args.repair_max_iterations)
                hook_args+=' '+ ' '.join(f'PRE_{stage}_TCL={hook_path}' for stage in ['FLOORPLAN','RESIZE','CTS','GLOBAL_ROUTE'])
            if args.skip_physical_lec:
                hook_args+=' LEC_CHECK=0'
            if args.skip_post_route_repair:
                hook_args+=' SKIP_INCREMENTAL_REPAIR=1 RECOVER_POWER=0'
            manifest['physical_profile']={'repair_max_iterations':args.repair_max_iterations,
                'post_route_incremental_repair':not args.skip_post_route_repair,
                'detail_route_cores':args.detail_route_cores or args.physical_cores,
                'reuse_pin_access':args.reuse_pin_access,
                'num_cores':args.physical_cores,
                'num_cores_scope':'Current invocation; reused checkpoints retain creation settings recorded in stage logs',
                'vectorless_power':not args.skip_vectorless_power,
                'repair_limit_scope':'OpenROAD max_iterations limits setup repair; hold repair retains tool-native limits',
                'lec_check':False if args.skip_physical_lec else None,
                'formal_equivalence':'Not established; optional Kepler child failed with illegal instruction' if args.skip_physical_lec else 'Tool-default optional check; consult actual stage logs',
                'hook_sha256':digest(hook) if args.repair_max_iterations is not None else None,
                'description':('Bounded diagnostic timing repair; original SDC unchanged; not timing closure'
                    if args.repair_max_iterations is not None else 'Default ORFS timing-repair iteration limits')}
            metrics_mount=[]
            if args.skip_vectorless_power:
                original_metrics=assets/'scripts/report_metrics.tcl'
                profiled_metrics=work/'report_metrics_no_power.tcl'
                write_if_changed(profiled_metrics,metrics_without_power(original_metrics.read_text()))
                metrics_mount=['-v',f'{profiled_metrics}:/OpenROAD-flow-scripts/flow/scripts/report_metrics.tcl:ro']
                manifest['physical_profile']['metrics_original_sha256']=digest(original_metrics)
                manifest['physical_profile']['metrics_profile_sha256']=digest(profiled_metrics)
            if args.resume_global_route_checkpoint:
                key=verify_route_checkpoint(args.resume_global_route_checkpoint,memory_dir,mapped,snapshot/'constraint.sdc')
                route_script=work/'global_route_resume.tcl'
                write_if_changed(route_script,global_route_resume((assets/'scripts/global_route.tcl').read_text()))
                metrics_mount+=['-v',f'{route_script}:/OpenROAD-flow-scripts/flow/scripts/global_route.tcl:ro']
                manifest['physical_profile']['resumed_route_checkpoint']={
                    'key':key,'manifest_sha256':digest(args.resume_global_route_checkpoint),
                    'script_sha256':digest(route_script)}
            elif args.checkpoint_global_route:
                original_route=assets/'scripts/global_route.tcl'
                checkpoint_inputs={'mapped_sha256':digest(mapped),
                    'sdc_sha256':digest(snapshot/'constraint.sdc'),
                    'cts_sha256':digest(memory_dir/'4_cts.odb') if (memory_dir/'4_cts.odb').exists() else None,
                    'profile':dict(manifest['physical_profile']),
                    'global_route_original_sha256':digest(original_route)}
                key=hashlib.sha256(json.dumps(checkpoint_inputs,sort_keys=True).encode()).hexdigest()
                route_script=work/'global_route_checkpoint.tcl'
                write_if_changed(route_script,global_route_with_checkpoint(original_route.read_text(),key))
                metrics_mount+=['-v',f'{route_script}:/OpenROAD-flow-scripts/flow/scripts/global_route.tcl:ro']
                manifest['physical_profile']['global_route_checkpoint']={
                    'key':key,'inputs':checkpoint_inputs,'script_sha256':digest(route_script)}
            if args.detail_route_cores is not None or args.reuse_pin_access:
                original_detail=assets/'scripts/detail_route.tcl'
                detail_script=work/'detail_route_profile.tcl'
                write_if_changed(detail_script,detail_route_profile(original_detail.read_text(),args.detail_route_cores,args.reuse_pin_access))
                metrics_mount+=['-v',f'{detail_script}:/OpenROAD-flow-scripts/flow/scripts/detail_route.tcl:ro']
                manifest['physical_profile']['detail_route_original_sha256']=digest(original_detail)
                manifest['physical_profile']['detail_route_profile_sha256']=digest(detail_script)
            save(physical/'physical-profile.json',manifest['physical_profile'])
            save(work/'manifest.json',manifest)
            command=['docker','run','--rm','--platform','linux/amd64',*metrics_mount,'-v',f'{ROOT}:/work',
                '-e',f'LL_NATIVE_NETLIST=/work/{mapped.relative_to(ROOT)}',
                '-e',f'LL_WORK=/work/{physical.relative_to(ROOT)}',
                '-e',f'LL_SDC=/work/{(snapshot/"constraint.sdc").relative_to(ROOT)}',
                '-e',f'LL_CACHED_REPORTS={cached}',
                '-e',f'LL_CONFIG=/work/synth/orfs/config_{args.variant}.mk',IMAGE,'bash','-lc',
                'set -euo pipefail; source /OpenROAD-flow-scripts/env.sh; cd /OpenROAD-flow-scripts/flow; '
                'make DESIGN_CONFIG="$LL_CONFIG" SYNTH_NETLIST_FILES="$LL_NATIVE_NETLIST" CACHED_REPORTS="$LL_CACHED_REPORTS" '
                'SDC_FILE="$LL_SDC" WORK_HOME="$LL_WORK" NUM_CORES='+str(args.physical_cores)+hook_args+' finish']
            call(command,work/'physical-console.log')
            call([sys.executable,ROOT/'scripts/collect_ppa.py',args.variant,'--base',physical,
                  '--require-finish','--json',work/'ppa.json'],work/'ppa-console.log')
            manifest['ppa_sha256']=digest(work/'ppa.json')
            manifest['status']='PHYSICAL_FINISHED';manifest['physical_directory']=str(physical)
            save(work/'manifest.json',manifest)
    except BaseException as error:
        manifest.update(status='FAILED',error=f'{type(error).__name__}: {error}')
        save(work/'manifest.json',manifest);raise


if __name__=='__main__':main()
